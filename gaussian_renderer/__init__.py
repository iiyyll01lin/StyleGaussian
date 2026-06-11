#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
# >>> stylegaussian_patches:gsplat_backend (BEGIN — applied by apply_patches.py)
import os as _sg_os
if _sg_os.environ.get('STYLEGAUSSIAN_BACKEND', 'gsplat') == 'original':
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from feature_gaussian_rasterization import GaussianRasterizer as FeatureGaussianRasterizer
else:
    from stylegaussian_patches.gsplat_backend import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        FeatureGaussianRasterizer,
    )
# <<< stylegaussian_patches:gsplat_backend (END)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, SH2RGB

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, feature_linear = None, feature_override = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if feature_linear is None else int(viewpoint_camera.feature_height),
        image_width=int(viewpoint_camera.image_width) if feature_linear is None else int(viewpoint_camera.feature_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    if feature_linear is None:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    else: 
        rasterizer = FeatureGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if feature_linear is None:
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color
    else:
        # Feature path. Default splats pc._vgg_features (the VGG style field).
        # `feature_override` (Eval 2) lets a caller splat a DIFFERENT per-Gaussian
        # field — the CLIP language field `_clip_features` for train_language.py —
        # without disturbing the VGG behaviour; None preserves the upstream path
        # exactly, so existing feature training / inference are byte-identical.
        colors_precomp = pc._vgg_features if feature_override is None else feature_override # [N, low_dim]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if feature_linear is None:
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, sum_w, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image = feature_linear(rendered_image, sum_w) # [out_dim, H, W]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    #
    # gsplat backend: the rasterizer stashes its internal screen-space means
    # tensor (which receives .absgrad/.grad only AFTER backward) onto our
    # screenspace_points placeholder; return that as viewspace_points so
    # densification reads a populated gradient. original (INRIA) backend: the
    # attribute is absent, so we keep the placeholder whose .grad INRIA fills.
    viewspace_points = getattr(screenspace_points, "sg_gsplat_means2d", screenspace_points)
    return {"render": rendered_image,
            "viewspace_points": viewspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


# =============================================================================
# View-dependent stylized SH color synthesis (Evaluation 1)
#
# The functions below are deliberately pure (torch + eval_sh + SH2RGB only): no
# CUDA, no GaussianModel/camera objects, no in-place state. That keeps the core
# math unit-testable on CPU (reimpl/tests/test_view_dependent_sh.py). The two
# `*_color`-on-model wrappers (`view_directions`, `stylized_view_dependent_color`)
# adapt a GaussianModel + camera to these pure kernels and are used by render.py
# and train_artistic.py. render() itself is untouched — callers precompute the
# per-view `override_color` and pass it through the existing flat path, so with
# --view_dependent OFF nothing here ever runs.
# =============================================================================

def sh_to_rgb_residual(shs, dirs, deg):
    """Per-Gaussian view-dependent RGB residual from reconstruction SH.

    Args:
        shs:  [..., 3, (D+1)**2] full SH coeffs (DC at index 0, 'rest' after).
        dirs: [..., 3]           per-Gaussian unit view directions.
        deg:  int                SH degree to evaluate (<= D).

    Returns ``full_rgb - dc_rgb`` = the purely directional (view-dependent) part:

        full_rgb = eval_sh(deg, shs, dirs) + 0.5   (the 3DGS SH->RGB convention)
        dc_rgb   = SH2RGB(shs[..., 0]) = C0*dc + 0.5  (view-independent base)

    The constant +0.5 bias cancels in the difference, so this is identically
    **zero when the higher-order ('rest') coeffs are zero** — i.e. a flat-color
    Gaussian contributes no view dependence.
    """
    dc = shs[..., 0]                              # [..., 3]
    full_rgb = eval_sh(deg, shs, dirs) + 0.5      # reconstruction SH -> RGB
    dc_rgb = SH2RGB(dc)                           # = C0*dc + 0.5 (view-independent)
    return full_rgb - dc_rgb


def synthesize_view_dependent_color(base_rgb, shs, dirs, deg, residual_scale=1.0):
    """Variant A: stylized per-view color = clamp(base + scale*residual, 0, 1).

    ``base_rgb`` [..., 3] is the (view-independent) stylized decoder output; the
    residual is the geometry-grounded view dependence carried from reconstruction.
    With ``residual_scale == 0`` (or all-zero 'rest' coeffs) this reproduces the
    clamped flat output exactly.
    """
    residual = sh_to_rgb_residual(shs, dirs, deg)
    return torch.clamp(base_rgb + residual_scale * residual, 0.0, 1.0)


def decoder_sh_to_color(decoder_sh, dirs, deg):
    """Variant B: evaluate decoder-predicted per-Gaussian SH coeffs for one view.

    Args:
        decoder_sh: [N, 3*(deg+1)**2] raw SH head output, reshaped channel-major
                    to [N, 3, (deg+1)**2] (R coeffs, then G, then B).
        dirs:       [N, 3] unit view directions.
        deg:        int SH degree.

    Returns clamped RGB [N, 3] = clamp(eval_sh(deg, shs, dirs) + 0.5, 0, 1).
    """
    n = decoder_sh.shape[0]
    coeffs = (deg + 1) ** 2
    shs = decoder_sh.view(n, 3, coeffs)
    rgb = eval_sh(deg, shs, dirs) + 0.5
    return torch.clamp(rgb, 0.0, 1.0)


def sh_consistency_regularizer(decoder_sh, deg):
    """Variant B consistency regularizer (scalar): mean square of the higher-order
    (non-DC) SH coeffs. Penalizing these suppresses per-view flicker, trading view
    dependence for the multi-view consistency this project is graded on. Returns 0
    when there are no higher-order coeffs (deg == 0)."""
    n = decoder_sh.shape[0]
    coeffs = (deg + 1) ** 2
    shs = decoder_sh.view(n, 3, coeffs)
    if coeffs <= 1:
        return decoder_sh.new_zeros(())
    return (shs[..., 1:] ** 2).mean()


def view_directions(pc, viewpoint_camera):
    """Per-Gaussian unit view directions (camera_center -> Gaussian), matching the
    dir_pp computation render() uses for SH evaluation."""
    n = pc.get_xyz.shape[0]
    dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(n, 1)
    return dir_pp / dir_pp.norm(dim=1, keepdim=True)


def stylized_view_dependent_color(pc, viewpoint_camera, decoder_out, mode="residual",
                                   residual_scale=1.0, sh_degree_style=None):
    """Synthesize a view-dependent ``override_color`` for one camera.

    mode == 'residual'   (Variant A): ``decoder_out`` is the [N,3] stylized base
        RGB; add the frozen reconstruction SH residual for this view. The residual
        depends only on frozen geometry/SH, so it is detached (no grad), keeping
        gradients flowing solely through the decoder output during training.
    mode == 'decoder_sh' (Variant B): ``decoder_out`` is the [N, 3*(deg+1)^2] raw
        SH head; evaluate it for this view (decoder_out keeps its gradient).
    """
    dirs = view_directions(pc, viewpoint_camera).detach()
    if mode == "residual":
        shs = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2).detach()
        return synthesize_view_dependent_color(decoder_out, shs, dirs, pc.max_sh_degree, residual_scale)
    if mode == "decoder_sh":
        deg = pc.max_sh_degree if sh_degree_style is None else sh_degree_style
        return decoder_sh_to_color(decoder_out, dirs, deg)
    raise ValueError(f"unknown view_dependent_mode: {mode!r} (expected 'residual' or 'decoder_sh')")


# =============================================================================
# Language-localized style blend (Evaluation 2, LangSplat-Lite)
#
# Pure torch (no CUDA / model state), so the blend math is unit-testable on CPU
# (reimpl/tests/test_language_field.py) and composes with BOTH the flat and the
# view-dependent (per-camera) stylized/original colors that render.py computes
# inside its camera loop. render.py owns the orchestration (load language ckpt ->
# final_clip_features -> compute_relevancy -> relevancy_to_mask -> this blend).
# =============================================================================

def photoreal_view_color(shs, dirs, deg):
    """Per-Gaussian photoreal reconstruction color for one view.

        clamp(eval_sh(deg, shs, dirs) + 0.5, 0, 1)

    ``shs`` is the reconstruction SH ``[N, 3, (deg+1)**2]`` (DC at index 0). This
    is the view-dependent ORIGINAL color the localized blend keeps outside the
    masked (stylized) region — the same SH->RGB convention render() uses, but
    clamped to [0,1] so it is a clean blend input. With only the DC term present
    it reduces to the flat albedo C0*dc + 0.5.
    """
    rgb = eval_sh(deg, shs, dirs) + 0.5
    return torch.clamp(rgb, 0.0, 1.0)


def localized_blend(mask, stylized_rgb, original_rgb):
    """Per-Gaussian language-localized style blend.

        final[n] = M[n] * stylized_rgb[n] + (1 - M[n]) * original_rgb[n]

    Args:
        mask:         [N] or [N, 1] soft mask in [0, 1] (1 -> stylize, 0 -> keep
                      the photoreal original). From relevancy_to_mask.
        stylized_rgb: [N, 3] the (possibly view-dependent) artistic color.
        original_rgb: [N, 3] the photoreal reconstruction color for this view.

    Returns the blended per-Gaussian color [N, 3]. It is a convex combination of
    two valid colors so — when both inputs are already in [0, 1] — the result is
    too; clamping is intentionally left to the caller / rasterizer post-step so
    the blend stays an exact linear interpolation (M == 0 -> original exactly,
    M == 1 -> stylized exactly).
    """
    m = mask.reshape(-1, 1)
    return m * stylized_rgb + (1.0 - m) * original_rgb
