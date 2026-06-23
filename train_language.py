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

"""Train the LangSplat-Lite CLIP language feature field (Evaluation 2).

A near-clone of ``train_feature.py``: it distills a frozen CLIP image tower's
dense grid embeddings (``Camera.clip_features``) into a per-Gaussian low-dim
field (``GaussianModel._clip_features``) plus a ``clip_linear`` decoder back to
CLIP space, exactly mirroring how the feature stage distills VGG ``relu3_1``.

Two small, fully-additive deltas vs the VGG feature stage:
  1. The render feature path hard-codes ``colors_precomp = pc._vgg_features``,
     so we splat the CLIP field via the new ``render(..., feature_override=...)``
     hook instead (None preserves the VGG path).
  2. The CLIP GT grid is much smaller than a VGG conv map, so we render at the
     CLIP dims by setting ``cam.feature_height/width = cam.clip_feature_*`` per
     train camera before the loop.

ALIGNMENT (important): pass the SAME ``recon.ply`` that the feature / artistic
stages were derived from (feature/artistic never densify, so the per-Gaussian
order is identical). The localized blend in ``render.py`` then asserts the
language model's xyz / point-count matches the artistic model's.

The saved checkpoint goes to ``output/<scene>/language/<exp>/chkpnt/language.pth``
and is loaded by ``render.py --text_query`` for language-localized stylization.
This script is brand new; nothing in the existing pipeline imports it, so the
reconstruction / feature / artistic / render paths are untouched.
"""

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.clip_encoder import CLIPEncoder
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter


def training(dataset, opt, pipe, ply_path, debug_from, low_dim=32,
             clip_model="ViT-B-16", clip_pretrained="openai", clip_grid=8,
             distill_loss="l1", clip_input_res=224, gt_mode="maskclip",
             sam_checkpoint=None, sam_model_type="vit_b", sam_grid=14,
             sam_points_per_side=16, sam_cache_dir=None, sam_out_stride=4,
             clip_dense_mode="maskclip", ae_checkpoint=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # Frontend CLIP (Worker B's API). fp32 on gfx1151 (NEVER bf16). The Scene
    # precomputes cam.clip_features [D, H', W'] over the TRAIN cameras, mirroring
    # the VGG precompute loop. input_resolution drives the MaskCLIP patch grid
    # (ViT-B/16: res/16 per side, so 224->14x14, 448->28x28 = denser GT).
    #
    # Two GT producers, SAME [D, gh, gw] contract downstream:
    #   * gt_mode="maskclip" (default): MaskCLIP per-patch dense tokens (no SAM).
    #   * gt_mode="sam_pooled" (roadmap A1 fallback): SAM regions -> per-region
    #     CLIP -> painted into the coarse grid, for cleaner object masks.
    #   * gt_mode="sam_perpixel" (roadmap A): SAM regions -> per-region masked-crop
    #     CLIP -> painted PER-PIXEL onto a high-res H/stride x W/stride grid (sharp,
    #     object-shaped GT; passes the A0 GT-level truck-vs-road corr gate).
    if gt_mode in ("sam_pooled", "sam_perpixel"):
        from scene.sam_pooled_encoder import SAMPooledEncoder
        if not sam_checkpoint:
            raise ValueError(f"gt_mode={gt_mode} 需要 --sam_checkpoint <path>.")
        clip_encoder = SAMPooledEncoder(
            sam_checkpoint=sam_checkpoint, sam_model_type=sam_model_type,
            clip_model=clip_model, clip_pretrained=clip_pretrained,
            device="cuda", grid=(sam_grid, sam_grid), dtype=torch.float32,
            input_resolution=clip_input_res, points_per_side=sam_points_per_side,
            cache_dir=sam_cache_dir,
            per_pixel=(gt_mode == "sam_perpixel"), out_stride=sam_out_stride,
        )
    elif gt_mode == "langsplat_ae":
        # Option B: MaskCLIP dense grid -> per-scene autoencoder denoise (still
        # CLIP-512 output, so the field / clip_linear decoder / relevancy / eval
        # are all unchanged). The AE is fit offline by build_langsplat_ae.py and
        # projects each noisy MaskCLIP token back onto the scene CLIP manifold.
        from scene.langsplat_ae_encoder import LangSplatAEEncoder
        if not ae_checkpoint:
            raise ValueError("gt_mode=langsplat_ae 需要 --ae_checkpoint <path> "
                             "(先用 experiments/build_langsplat_ae.py 產生).")
        clip_encoder = LangSplatAEEncoder(
            ae_checkpoint=ae_checkpoint, clip_model=clip_model,
            clip_pretrained=clip_pretrained, device="cuda",
            input_resolution=clip_input_res, dtype=torch.float32,
            clip_grid=(clip_grid, clip_grid), dense_mode=clip_dense_mode,
        )
    elif gt_mode == "siglip_dense":
        # Phase-2 (stronger-VLM-signal): SigLIP2 dense GT via the MAP-head
        # value-bypass (繞過 latent-query pooling). Still a [D, gh, gw] grid in the
        # SigLIP joint space, so the field / clip_linear decoder / relevancy / eval
        # are unchanged — BUT eval MUST use the same SigLIP backbone as text tower.
        from scene.siglip_dense_encoder import SigLIPDenseEncoder
        clip_encoder = SigLIPDenseEncoder(
            model_name=clip_model, pretrained=clip_pretrained,
            device="cuda", input_resolution=clip_input_res, dtype=torch.float32,
        )
    else:
        clip_encoder = CLIPEncoder(
            model_name=clip_model, pretrained=clip_pretrained,
            device="cuda", grid=(clip_grid, clip_grid), dtype=torch.float32,
            input_resolution=clip_input_res, dense_mode=clip_dense_mode,
        )

    # Load the (shared, same-as-feature/artistic) reconstruction ply and bake the
    # CLIP grid GT into every train camera.
    scene = Scene(dataset, gaussians, load_path=ply_path, clip_encoder=clip_encoder)
    gaussians.training_setup_language(opt, low_dim=low_dim, clip_dim=clip_encoder.embed_dim)

    # The feature rasterizer renders at cam.feature_height/width; point it at the
    # CLIP grid dims so the rendered map lines up with the CLIP GT (the distill
    # loss is per-cell, so rendered H'xW' MUST equal the MaskCLIP gh x gw).
    for cam in scene.getTrainCameras():
        cam.feature_height = cam.clip_feature_height
        cam.feature_width = cam.clip_feature_width

    # Feature-splat background has `low_dim` channels (not 3). Mirrors the VGG
    # feature stage's [*]*low_dim background.
    bg_color = [1] * low_dim if dataset.white_background else [0] * low_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Language training", bar_format='{l_bar}{r_bar}')
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Splat the CLIP language field (feature_override) and decode with clip_linear.
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            feature_linear=gaussians.clip_linear,
                            feature_override=gaussians._clip_features)
        rendered_feature = render_pkg["render"]  # [D=clip_dim, H', W']

        # Loss against the CLIP grid GT (each cell L2-normalized along D).
        gt_feature = viewpoint_cam.clip_features  # [D, H', W']
        if distill_loss == "cosine":
            # CLIP features are directional; match the GT direction per cell
            # (L2-normalize the decoded render along D, GT is already unit) so
            # we distill semantic direction rather than L1 magnitude.
            rendered_n = F.normalize(rendered_feature, dim=0)
            cos = (rendered_n * gt_feature).sum(dim=0)  # [H', W'] per-cell cos
            loss = (1.0 - cos).mean()
        else:
            loss = l1_loss(rendered_feature, gt_feature)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            tb_writer.add_scalar(f'train_loss/{distill_loss}_loss', loss.item(), iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
    # Save model
    os.makedirs(args.model_path + "/chkpnt", exist_ok=True)
    torch.save(gaussians.capture(is_language_model=True), args.model_path + "/chkpnt" + "/language.pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)

    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="SAME recon.ply the feature/artistic stages used (per-Gaussian order must match)")
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--low_dim", type=int, default=32,
                        help="per-Gaussian CLIP language field dim D' (default 32, like the VGG feature field)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16",
                        help="open_clip architecture (ViT-B-16 -> CLIP dim 512)")
    parser.add_argument("--clip_pretrained", type=str, default="openai",
                        help="open_clip pretrained tag ('openai'); pass 'none' for random weights (offline smoke)")
    parser.add_argument("--clip_grid", type=int, default=8,
                        help="CLIP dense-grid GT resolution (rows==cols); raise (e.g. 16-32) for sharper masks")
    parser.add_argument("--distill_loss", type=str, default="l1", choices=["l1", "cosine"],
                        help="distillation loss vs the CLIP GT grid: 'l1' (default, back-compat) "
                             "or 'cosine' (1 - per-cell cosine, preserves CLIP semantic direction)")
    parser.add_argument("--clip_input_res", type=int, default=224,
                        help="CLIP image-tower input resolution; drives the MaskCLIP patch grid "
                             "(ViT-B/16: res/16 per side, so 224->14x14, 448->28x28 = denser GT)")
    parser.add_argument("--gt_mode", type=str, default="maskclip",
                        choices=["maskclip", "sam_pooled", "sam_perpixel", "langsplat_ae", "siglip_dense"],
                        help="CLIP GT producer: 'maskclip' (per-patch dense tokens, default), "
                             "'sam_pooled' (SAM region-pooled CLIP into coarse grid), "
                             "'sam_perpixel' (roadmap A: SAM masked-crop CLIP painted per-pixel "
                             "onto a high-res H/stride grid, sharp object-shaped GT) or "
                             "'langsplat_ae' (Option B: MaskCLIP dense grid denoised through a "
                             "per-scene autoencoder; still CLIP-512 so eval is unchanged — "
                             "requires --ae_checkpoint from build_langsplat_ae.py)")
    parser.add_argument("--clip_dense_mode", type=str, default="maskclip",
                        choices=["maskclip", "multiscale"],
                        help="MaskCLIP dense path (gt_mode=maskclip only): 'maskclip' "
                             "(single full-image value-projection grid, default) or 'multiscale' "
                             "(fuse value-projection across crop scales to cut global-attention "
                             "contamination — B2 stronger-GT bet)")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="SAM checkpoint path (required for --gt_mode sam_pooled)")
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM architecture matching --sam_checkpoint")
    parser.add_argument("--sam_grid", type=int, default=14,
                        help="SAM-pooled GT grid (rows==cols); 14 matches the MaskCLIP field")
    parser.add_argument("--sam_points_per_side", type=int, default=16,
                        help="SAM automatic-mask sampling density (fewer = larger, cleaner regions)")
    parser.add_argument("--sam_cache_dir", type=str, default=None,
                        help="dir to cache SAM-pooled [D,gh,gw] GT grids (skips SAM+CLIP on re-run)")
    parser.add_argument("--sam_out_stride", type=int, default=4,
                        help="gt_mode=sam_perpixel: high-res output grid downsample factor "
                             "(H/stride x W/stride; 4 -> 136x244 for 546x979)")
    parser.add_argument("--ae_checkpoint", type=str, default=None,
                        help="per-scene CLIP autoencoder checkpoint (required for "
                             "--gt_mode langsplat_ae; produced by experiments/build_langsplat_ae.py)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]

    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "language", args.exp_name)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 'none' -> None so the offline smoke test uses random CLIP weights (no network).
    clip_pretrained = None if str(args.clip_pretrained).lower() == "none" else args.clip_pretrained

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.ply_path, args.debug_from,
             low_dim=args.low_dim, clip_model=args.clip_model,
             clip_pretrained=clip_pretrained, clip_grid=args.clip_grid,
             distill_loss=args.distill_loss, clip_input_res=args.clip_input_res,
             gt_mode=args.gt_mode, sam_checkpoint=args.sam_checkpoint,
             sam_model_type=args.sam_model_type, sam_grid=args.sam_grid,
             sam_points_per_side=args.sam_points_per_side, sam_cache_dir=args.sam_cache_dir,
             sam_out_stride=args.sam_out_stride, clip_dense_mode=args.clip_dense_mode,
             ae_checkpoint=args.ae_checkpoint)

    # All done
    print("\nLanguage training complete.")
