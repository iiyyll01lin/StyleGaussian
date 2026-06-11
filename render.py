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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import (
    render, stylized_view_dependent_color, view_directions,
    photoreal_view_color, localized_blend,
)
import torchvision
import re
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from scene.VGG import VGGEncoder, normalize_vgg

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, style=None,
               view_dependent=False, view_dependent_mode="residual", residual_scale=1.0, sh_degree_style=2,
               localized=None):

    if style:
        (style_img, style_name) = style
        render_path = os.path.join(model_path, name, style_name, "renders")
        gts_path = os.path.join(model_path, name, style_name, "gt")
        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # Language-localized render (Eval 2) writes to its own subdir so it never
    # clobbers the normal stylized render.
    if style and localized is not None:
        tag = f"{style_name}_localized_{localized['slug']}"
        render_path = os.path.join(model_path, name, tag, "renders")
        gts_path = os.path.join(model_path, name, tag, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # View-dependent (Eval 1) only applies to the stylized path. For residual mode
    # we need the carried reconstruction SH; if it is missing (e.g. a flat ckpt),
    # fall back to flat rendering so old checkpoints keep working unchanged.
    if view_dependent and style and view_dependent_mode == "residual" and not getattr(gaussians, "_carry_sh", False):
        print("[view_dependent] this style checkpoint carries no SH residual; rendering flat. "
              "Re-train feature + artistic with --view_dependent to enable it.")
        view_dependent = False

    override_color = None
    stylized_out = None
    if style:
        tranfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), # point cloud features [N, C]
            style_img_features.relu3_1,
        )

        # Decoder output is view-independent and expensive, so compute it ONCE for all
        # cameras. Flat: this is the final [N,3] color. View-dependent: it is the base
        # ([N,3] for residual, or the raw [N,3*(deg+1)^2] SH head for decoder_sh) that
        # gets turned into a per-camera color inside the loop below.
        stylized_out = gaussians.decoder(tranfered_features)
        if not view_dependent:
            override_color = stylized_out # [N, 3] — unchanged flat path

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if view_dependent and style:
            per_view_color = stylized_view_dependent_color(
                gaussians, view, stylized_out, mode=view_dependent_mode,
                residual_scale=residual_scale, sh_degree_style=sh_degree_style)
        else:
            per_view_color = override_color

        # Language-localized blend (Eval 2): keep the photoreal ORIGINAL color
        # outside the masked region, the (possibly view-dependent) stylized color
        # inside it. The mask M is a 3D per-Gaussian field (computed once), so this
        # composes with Worker A's per-camera stylized color and stays multi-view
        # consistent. The original color is recomputed per view from recon SH.
        if localized is not None and style:
            if localized["orig_shs"] is not None:
                dirs = view_directions(gaussians, view)
                original_rgb = photoreal_view_color(localized["orig_shs"], dirs, localized["orig_deg"])
            else:
                original_rgb = per_view_color  # no photoreal source -> blend is a no-op (warned in _build_localized)
            per_view_color = localized_blend(localized["mask"], per_view_color, original_rgb)

        rendering = render(view, gaussians, pipeline, background, override_color=per_view_color)["render"]
        rendering = rendering.clamp(0, 1)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def _slugify(text):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "query"


def _build_localized(gaussians, dataset, language_model, text_query, mask_threshold, mask_temperature,
                     recon_ply=None, clip_model="ViT-B-16", clip_pretrained="openai"):
    """Build the language-localized blend payload for ``text_query`` (Eval 2).

    Loads the language checkpoint into a SEPARATE model, asserts it shares the
    artistic model's recon geometry, computes the per-Gaussian soft mask M, and
    picks the photoreal ORIGINAL color source (recon SH). Returns a dict consumed
    by ``render_set`` (mask / orig_shs / orig_deg / slug)."""
    # open_clip is imported lazily inside CLIPEncoder, so these imports stay
    # cheap and only happen on the localized path.
    from scene.clip_encoder import CLIPEncoder
    from scene.language import compute_relevancy, relevancy_to_mask, encode_text, encode_negatives

    # --- load the language field into a SEPARATE GaussianModel -----------------
    lang = GaussianModel(dataset.sh_degree)
    lang.restore(torch.load(language_model), from_language_model=True)

    # --- ALIGNMENT assert: language vs artistic share the recon geometry -------
    assert lang.get_xyz.shape[0] == gaussians.get_xyz.shape[0], (
        f"language field has {lang.get_xyz.shape[0]} gaussians but the artistic model has "
        f"{gaussians.get_xyz.shape[0]}; both must derive from the SAME recon.ply "
        "(feature/artistic/language never densify, so the per-Gaussian order is identical).")
    assert torch.allclose(lang.get_xyz, gaussians.get_xyz, atol=1e-5), (
        "language vs artistic xyz mismatch; train the language field from the SAME recon.ply "
        "the feature/artistic stages used.")

    # --- text relevancy -> 3D per-Gaussian soft mask M (computed ONCE) ---------
    pretrained = None if str(clip_pretrained).lower() == "none" else clip_pretrained
    enc = CLIPEncoder(model_name=clip_model, pretrained=pretrained, device="cuda", dtype=torch.float32)
    qemb = encode_text(text_query, enc)              # [Dclip]
    negs = encode_negatives(enc)                     # [K, Dclip]
    phi = lang.final_clip_features                   # [N, Dclip]
    scores = compute_relevancy(phi, qemb, negs)      # [N]
    mask = relevancy_to_mask(scores, threshold=mask_threshold, temperature=mask_temperature)  # [N]

    # --- photoreal ORIGINAL color source (per-Gaussian recon SH) ---------------
    deg = gaussians.max_sh_degree
    orig_shs = None
    if getattr(gaussians, "_carry_sh", False):
        orig_shs = gaussians.get_features.transpose(1, 2).view(-1, 3, (deg + 1) ** 2)
        print("[localized] original color: artistic model's carried recon SH (Variant A).")
    elif recon_ply:
        recon = GaussianModel(dataset.sh_degree)
        recon.load_ply(recon_ply)
        assert recon.get_xyz.shape[0] == gaussians.get_xyz.shape[0], (
            f"--recon_ply has {recon.get_xyz.shape[0]} gaussians but the artistic model has "
            f"{gaussians.get_xyz.shape[0]}; pass the SAME recon.ply.")
        deg = recon.max_sh_degree
        orig_shs = recon.get_features.transpose(1, 2).view(-1, 3, (deg + 1) ** 2)
        print(f"[localized] original color: --recon_ply {recon_ply}.")
    elif getattr(lang, "_carry_sh", False):
        deg = lang.max_sh_degree
        orig_shs = lang.get_features.transpose(1, 2).view(-1, 3, (deg + 1) ** 2)
        print("[localized] original color: language model's carried recon SH.")
    else:
        print("[localized][warn] no photoreal SH source (flat artistic, no --recon_ply, language "
              "field carries no SH) -> blend falls back to the stylized color, so localization is a "
              "no-op. Pass --recon_ply <recon.ply> (same one used for feature/artistic) to fix this.")

    frac = float((mask > 0.5).float().mean().item())
    print(f"[localized] query={text_query!r}  mask>0.5 fraction={frac:.3f}  "
          f"threshold={mask_threshold}  temperature={mask_temperature}")
    return {"mask": mask, "orig_shs": orig_shs, "orig_deg": deg, "slug": _slugify(text_query)}


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, style_img_path, skip_train : bool, skip_test : bool,
                residual_scale=None, force_flat=False, text_query=None, mask_threshold=0.5, mask_temperature=10.0,
                language_model=None, recon_ply=None, clip_model="ViT-B-16", clip_pretrained="openai"):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        style = None
        if style_img_path:
            ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
            scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

            # read style image
            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
            style_name = Path(style_img_path).stem
            style = (style_img, style_name)
        else:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # View-dependent (Eval 1) is read off the loaded checkpoint, which is
        # self-describing (residual mode -> carried SH; decoder_sh -> wide head).
        # --no_view_dependent forces flat; --residual_scale overrides the blend.
        view_dependent = bool(getattr(gaussians, "view_dependent", False)) and not force_flat
        view_dependent_mode = getattr(gaussians, "view_dependent_mode", "residual")
        sh_degree_style = int(getattr(gaussians, "sh_degree_style", gaussians.max_sh_degree))
        rscale = 1.0 if residual_scale is None else float(residual_scale)
        if view_dependent:
            print(f"[view_dependent] ON  mode={view_dependent_mode}  residual_scale={rscale}  sh_degree_style={sh_degree_style}")
        vd_kwargs = dict(view_dependent=view_dependent, view_dependent_mode=view_dependent_mode,
                         residual_scale=rscale, sh_degree_style=sh_degree_style)

        # Language-localized stylization (Eval 2). Only engages with --text_query;
        # absent -> localized stays None and every render path below is unchanged.
        localized = None
        if text_query:
            if not style:
                raise ValueError("--text_query needs a style image: it stylizes only the masked "
                                 "region and keeps the rest photoreal.")
            if not language_model:
                raise ValueError("--text_query needs --language_model <path to language.pth> "
                                 "(train it with train_language.py).")
            localized = _build_localized(gaussians, dataset, language_model, text_query, mask_threshold,
                                         mask_temperature, recon_ply=recon_ply, clip_model=clip_model,
                                         clip_pretrained=clip_pretrained)
        vd_kwargs["localized"] = localized

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, style, **vd_kwargs)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, style, **vd_kwargs)


def render_sets_style_interpolate(dataset : ModelParams,  pipeline : PipelineParams, style_img_paths, view_id=0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        # read 4 style images
        trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])

        style_img0 = trans(Image.open(style_img_paths[0])).cuda()[None, :3, :, :]
        style_img1 = trans(Image.open(style_img_paths[1])).cuda()[None, :3, :, :]
        style_img2 = trans(Image.open(style_img_paths[2])).cuda()[None, :3, :, :]
        style_img3 = trans(Image.open(style_img_paths[3])).cuda()[None, :3, :, :]

        style_name0 = Path(style_img_paths[0]).stem
        style_name1 = Path(style_img_paths[1]).stem
        style_name2 = Path(style_img_paths[2]).stem
        style_name3 = Path(style_img_paths[3]).stem

        all_style_name = f'{style_name0}_{style_name1}_{style_name2}_{style_name3}'

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "style_interpolation")
        makedirs(render_path, exist_ok=True)
        
        # get the style features
        vgg_encoder = VGGEncoder().cuda()
        style_img_features0 = vgg_encoder(normalize_vgg(style_img0))
        style_img_features1 = vgg_encoder(normalize_vgg(style_img1))
        style_img_features2 = vgg_encoder(normalize_vgg(style_img2))
        style_img_features3 = vgg_encoder(normalize_vgg(style_img3))

        # get the transfered features
        tranfered_features0 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features0.relu3_1,
        )
        tranfered_features1 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features1.relu3_1,
        )
        tranfered_features2 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features2.relu3_1,
        )
        tranfered_features3 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features3.relu3_1,
        )


        v = torch.linspace(0,1,steps=5)
        up_maps = []
        for i in range(5):
            up_maps.append(tranfered_features0 * v[i] + tranfered_features1 * v[4-i])
        down_maps = []
        for i in range(5):
            down_maps.append(tranfered_features2 * v[i] + tranfered_features3 * v[4-i])

        images = []
        w = torch.linspace(0,1,steps=4)
        for y in range(4):
            for x in range(5):
                tranfered_features_interpolated = up_maps[x] * w[y] + down_maps[x] * w[3-y]
                override_color = gaussians.decoder(tranfered_features_interpolated) # [N, 3]

                view = scene.getTrainCameras()[view_id]
                rendering = render(view, gaussians, pipeline, background, override_color=override_color)["render"]
                rendering = rendering.clamp(0, 1)

                rendering = torchvision.transforms.functional.resize(rendering, 300)

                images.append(rendering)

        torchvision.utils.save_image(images, fp=f'{render_path}/{all_style_name}.png', nrow=5, padding=0)


def render_sets_content_interpolate(dataset : ModelParams,  pipeline : PipelineParams, style_img_path, view_id=0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        # read style features
        trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
        style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
        style_name = Path(style_img_path).stem
        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "content_interpolation")
        makedirs(render_path, exist_ok=True)

        # get the transfered features
        tranfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features.relu3_1,
        )

        v = torch.linspace(0,1,steps=5)

        images = []
        for x in range(5):
            tranfered_features_interpolated = tranfered_features * v[x] + gaussians.final_vgg_features * v[4-x]
            override_color = gaussians.decoder(tranfered_features_interpolated)

            view = scene.getTrainCameras()[view_id]
            rendering = render(view, gaussians, pipeline, background, override_color=override_color)["render"]
            rendering = rendering.clamp(0, 1)

            rendering = torchvision.transforms.functional.resize(rendering, 300)

            images.append(rendering)

        torchvision.utils.save_image(images, fp=f'{render_path}/{style_name}.png', nrow=5, padding=0)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--style", nargs='+', default='', type=str)
    parser.add_argument("--content_interpolate", action="store_true", default=False)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # View-dependent (Eval 1) render-time controls. Whether view-dependence was
    # trained, and in which mode / degree, is read off the (self-describing) style
    # checkpoint inside render_sets — so these flags only OVERRIDE: scale the
    # Variant A residual blend (handy for the flat-vs-A ablation), or force a flat
    # render of a view-dependent checkpoint. Old checkpoints render exactly as before.
    parser.add_argument("--residual_scale", type=float, default=None,
                        help="Variant A: override the view-dependent residual blend weight (default 1.0)")
    parser.add_argument("--no_view_dependent", action="store_true", default=False,
                        help="force flat (view-independent) rendering even for a view-dependent checkpoint")
    # Language-localized stylization (Eval 2). Absent --text_query -> every
    # existing render path is byte-identical to before. With --text_query, the
    # masked region (CLIP relevancy of the query over the language field) is
    # stylized and the rest kept photoreal.
    parser.add_argument("--text_query", type=str, default=None,
                        help="Eval 2: localize the style to gaussians matching this text (e.g. 'the truck')")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="relevancy decision midpoint for the soft mask (higher -> smaller mask)")
    parser.add_argument("--mask_temperature", type=float, default=10.0,
                        help="softness of the mask boundary (higher -> sharper)")
    parser.add_argument("--language_model", type=str, default=None,
                        help="path to a language.pth from train_language.py (required with --text_query)")
    parser.add_argument("--recon_ply", type=str, default=None,
                        help="recon.ply supplying the photoreal ORIGINAL color when the artistic model is flat")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16",
                        help="open_clip architecture for the query text encoder (must match train_language.py)")
    parser.add_argument("--clip_pretrained", type=str, default="openai",
                        help="open_clip pretrained tag (must match train_language.py); 'none'=random (offline smoke)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # get_combined_args drops CLI args left at their None default, so use getattr.
    vd_kwargs = dict(residual_scale=getattr(args, "residual_scale", None),
                     force_flat=getattr(args, "no_view_dependent", False))
    loc_kwargs = dict(text_query=getattr(args, "text_query", None),
                      mask_threshold=getattr(args, "mask_threshold", 0.5),
                      mask_temperature=getattr(args, "mask_temperature", 10.0),
                      language_model=getattr(args, "language_model", None),
                      recon_ply=getattr(args, "recon_ply", None),
                      clip_model=getattr(args, "clip_model", "ViT-B-16"),
                      clip_pretrained=getattr(args, "clip_pretrained", "openai"))

    if not args.style:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), None, args.skip_train, args.skip_test, **vd_kwargs, **loc_kwargs)
    if len(args.style) == 1: 
        if args.content_interpolate:
            render_sets_content_interpolate(model.extract(args), pipeline.extract(args), args.style[0])
        else:
            render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.style[0], args.skip_train, args.skip_test, **vd_kwargs, **loc_kwargs)
    elif len(args.style) == 4:
        render_sets_style_interpolate(model.extract(args), pipeline.extract(args), args.style)
    else:
        print("Invalid style argument, should provide 1 or 4 styles. 1 for style transfer, 4 for style interpolation.")