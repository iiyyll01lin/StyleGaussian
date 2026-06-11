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
             clip_model="ViT-B-16", clip_pretrained="openai", clip_grid=8):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # Frontend CLIP (Worker B's API). fp32 on gfx1151 (NEVER bf16). The Scene
    # precomputes cam.clip_features [D, H', W'] over the TRAIN cameras, mirroring
    # the VGG precompute loop.
    clip_encoder = CLIPEncoder(
        model_name=clip_model, pretrained=clip_pretrained,
        device="cuda", grid=(clip_grid, clip_grid), dtype=torch.float32,
    )

    # Load the (shared, same-as-feature/artistic) reconstruction ply and bake the
    # CLIP grid GT into every train camera.
    scene = Scene(dataset, gaussians, load_path=ply_path, clip_encoder=clip_encoder)
    gaussians.training_setup_language(opt, low_dim=low_dim, clip_dim=clip_encoder.embed_dim)

    # The feature rasterizer renders at cam.feature_height/width; point it at the
    # CLIP grid dims so the rendered map lines up with the CLIP GT for the L1 loss.
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

        # Loss: L1 against the CLIP grid GT (L2-normalized per cell).
        gt_feature = viewpoint_cam.clip_features  # [D, H', W']
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

            tb_writer.add_scalar('train_loss/l1_loss', loss.item(), iteration)

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
             clip_pretrained=clip_pretrained, clip_grid=args.clip_grid)

    # All done
    print("\nLanguage training complete.")
