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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _anti_floater_reg(gaussians, kind):
    """Anti-floater regularizer on the (sigmoid-activated) opacity.

    The native truck reconstruction is PSNR-healthy but ~53% of its gaussians
    are near-transparent floaters (opacity<0.005).  Aggressive densify-control
    guards remove them but collapse held-out PSNR around the opacity-reset.
    This is the gentler lever: keep the *default* (PSNR-stable) densify
    dynamics untouched and add a tiny pressure that makes floaters droppable by
    the densify prune (which already culls opacity<min_opacity=0.005).

      * "opacity_entropy" — binary entropy -(o·ln o+(1-o)·ln(1-o)); minimizing
        it is BIMODAL: it pushes o<0.5 toward 0 (→ pruned) and o>0.5 toward 1,
        so it does NOT penalize the opaque gaussians that carry the image
        (PSNR-preserving), only resolves the ambiguous/floater population.
      * "opacity_l1"      — mean(o); a plain sparsity pressure on total opacity
        mass (cheaper, but pushes all opacities down, so use a small weight).

    Returns a scalar tensor (0 when kind=="none").
    """
    o = gaussians.get_opacity
    if kind == "opacity_l1":
        return o.mean()
    if kind == "opacity_entropy":
        o = o.clamp(1e-6, 1.0 - 1e-6)
        return (-(o * torch.log(o) + (1.0 - o) * torch.log(1.0 - o))).mean()
    return o.sum() * 0.0  # "none": exactly zero, keeps the graph well-formed

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             max_point_num=400_000, decouple_prune=True,
             postreset_prune_cooldown=0, postreset_freeze_cooldown=0,
             overcap_prune_size_only=False, overcap_prune_every=1,
             anti_floater="none", anti_floater_weight=0.0, anti_floater_from_iter=0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup_reconstruction(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Reconstruction training")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # Anti-floater regularizer (default OFF -> byte-identical to upstream).
        if anti_floater != "none" and anti_floater_weight > 0.0 and iteration >= anti_floater_from_iter:
            loss = loss + anti_floater_weight * _anti_floater_reg(gaussians, anti_floater)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Track-C decouple-prune safety guards: pause densify activity
                    # for a cooldown window after each opacity_reset so freshly
                    # dimmed gaussians can recover before pruning/cloning resumes.
                    since_reset = iteration % opt.opacity_reset_interval
                    past_first_reset = iteration >= opt.opacity_reset_interval
                    in_freeze = (postreset_freeze_cooldown > 0 and past_first_reset
                                 and 0 < since_reset <= postreset_freeze_cooldown)
                    skip_prune = (postreset_prune_cooldown > 0 and past_first_reset
                                  and 0 < since_reset <= postreset_prune_cooldown)
                    if not in_freeze:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                    max_point_num=max_point_num, decouple_prune=decouple_prune,
                                                    skip_prune=skip_prune, overcap_prune_size_only=overcap_prune_size_only,
                                                    overcap_prune_every=overcap_prune_every)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default='default')
    # Growth cap for densification.  Default 4e5 leaves headroom above truck's
    # natural ~337k so the cap rarely binds; when it does, the prune stays live
    # (see GaussianModel.densify_and_prune).  --legacy_cap_freeze restores the
    # old behaviour where exceeding the cap froze clone/split AND the prune.
    parser.add_argument("--max_point_num", type=int, default=400_000)
    parser.add_argument("--legacy_cap_freeze", action="store_true", default=False)
    # Track-C decouple-prune safety guards (default off = plain decouple).
    parser.add_argument("--postreset_prune_cooldown", type=int, default=0,
                        help="skip the prune for N iters after each opacity_reset")
    parser.add_argument("--postreset_freeze_cooldown", type=int, default=0,
                        help="skip ALL densify+prune for N iters after each opacity_reset")
    parser.add_argument("--overcap_prune_size_only", action="store_true", default=False,
                        help="over the cap, prune only oversized/large-screen splats (skip opacity prune)")
    parser.add_argument("--overcap_prune_every", type=int, default=1,
                        help="over the cap, run the prune only every Kth densify step")
    # Anti-floater regularizer (A2): gentle, PSNR-preserving lever that keeps the
    # default densify dynamics and makes near-transparent floaters droppable by
    # the existing densify prune.  Default off = upstream behaviour.
    parser.add_argument("--anti_floater", type=str, default="none",
                        choices=["none", "opacity_l1", "opacity_entropy"],
                        help="opacity regularizer to suppress floaters (default none)")
    parser.add_argument("--anti_floater_weight", type=float, default=0.0,
                        help="loss weight for the anti-floater regularizer")
    parser.add_argument("--anti_floater_from_iter", type=int, default=0,
                        help="start applying the anti-floater reg at this iter")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]

    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "reconstruction", args.exp_name)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             max_point_num=args.max_point_num, decouple_prune=not args.legacy_cap_freeze,
             postreset_prune_cooldown=args.postreset_prune_cooldown, postreset_freeze_cooldown=args.postreset_freeze_cooldown,
             overcap_prune_size_only=args.overcap_prune_size_only, overcap_prune_every=args.overcap_prune_every,
             anti_floater=args.anti_floater, anti_floater_weight=args.anti_floater_weight,
             anti_floater_from_iter=args.anti_floater_from_iter)

    # All done
    print("\nReconstruction complete.")
