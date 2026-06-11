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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.linear_layer import LinearLayer
from scene.gaussian_conv import GaussianConv
from scene.style_transfer import MulLayer

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, is_feature_model=False, is_style_model=False, is_language_model=False):
        
        if is_feature_model:
            data = (
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self._vgg_features,
                self.feature_linear.state_dict(),
            )
            # View-dependent (Eval 1, Variant A): append the frozen reconstruction
            # SH residual so it survives feature -> artistic. Only when opted-in;
            # otherwise the tuple stays length-6 and old loaders are unaffected.
            if getattr(self, "_carry_sh", False):
                data = data + (self._features_dc, self._features_rest)
            return data

        if is_language_model:
            # LangSplat-Lite CLIP language field (Eval 2). Same 6-entry shape as
            # the feature model (xyz/scaling/rotation/opacity + the learned field
            # + its linear decoder), so restore() can recover low_dim/CLIP dim from
            # the saved weight. The optional carried SH follows Worker A's additive,
            # length-tolerant pattern (default OFF -> a clean length-6 tuple).
            data = (
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self._clip_features,
                self.clip_linear.state_dict(),
            )
            if getattr(self, "_carry_sh", False):
                data = data + (self._features_dc, self._features_rest)
            return data

        if is_style_model:
            data = (
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self.final_vgg_features,
                self.decoder.state_dict(),
            )
            if getattr(self, "_carry_sh", False):
                data = data + (self._features_dc, self._features_rest)
            return data

        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None, from_feature_model=False, from_style_model=False, from_language_model=False):

        if from_language_model:
            (self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._clip_features,
            self.clip_linear_state_dict) = model_args[:6]
            # Recover low_dim (inChanel) and CLIP dim (out_dim) straight off the
            # decoder weight [out_dim, inChanel], so any low_dim / CLIP model loads
            # (mirrors the feature model's self-describing restore).
            self.low_dim_clip = self.clip_linear_state_dict['layer.weight'].shape[1]
            self.clip_dim = self.clip_linear_state_dict['layer.weight'].shape[0]
            self.clip_linear = LinearLayer(inChanel=self.low_dim_clip, out_dim=self.clip_dim).cuda()
            self.clip_linear.load_state_dict(self.clip_linear_state_dict)
            # Length-tolerant: reattach the optional carried SH (len>6) exactly like
            # the feature/style restore; a clean length-6 language ckpt is flat.
            self._restore_carried_sh(model_args)
            return

        if from_feature_model:
            (self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._vgg_features,
            self.feature_linear_state_dict) = model_args[:6]
            # 從 state_dict 推回 D'(low-dim)：LinearLayer.layer.weight 形狀
            # [out_dim=256, inChanel]（feape=0），inChanel 即訓練時用的 low_dim。
            # 這樣不論 checkpoint 用 16/32/... 都能正確還原（向後相容舊的 32）。
            self.low_dim = self.feature_linear_state_dict['layer.weight'].shape[1]
            self.feature_linear = LinearLayer(inChanel=self.low_dim, out_dim=256).cuda()
            self.feature_linear.load_state_dict(self.feature_linear_state_dict)
            # View-dependent (Variant A): a checkpoint trained with --view_dependent
            # carries the frozen reconstruction SH after the 6 base entries. Old
            # length-6 checkpoints take the else-branch and behave exactly as before.
            self._restore_carried_sh(model_args)
            return
        
        if from_style_model:
            (self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.final_vgg_features,
            self.decoder_state_dict) = model_args[:6]
            # 從 decoder state_dict 推回 K 與每層寬度：kernels.0 形狀 [256, K*256]
            # → K = shape[1] // 256；每層 out width = kernels.i 的 shape[0]。直接從
            # 權重推回 layers_channel 同時支援舊的 RGB head (out=3) 與 Variant B 的
            # SH head (out=3*(deg+1)^2)，且舊 checkpoint 還原成 [256,128,64,32,3]。
            input_channel = 256
            sd = self.decoder_state_dict
            kernel_keys = sorted([k for k in sd if k.startswith("kernels.")],
                                 key=lambda x: int(x.split(".")[1]))
            layers_channel = [sd[k].shape[0] for k in kernel_keys]
            K = sd['kernels.0'].shape[1] // input_channel
            self.decoder = GaussianConv(
                self.get_xyz.detach(), input_channel=input_channel,
                layers_channel=layers_channel, K=K
            ).cuda()
            self.decoder.load_state_dict(sd)
            self.style_transfer = MulLayer().cuda()
            self._restore_carried_sh(model_args)
            # Variant B (decoder_sh): a non-RGB decoder head (out != 3) means the
            # decoder predicts per-Gaussian SH coeffs. Detect it from the head width
            # so the checkpoint is self-describing for render() — cfg_args only
            # persists ModelParams, so render cannot read the optimisation flags.
            out_width = layers_channel[-1]
            if out_width != 3 and not self._carry_sh:
                self.view_dependent = True
                self.view_dependent_mode = "decoder_sh"
                self.sh_degree_style = int(round((out_width / 3.0) ** 0.5)) - 1
            return

        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup_reconstruction(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def _restore_carried_sh(self, model_args):
        """Restore the optional view-dependent SH residual appended to a feature /
        style checkpoint (Eval 1, Variant A). Length-6 (legacy) checkpoints carry no
        SH → behaviour is unchanged. When present, the frozen reconstruction SH is
        reattached and the view-dependent flags are set so render() can synthesize
        per-view color."""
        if len(model_args) > 6:
            self._features_dc = model_args[6]
            self._features_rest = model_args[7]
            self._carry_sh = True
            self.view_dependent = True
            self.view_dependent_mode = "residual"
            # The carried SH is full-degree (from a finished reconstruction), so make
            # the active degree match for any code that reads active_sh_degree.
            self.active_sh_degree = self.max_sh_degree
        else:
            # Flat baseline; the from_style_model caller may still detect a Variant B
            # decoder_sh head from the decoder geometry and flip this back on.
            self._carry_sh = False
            self.view_dependent = False
            self.view_dependent_mode = "residual"

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def final_clip_features(self):
        # Per-Gaussian decoded CLIP features [N, clip_dim] for language relevancy
        # (Eval 2). Decodes the learned low-dim field through clip_linear, the same
        # forward_directly_on_point used to bake final_vgg_features. Only valid
        # after training_setup_language / restore(from_language_model=True).
        return self.clip_linear.forward_directly_on_point(self._clip_features)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup_reconstruction(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def training_setup_feature(self, training_args, low_dim=32):
        # View-dependent (Eval 1, Variant A): keep the reconstruction's SH residual
        # alive so it can survive into the style checkpoint. It is frozen (detached,
        # not an nn.Parameter), never added to the optimizer, and the feature render
        # path splats `_vgg_features` (never `get_features`) — so feature training
        # stays bit-identical; only the saved feature.pth carries the extra SH.
        # `training_args` is the optimization param group (train_feature.py forwards
        # it here), which carries the --view_dependent flags. Default OFF -> the SH
        # is deleted exactly as upstream.
        self.view_dependent = bool(getattr(training_args, "view_dependent", False))
        self.view_dependent_mode = getattr(training_args, "view_dependent_mode", "residual")
        self._carry_sh = self.view_dependent and self.view_dependent_mode == "residual"
        if self._carry_sh:
            self._features_dc = self._features_dc.detach()
            self._features_rest = self._features_rest.detach()
        else:
            # delete spherical harmonics because we don't need them for feature reconstruction
            del self._features_rest
            del self._features_dc

        # low_dim = paper 的 D'（每個 Gaussian 學的低維特徵維度，預設 32）。可配置。
        self.low_dim = low_dim
        _vgg_features = torch.randn((self.get_xyz.shape[0], low_dim), device="cuda").requires_grad_(True)
        self._vgg_features = nn.Parameter(_vgg_features)
        self.feature_linear = LinearLayer(inChanel=low_dim, out_dim=256).cuda()

        l = [
            {'params': [self._vgg_features], 'lr': 0.01, "name": "vgg_features"},
            {'params': self.feature_linear.parameters(), 'lr': 1e-3, "name": "feature_linear"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def training_setup_language(self, training_args, low_dim=32, clip_dim=512):
        # LangSplat-Lite CLIP language field (Eval 2), mirroring
        # training_setup_feature but with a SEPARATE per-Gaussian field
        # `_clip_features [N, low_dim]` and a `clip_linear` LinearLayer decoding it
        # back to CLIP space (out_dim == clip_encoder.embed_dim, 512 for ViT-B/16).
        # The render feature path splats `_clip_features` via render(..., 
        # feature_override=...) and the L1 loss matches the splatted+decoded map
        # against Camera.clip_features. This is a brand-new, additive setup entry:
        # the existing feature/artistic/recon setups are untouched.
        #
        # device-following (self.get_xyz.device) so the shape/decoder logic is
        # unit-testable on CPU; on the GPU box xyz is cuda so behaviour is
        # identical to the feature stage's hard-coded .cuda().
        device = self.get_xyz.device

        # The localized blend reads the photoreal original color off the ARTISTIC
        # model, so the language field itself needs no SH. Still, follow Worker A's
        # default-OFF `_carry_sh` pattern for a self-describing checkpoint; with it
        # OFF (the LITE default) the SH is dropped exactly like the feature stage.
        self.view_dependent = bool(getattr(training_args, "view_dependent", False))
        self.view_dependent_mode = getattr(training_args, "view_dependent_mode", "residual")
        self._carry_sh = self.view_dependent and self.view_dependent_mode == "residual"
        if self._carry_sh:
            self._features_dc = self._features_dc.detach()
            self._features_rest = self._features_rest.detach()
        else:
            del self._features_rest
            del self._features_dc

        self.low_dim_clip = low_dim
        self.clip_dim = clip_dim
        _clip_features = torch.randn((self.get_xyz.shape[0], low_dim), device=device).requires_grad_(True)
        self._clip_features = nn.Parameter(_clip_features)
        self.clip_linear = LinearLayer(inChanel=low_dim, out_dim=clip_dim).to(device)

        l = [
            {'params': [self._clip_features], 'lr': 0.01, "name": "clip_features"},
            {'params': self.clip_linear.parameters(), 'lr': 1e-3, "name": "clip_linear"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def training_setup_decoder(self, training_args):
        # compute the final vgg features for each point
        self.final_vgg_features = self.feature_linear.forward_directly_on_point(self._vgg_features)

        # delete vgg features and linear layer because we have the perpoint features now
        del self._vgg_features
        del self.feature_linear

        # init gaussian conv
        self.decoder = GaussianConv(self.get_xyz.detach()).cuda()
       
        l = [
            {'params': self.decoder.parameters(), 'lr': 2e-3, "name": "decoder"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def training_setup_style(self, training_args, decoder_path, photorealistic=False, K=8):
        # View-dependent flags (Eval 1). `training_args` is the optimization param
        # group forwarded by train_artistic.py. All default OFF -> upstream behaviour.
        self.view_dependent = bool(getattr(training_args, "view_dependent", False))
        self.view_dependent_mode = getattr(training_args, "view_dependent_mode", "residual")
        sh_degree_style = int(getattr(training_args, "sh_degree_style", 2))

        # compute the final vgg features for each point
        self.final_vgg_features = self.feature_linear.forward_directly_on_point(self._vgg_features)
        self.final_vgg_features += torch.randn_like(self.final_vgg_features) # Hack: randomness improves stylization quality

        # delete vgg features and linear layer because we have the perpoint features now
        del self._vgg_features
        del self.feature_linear

        # Variant A: keep the (already-restored, frozen) reconstruction SH residual so
        # it survives into the style checkpoint. Do NOT delete it here. If residual
        # mode was requested but the feature ckpt carried no SH, fall back to flat.
        self._carry_sh = (self.view_dependent and self.view_dependent_mode == "residual"
                          and hasattr(self, "_features_dc"))
        if self.view_dependent and self.view_dependent_mode == "residual" and not hasattr(self, "_features_dc"):
            print("[view_dependent] residual mode requested but the feature checkpoint carried no "
                  "SH residual — re-run feature training with --view_dependent. Falling back to flat.")
            self.view_dependent = False

        # Variant B: decoder head emits per-Gaussian SH coeffs (3*(deg+1)^2) not RGB.
        out_channel = None
        if self.view_dependent and self.view_dependent_mode == "decoder_sh":
            out_channel = 3 * (sh_degree_style + 1) ** 2

        # init gaussian conv（K = paper 的 KNN 鄰居數，可配置；photorealistic 仍強制 K=1）
        self.decoder = GaussianConv(self.get_xyz.detach(), K=(1 if photorealistic else K),
                                    out_channel=out_channel).cuda()
        if decoder_path:
            print('Init decoder from {}'.format(decoder_path))
            (_xyz,
            _scaling,
            _rotation,
            _opacity,
            final_vgg_features,
            decoder_state_dict) = torch.load(decoder_path)[:6]
            self.decoder.load_state_dict(decoder_state_dict)

        # init style transfer module
        self.style_transfer = MulLayer().cuda()

        l = [
            {'params': self.decoder.parameters(), 'lr': 1e-3, "name": "decoder"},
            {'params': self.style_transfer.parameters(), 'lr': 1e-3, "name": "style_transfer"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_point_num=3e5, decouple_prune=True,
                          skip_prune=False, overcap_prune_size_only=False, overcap_prune_every=1):
        # The growth cap (max_point_num) must gate ONLY clone/split, never the
        # opacity/size prune.  The legacy code early-returned the whole method
        # once the count exceeded the cap, which also froze the
        # opacity(<min_opacity)/size prune for the rest of densification, so
        # faded + oversized gaussians (floaters) accumulated and were never
        # culled.  Decoupling keeps the prune running at the cap: the count
        # oscillates just below it while floaters are continuously removed and
        # clone/split refills the freed budget with well-placed gaussians.
        # decouple_prune=False restores the legacy freeze-everything behaviour.
        #
        # Plain decoupling, however, collapses the reconstruction on gfx1151:
        # pruning/cloning keeps churning right after each opacity_reset and the
        # freshly-dimmed gaussians never recover (held-out PSNR 22->9->6).  These
        # opt-in guards make the decoupled prune safe (all default to the plain
        # decouple behaviour so the legacy/baseline paths are untouched):
        #   skip_prune               - caller signals a post-reset cooldown; skip
        #                              the prune this call so opacities recover.
        #   overcap_prune_size_only  - over the cap, drop only oversized / large-
        #                              screen splats, never the opacity(<min) set.
        #   overcap_prune_every      - over the cap, run the prune only every Kth
        #                              densify call (gentler cadence).
        over_cap = self.get_xyz.shape[0] > max_point_num
        if over_cap and not decouple_prune:
            return

        if not over_cap:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        run_prune = not skip_prune
        prune_opacity = True
        if over_cap:
            self._overcap_densify_calls = getattr(self, "_overcap_densify_calls", 0) + 1
            if overcap_prune_every > 1 and (self._overcap_densify_calls % overcap_prune_every) != 0:
                run_prune = False
            if overcap_prune_size_only:
                prune_opacity = False

        if run_prune:
            if prune_opacity:
                prune_mask = (self.get_opacity < min_opacity).squeeze()
            else:
                prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # Read the screen-space gradient that drives clone/split.  Prefer
        # gsplat's absolute pixel-space gradient (info["means2d"].absgrad, the
        # AbsGS criterion, set during gsplat's rasterize backward); fall back
        # to .grad (the INRIA `original` backend writes the NDC screen-space
        # gradient there directly).  See gsplat_backend.py section 3.
        grad = getattr(viewspace_point_tensor, "absgrad", None)
        if grad is None:
            grad = viewspace_point_tensor.grad
        if grad is None:
            return
        if grad.dim() == 3:
            # gsplat means2d is [C, N, 2]; StyleGaussian renders one camera at
            # a time, so collapse the (size-1) camera dim to [N, 2].
            grad = grad[0]
        ndc_scale = getattr(viewspace_point_tensor, "sg_ndc_scale", None)
        if ndc_scale is not None:
            # Convert gsplat pixel-space grads to INRIA's NDC convention
            # (x*=0.5*W, y*=0.5*H) so densify_grad_threshold=0.0002 still applies.
            grad = grad * ndc_scale
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1