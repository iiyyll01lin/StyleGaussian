"""LangSplat-style autoencoder CLIP encoder for the language field (Option B).

Motivation
----------
Triangulation (docs/06, the language-localization report) found the per-Gaussian
CLIP field's bottleneck is **per-Gaussian / per-token CLIP semantic noise**
(``truck ≈ road``), not the read-out dimension, iters, resolution or multi-scale
crop. LangSplat (Qin et al., CVPR 2024) trains a small **scene autoencoder** that
compresses the dense CLIP-512 grid to a low-dim latent and decodes back; the
autoencoder learns the *scene-specific* CLIP manifold, so encoding→decoding a
noisy per-cell CLIP vector projects it back onto the clean manifold (a denoiser).

This module is the **Option B** wiring of that idea:

* :class:`CLIPAutoencoder` — a tiny per-scene MLP autoencoder over CLIP-512 cells
  (fit offline by ``experiments/build_langsplat_ae.py``).
* :class:`LangSplatAEEncoder` — wraps the frozen :class:`CLIPEncoder` (MaskCLIP
  dense path) and pipes each grid cell through ``AE.decode(AE.encode(·))``, then
  re-L2-normalizes per cell. **The output contract is unchanged**: still
  ``[D=512, gh, gw]`` per-cell L2-normalized, so
  ``cameras.extract_clip_features`` / ``train_language.py`` / the per-Gaussian
  field + ``clip_linear`` decoder / relevancy / the eval harness all stay
  byte-for-byte the same downstream. The field is still distilled in CLIP-512
  space; only the GT is denoised.

Option A (training the field directly on the AE *latent*, needs the eval harness
to learn the decoder) is intentionally NOT done here — it is only worth the eval
changes if Option B shows real lift.

ROCm note: fp32 on the gfx1151 APU; bf16 is explicitly rejected (HANDOFF §11).

Import safety
-------------
``open_clip`` is imported lazily by :class:`CLIPEncoder`, so importing this
module never requires it; the AE itself is pure ``torch.nn``.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene.clip_encoder import CLIPEncoder


class CLIPAutoencoder(nn.Module):
    """Tiny per-scene MLP autoencoder over CLIP-512 grid cells.

    Encodes an L2-normalized CLIP embedding ``[..., input_dim]`` to a low-dim
    bottleneck and decodes back to ``input_dim``. The decoded vector is **not**
    re-normalized inside :meth:`forward`/:meth:`decode` (callers normalize), so
    the raw reconstruction can be supervised with an L2 / cosine loss offline.

    Parameters
    ----------
    input_dim : int
        CLIP joint embedding dim (512 for ViT-B/16).
    hidden_dims : sequence[int]
        Encoder hidden widths (decoder mirrors them in reverse).
    bottleneck_dim : int
        Latent width (LangSplat uses ~3; we keep it larger, e.g. 64, as a
        denoiser rather than an extreme compressor).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128),
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.bottleneck_dim = int(bottleneck_dim)

        enc_layers: List[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        enc_layers += [nn.Linear(prev, self.bottleneck_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = self.bottleneck_dim
        for h in reversed(self.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        dec_layers += [nn.Linear(prev, self.input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x`` through the bottleneck (NOT re-normalized)."""
        return self.decode(self.encode(x))

    @property
    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "bottleneck_dim": self.bottleneck_dim,
        }

    @classmethod
    def from_checkpoint(cls, path: str, map_location: Union[str, torch.device] = "cuda") -> "CLIPAutoencoder":
        """Build + load an AE from a checkpoint saved by build_langsplat_ae.py.

        The checkpoint is ``{"config": {...}, "state_dict": {...}}``.
        """
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
        cfg = blob["config"]
        ae = cls(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=tuple(int(h) for h in cfg["hidden_dims"]),
            bottleneck_dim=int(cfg["bottleneck_dim"]),
        )
        ae.load_state_dict(blob["state_dict"])
        return ae


class LangSplatAEEncoder(nn.Module):
    """Frozen MaskCLIP grid encoder + per-scene AE denoiser (Option B).

    Produces the SAME ``[embed_dim, gh, gw]`` per-cell-L2-normalized contract as
    :class:`CLIPEncoder`, but each grid cell is passed through the fitted scene
    autoencoder (``AE.decode(AE.encode(·))``) and re-normalized, projecting the
    noisy MaskCLIP token back onto the learned scene CLIP manifold.

    Parameters
    ----------
    ae_checkpoint : str
        Path to the per-scene AE checkpoint (built offline). Its ``input_dim``
        must equal the CLIP embed dim.
    clip_model, clip_pretrained, device, input_resolution, dtype, clip_grid,
    dense_mode :
        Forwarded to the inner :class:`CLIPEncoder` (the MaskCLIP GT producer).
    """

    embed_dim: int

    def __init__(
        self,
        ae_checkpoint: str,
        clip_model: str = "ViT-B-16",
        clip_pretrained: Union[str, None] = "openai",
        device: Union[str, torch.device] = "cuda",
        input_resolution: int = 224,
        dtype: torch.dtype = torch.float32,
        clip_grid: Union[int, Tuple[int, int]] = (14, 14),
        dense_mode: str = "maskclip",
    ) -> None:
        super().__init__()
        if dtype == torch.bfloat16:
            raise ValueError("gfx1151 已知 bf16 bug：請改用 fp32 或 fp16 (HANDOFF §11)。")

        self.clip = CLIPEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
            grid=clip_grid,
            dtype=dtype,
            input_resolution=input_resolution,
            dense_mode=dense_mode,
        )
        self.embed_dim = self.clip.embed_dim

        ae = CLIPAutoencoder.from_checkpoint(ae_checkpoint, map_location=device)
        if ae.input_dim != self.embed_dim:
            raise ValueError(
                f"AE input_dim={ae.input_dim} != CLIP embed_dim={self.embed_dim}；"
                f"請用同一 clip_model 重新 fit AE。"
            )
        ae = ae.to(device=device, dtype=torch.float32)
        ae.eval()
        ae.requires_grad_(False)
        self.ae = ae
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @property
    def device(self) -> torch.device:
        return self.clip.device

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Encode one image into a denoised MaskCLIP grid ``[embed_dim, gh, gw]``.

        Runs the frozen MaskCLIP dense path, pipes every cell through the scene
        AE (decode∘encode), and re-L2-normalizes per cell. Same contract as
        :meth:`CLIPEncoder.forward`.
        """
        grid = self.clip(image)  # [D, gh, gw], per-cell L2-normalized
        D, gh, gw = grid.shape
        cells = grid.reshape(D, gh * gw).t().contiguous()  # [gh*gw, D]
        recon = self.ae(cells.to(torch.float32))  # [gh*gw, D]
        recon = F.normalize(recon, dim=-1)  # per-cell L2
        return recon.t().reshape(D, gh, gw).contiguous()
