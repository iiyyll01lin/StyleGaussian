"""CLIP encoder for the LangSplat-style language feature field (Lite, no SAM).

This mirrors the role of :mod:`scene.VGG` in the existing pipeline. Where
``VGGEncoder`` turns an image into a dense conv feature map (``relu3_1`` →
``[256, H, W]``) that supervises the per-Gaussian VGG style field, ``CLIPEncoder``
turns an image into a *grid of CLIP image embeddings* ``[D, gh, gw]`` that can
supervise a per-Gaussian CLIP **language** field in exactly the same way
(distill a low-dim field + linear decode back to CLIP dim, see
``train_feature.py`` / ``GaussianModel.training_setup_feature``).

Lite path (this file)
---------------------
Each spatial cell of the output map is the CLIP embedding of one image window
obtained by sliding-window / grid cropping (optionally overlapping). This needs
nothing beyond a frozen CLIP image tower, so it is cheap and ROCm-safe (fp32 /
fp16, **no bf16** per the gfx1151 notes). Boundaries are coarse but enough to
drive a 3D, multi-view-consistent language mask.

The full LangSplat hierarchy (SAM 3-level masks + per-region CLIP + scene
autoencoder) is intentionally **out of scope** here and documented as future
work.

Import safety
-------------
``open_clip`` is imported *lazily inside* :meth:`CLIPEncoder.__init__`, so simply
``import``-ing this module never requires ``open_clip`` to be installed and the
existing reconstruction / feature / artistic pipeline keeps working unchanged.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# OpenAI/OpenCLIP image normalization (used as a fallback if we cannot read the
# exact mean/std off the model's preprocess transform).
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPEncoder(nn.Module):
    """Frozen CLIP image/text encoder producing dense grid features (Lite).

    Parameters
    ----------
    model_name : str
        ``open_clip`` architecture name, e.g. ``"ViT-B-16"`` (CLIP dim 512).
    pretrained : str | None
        ``open_clip`` pretrained tag, e.g. ``"openai"`` or
        ``"laion2b_s34b_b88k"``. Pass ``None`` for random weights (offline /
        unit-test use — no network access).
    device : str | torch.device
        Device to place the model on (mirrors ``VGGEncoder().cuda()`` usage).
        Use ``"cuda"`` on the gfx1151 box.
    grid : int | tuple[int, int]
        Number of windows ``(rows, cols)`` for the dense map. An ``int`` ``g``
        is treated as ``(g, g)``. Output feature map is ``[D, rows, cols]``.
    overlap : float
        Fractional overlap between neighbouring windows in ``[0, 1)``. ``0.0``
        is a non-overlapping tiling; ``0.5`` gives 50%-overlapping windows
        (sliding-window flavour, smoother boundaries).
    input_resolution : int
        Square resolution each window is resized to before the CLIP image
        tower (CLIP ViT-B/16 expects 224).
    dtype : torch.dtype
        Compute dtype for the CLIP tower. Use ``torch.float32`` (default) or
        ``torch.float16``; **never** ``torch.bfloat16`` on gfx1151.
    chunk_size : int
        Max windows encoded per forward chunk (bounds peak memory).

    Attributes
    ----------
    embed_dim : int
        The CLIP joint image/text embedding dimension ``D`` (512 for ViT-B/16).
        Phase-2 builds ``LinearLayer(inChanel=low_dim, out_dim=embed_dim)`` to
        decode the per-Gaussian language field back to CLIP space.
    model : nn.Module
        The underlying frozen ``open_clip`` model (registered submodule, so
        ``.to(...)`` / ``.cuda()`` / ``.eval()`` propagate as usual).
    """

    embed_dim: int

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: Union[str, None] = "openai",
        device: Union[str, torch.device] = "cuda",
        grid: Union[int, Tuple[int, int]] = (8, 8),
        overlap: float = 0.0,
        input_resolution: int = 224,
        dtype: torch.dtype = torch.float32,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()

        # --- validate config (cheap, before any heavy import) -----------------
        if isinstance(grid, int):
            grid = (grid, grid)
        if len(grid) != 2 or grid[0] < 1 or grid[1] < 1:
            raise ValueError(f"grid 必須是 (rows, cols) 且都 >= 1，但拿到 {grid}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap 必須落在 [0, 1)，但拿到 {overlap}")
        if input_resolution < 1:
            raise ValueError(f"input_resolution 必須 >= 1，但拿到 {input_resolution}")
        if dtype == torch.bfloat16:
            raise ValueError("gfx1151 已知 bf16 bug：請改用 fp32 或 fp16 (HANDOFF §11)。")
        if chunk_size < 1:
            raise ValueError(f"chunk_size 必須 >= 1，但拿到 {chunk_size}")

        self.model_name = model_name
        self.pretrained = pretrained
        self.grid: Tuple[int, int] = (int(grid[0]), int(grid[1]))
        self.overlap = float(overlap)
        self.input_resolution = int(input_resolution)
        self.compute_dtype = dtype
        self.chunk_size = int(chunk_size)

        # --- lazy, guarded open_clip import ----------------------------------
        try:
            import open_clip  # noqa: F401  (imported lazily on purpose)
        except ImportError as exc:  # pragma: no cover - exercised only w/o dep
            raise ImportError(
                "CLIPEncoder 需要 open_clip：請先安裝 `pip install open_clip_torch` "
                "(已列入 requirements-rocm.txt)。注意：只有在實際建立 CLIPEncoder "
                "時才需要它；單純 import 本模組不需要。"
            ) from exc

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model = model.to(device=device, dtype=dtype)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # --- resolve the CLIP embedding dim robustly -------------------------
        self.embed_dim = self._resolve_embed_dim(open_clip, model_name, device)

        # --- image normalization (read exact mean/std off preprocess) --------
        # Create the buffers on `device` so they live alongside the CLIP tower
        # (the model was moved there above). Without this they default to CPU and
        # _normalize() raises a device-mismatch on a cuda image. register_buffer
        # still lets .to()/.cuda() move them later, so this is purely the correct
        # initial placement.
        mean, std = self._resolve_normalization(preprocess_val)
        self.register_buffer("pixel_mean", torch.tensor(mean, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std, device=device).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ helpers
    def _resolve_embed_dim(self, open_clip, model_name: str, device) -> int:
        cfg = open_clip.get_model_config(model_name)
        if cfg is not None and "embed_dim" in cfg:
            return int(cfg["embed_dim"])
        # Fallback: probe with a 1-token text forward (cheap, no network).
        with torch.no_grad():
            tokens = self.tokenizer(["x"]).to(device)
            return int(self.model.encode_text(tokens).shape[-1])

    @staticmethod
    def _resolve_normalization(preprocess_val) -> Tuple[Sequence[float], Sequence[float]]:
        transforms = getattr(preprocess_val, "transforms", [])
        for t in transforms:
            if type(t).__name__ == "Normalize":
                mean = tuple(float(v) for v in t.mean)
                std = tuple(float(v) for v in t.std)
                return mean, std
        return _OPENAI_CLIP_MEAN, _OPENAI_CLIP_STD

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _normalize(self, images: Tensor) -> Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _resize_square(self, image: Tensor) -> Tensor:
        # image: [3, h, w] -> [3, R, R] (CLIP uses bicubic resize).
        res = self.input_resolution
        return F.interpolate(
            image.unsqueeze(0),
            size=(res, res),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _crop_cell(self, image: Tensor, i: int, j: int) -> Tensor:
        # Crop window (i, j) of an overlapping (rows x cols) grid from [3, H, W].
        _, H, W = image.shape
        rows, cols = self.grid
        th, tw = H / rows, W / cols
        cy, cx = (i + 0.5) * th, (j + 0.5) * tw
        half_h = th * (1.0 + self.overlap) / 2.0
        half_w = tw * (1.0 + self.overlap) / 2.0
        top = max(0, int(round(cy - half_h)))
        bottom = min(H, int(round(cy + half_h)))
        left = max(0, int(round(cx - half_w)))
        right = min(W, int(round(cx + half_w)))
        # Guard against degenerate 0-size crops (very small images / large grid).
        bottom = max(bottom, top + 1)
        right = max(right, left + 1)
        return image[:, top:bottom, left:right]

    # ----------------------------------------------------------------- encoders
    @torch.no_grad()
    def encode_image(self, images: Tensor) -> Tensor:
        """Encode a batch of images into L2-normalized CLIP embeddings.

        Parameters
        ----------
        images : Tensor
            ``[B, 3, h, w]`` in ``[0, 1]`` (any device). Resized to
            ``input_resolution`` if needed and CLIP-normalized internally.

        Returns
        -------
        Tensor
            ``[B, embed_dim]``, L2-normalized along the last dim.
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"encode_image 需要 [B, 3, h, w]，但拿到 {tuple(images.shape)}")
        images = images.to(device=self.device, dtype=self.compute_dtype)
        res = self.input_resolution
        if images.shape[-2:] != (res, res):
            images = F.interpolate(
                images, size=(res, res), mode="bicubic", align_corners=False, antialias=True
            )
        images = self._normalize(images)

        feats: List[Tensor] = []
        for start in range(0, images.shape[0], self.chunk_size):
            chunk = images[start : start + self.chunk_size]
            emb = self.model.encode_image(chunk)
            emb = F.normalize(emb.float(), dim=-1)
            feats.append(emb)
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, Sequence[str]]) -> Tensor:
        """Encode text prompt(s) into L2-normalized CLIP embeddings.

        Parameters
        ----------
        texts : str | Sequence[str]
            A single prompt or a list of prompts.

        Returns
        -------
        Tensor
            ``[T, embed_dim]`` (always 2-D, ``T == len(texts)``), L2-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        emb = self.model.encode_text(tokens)
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Encode a single image into a dense grid of CLIP embeddings.

        Parameters
        ----------
        image : Tensor
            ``[3, H, W]`` or ``[1, 3, H, W]`` in ``[0, 1]`` (mirrors how
            ``VGGEncoder`` is fed ``Camera.original_image``).

        Returns
        -------
        Tensor
            Dense CLIP feature map ``[embed_dim, rows, cols]`` (``self.grid``),
            each cell L2-normalized. This is the per-view GT analogous to
            ``Camera.vgg_features``.
        """
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError(f"forward 只接受單張影像，但拿到 batch {tuple(image.shape)}")
            image = image[0]
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"forward 需要 [3, H, W] 或 [1, 3, H, W]，但拿到 {tuple(image.shape)}")

        image = image.to(device=self.device, dtype=self.compute_dtype)
        rows, cols = self.grid
        windows = [
            self._resize_square(self._crop_cell(image, i, j))
            for i in range(rows)
            for j in range(cols)
        ]
        batch = torch.stack(windows, dim=0)  # [rows*cols, 3, R, R]
        emb = self.encode_image(batch)  # [rows*cols, D]
        # row-major (i*cols + j) -> [D, rows, cols]
        return emb.t().reshape(self.embed_dim, rows, cols).contiguous()
