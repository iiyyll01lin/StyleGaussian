"""CLIP encoder for the LangSplat-style language feature field (MaskCLIP dense GT).

This mirrors the role of :mod:`scene.VGG` in the existing pipeline. Where
``VGGEncoder`` turns an image into a dense conv feature map (``relu3_1`` →
``[256, H, W]``) that supervises the per-Gaussian VGG style field, ``CLIPEncoder``
turns an image into a *grid of CLIP image embeddings* ``[D, gh, gw]`` that can
supervise a per-Gaussian CLIP **language** field in exactly the same way
(distill a low-dim field + linear decode back to CLIP dim, see
``train_feature.py`` / ``GaussianModel.training_setup_feature``).

MaskCLIP dense path (:meth:`CLIPEncoder.forward`)
-------------------------------------------------
The original "Lite" path cropped each grid cell and ran **whole-window** CLIP on
it; the resulting per-cell embeddings were the *image-level* CLIP vector of each
window, which is **not** spatially discriminative (truck≈road, IoU≈chance — see
the language localization report and docs/06). This version instead extracts
**MaskCLIP-style per-patch dense tokens** (Zhou et al., *Extract Free Dense
Labels from CLIP*, ECCV 2022) directly from the frozen open_clip ViT:

* run the ViT through all but the **last** transformer block normally;
* in the last block, **bypass the query·key attention pooling** and keep only the
  value→out-projection of each patch token (drop the residual + MLP), so each
  spatial token carries *its own* local semantics instead of the globally pooled
  image vector;
* apply the final ``ln_post`` + visual ``proj`` per token and reshape the patch
  tokens to ``[D, gh, gw]`` (``14×14`` for ViT-B/16 @ 224, ``D=512``).

The **output contract is unchanged** (still ``[D, gh, gw]``, each cell
L2-normalized), so ``cameras.extract_clip_features`` / ``train_language.py`` /
the field + ``clip_linear`` decoder / relevancy all stay byte-for-byte the same
downstream. Needs nothing beyond the frozen CLIP image tower — no SAM, no new
deps — and is ROCm-safe (fp32 / fp16, **no bf16** per the gfx1151 notes).

The full LangSplat hierarchy (SAM 3-level masks + per-region CLIP + scene
autoencoder) remains **out of scope** here and documented as a future fallback.

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
        dense_mode: str = "maskclip",
        ms_tilings: Sequence[int] = (1, 2, 3),
        ms_overlap: float = 0.3,
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
        if dense_mode not in ("maskclip", "multiscale"):
            raise ValueError(
                f"dense_mode 必須是 'maskclip' 或 'multiscale'，但拿到 {dense_mode!r}"
            )
        ms_tilings = tuple(int(t) for t in ms_tilings)
        if len(ms_tilings) < 1 or any(t < 1 for t in ms_tilings):
            raise ValueError(f"ms_tilings 必須是 >=1 的整數序列，但拿到 {ms_tilings}")
        if not (0.0 <= ms_overlap < 1.0):
            raise ValueError(f"ms_overlap 必須落在 [0, 1)，但拿到 {ms_overlap}")

        self.dense_mode = dense_mode
        self.ms_tilings: Tuple[int, ...] = ms_tilings
        self.ms_overlap = float(ms_overlap)
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

    # ------------------------------------------------------- MaskCLIP dense path
    @staticmethod
    def _embeds_dense(visual, images: Tensor) -> Tensor:
        """``visual._embeds`` but with bicubic-resampled positional embeddings.

        open_clip's ``VisionTransformer._embeds`` adds a *fixed-size* positional
        embedding (1 CLS + ``grid0**2`` patches, trained at the native resolution,
        e.g. 14×14 for ViT-B/16 @ 224), so feeding a larger ``input_resolution``
        raises a token-count mismatch. For dense MaskCLIP GT at a higher grid we
        bicubically resample the patch positional embeddings to the new grid (the
        standard ViT / DeiT / MaskCLIP recipe). At the native resolution the
        resample is an identity, so the 224 path stays byte-for-byte unchanged.
        """
        x = visual.conv1(images)  # [B, width, gh, gw]
        gh, gw = int(x.shape[-2]), int(x.shape[-1])
        width = x.shape[1]
        x = x.reshape(x.shape[0], width, gh * gw).permute(0, 2, 1)  # [B, gh*gw, width]

        cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+gh*gw, width]

        pe = visual.positional_embedding.to(x.dtype)  # [1+grid0**2, width]
        n_patch = gh * gw
        if pe.shape[0] != n_patch + 1:
            cls_pe, patch_pe = pe[:1], pe[1:]  # [1, width], [grid0**2, width]
            s0 = int(round(patch_pe.shape[0] ** 0.5))
            if s0 * s0 != patch_pe.shape[0]:
                raise RuntimeError(
                    f"無法把 {patch_pe.shape[0]} 個位置編碼還原成方形 grid 以重採樣。"
                )
            patch_pe = patch_pe.reshape(1, s0, s0, width).permute(0, 3, 1, 2)  # [1,width,s0,s0]
            patch_pe = F.interpolate(
                patch_pe, size=(gh, gw), mode="bicubic", align_corners=False, antialias=True,
            )
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(n_patch, width)
            pe = torch.cat([cls_pe, patch_pe], dim=0)
        x = x + pe

        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        return x

    @torch.no_grad()
    def _dense_tokens(self, images: Tensor) -> Tuple[Tensor, int, int]:
        """MaskCLIP per-patch dense tokens from the frozen open_clip ViT.

        Runs the ViT through all but the last transformer block normally, then in
        the last block **drops the query·key attention pooling** and keeps only
        the value→out-projection of each patch token (plus the block's layer-scale
        ``ls_1`` if any), discarding the residual + MLP. Applies the final
        ``ln_post`` + visual ``proj`` per token. This is the standard MaskCLIP
        reformulation that turns CLIP's globally-pooled tower into a dense,
        spatially-localized feature extractor without any training.

        Parameters
        ----------
        images : Tensor
            ``[B, 3, R, R]`` already CLIP-normalized and on the model device/dtype
            (``R == input_resolution``).

        Returns
        -------
        (tokens, gh, gw)
            ``tokens`` is ``[B, gh*gw, embed_dim]`` (patch tokens only, CLS
            dropped), float32, **not** yet L2-normalized; ``gh, gw`` is the patch
            grid (``14×14`` for ViT-B/16 @ 224).
        """
        visual = self.model.visual
        if getattr(visual, "attn_pool", None) is not None:
            raise RuntimeError(
                "MaskCLIP dense 路徑僅支援 CLS-token pooling 的 ViT（如 ViT-B-16）；"
                "此模型用 attn_pool，請改用支援的架構。"
            )

        x = self._embeds_dense(visual, images)  # [B, 1+N, width]; pos-embed resampled
        blocks = visual.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)

        # --- last block: MaskCLIP value-only bypass (no q·k attention) ----------
        last = blocks[-1]
        x_ln = last.ln_1(x)  # [B, 1+N, width]
        attn = last.attn
        width = x_ln.shape[-1]
        w_in = attn.in_proj_weight  # [3*width, width] (q, k, v stacked)
        b_in = attn.in_proj_bias
        w_v = w_in[2 * width : 3 * width]
        b_v = None if b_in is None else b_in[2 * width : 3 * width]
        v = F.linear(x_ln, w_v, b_v)  # [B, 1+N, width]
        v = attn.out_proj(v)  # [B, 1+N, width]
        # MaskCLIP keeps only the value path (drop residual + MLP). Apply the
        # block's layer-scale ls_1 if present (Identity for ViT-B/16, so a no-op).
        x = last.ls_1(v) if hasattr(last, "ls_1") else v

        x = visual.ln_post(x)  # [B, 1+N, width]
        if visual.proj is not None:
            x = x @ visual.proj  # [B, 1+N, embed_dim]

        tokens = x[:, 1:, :].float()  # drop CLS -> [B, N, embed_dim]
        n_patches = tokens.shape[1]
        grid = getattr(visual, "grid_size", None)
        if grid is not None and int(grid[0]) * int(grid[1]) == n_patches:
            gh, gw = int(grid[0]), int(grid[1])
        else:  # square fallback (we resize inputs to a square R x R)
            side = int(round(n_patches**0.5))
            if side * side != n_patches:
                raise RuntimeError(
                    f"無法把 {n_patches} 個 patch token 還原成方形 grid；"
                    f"請確認輸入已 resize 成方形。"
                )
            gh = gw = side
        return tokens, gh, gw

    def _square_grid(self, image: Tensor) -> Tensor:
        """MaskCLIP dense grid ``[D, gh, gw]`` from one ``[3, H, W]`` image.

        Squishes the whole image to a single ``input_resolution`` square and runs
        the value-projection dense path once. This is the original (``dense_mode
        == "maskclip"``) GT, factored out so the multi-scale fusion can reuse it.
        """
        res = self.input_resolution
        img = F.interpolate(
            image.unsqueeze(0), size=(res, res),
            mode="bicubic", align_corners=False, antialias=True,
        )  # [1, 3, R, R]
        img = self._normalize(img)
        tokens, gh, gw = self._dense_tokens(img)  # [1, gh*gw, D]
        tokens = F.normalize(tokens[0], dim=-1)  # [gh*gw, D], per-cell L2
        return tokens.t().reshape(self.embed_dim, gh, gw).contiguous()

    def _crop_tile(self, image: Tensor, i: int, j: int, n: int) -> Tensor:
        """Crop tile ``(i, j)`` of an overlapping ``n x n`` tiling from ``[3,H,W]``."""
        _, H, W = image.shape
        th, tw = H / n, W / n
        cy, cx = (i + 0.5) * th, (j + 0.5) * tw
        half_h = th * (1.0 + self.ms_overlap) / 2.0
        half_w = tw * (1.0 + self.ms_overlap) / 2.0
        top = max(0, int(round(cy - half_h)))
        bottom = min(H, int(round(cy + half_h)))
        left = max(0, int(round(cx - half_w)))
        right = min(W, int(round(cx + half_w)))
        bottom = max(bottom, top + 1)
        right = max(right, left + 1)
        return image[:, top:bottom, left:right]

    @torch.no_grad()
    def _multiscale_grid(self, image: Tensor) -> Tensor:
        """Multi-scale MaskCLIP fusion grid ``[D, G, G]`` from one ``[3, H, W]``.

        Runs the value-projection dense path at several **crop scales** and fuses
        the per-location features in a common ``G x G`` grid (``G`` = the native
        full-image patch grid, ``14`` for ViT-B/16 @ 224). For each tiling ``n``
        in ``ms_tilings`` we crop the image into an overlapping ``n x n`` grid,
        resize **each crop** to ``input_resolution`` square and extract its
        MaskCLIP dense tokens, then paint the crop's tokens into its footprint of
        the global grid (averaging where scales/crops overlap), and finally
        re-normalize per cell.

        Motivation: vanilla MaskCLIP only de-pools the **last** attention layer —
        every patch token has still globally attended across all earlier layers,
        so a token over the truck is contaminated by road/sky context (the
        sharpened root cause: whole-image CLIP token semantics mix truck and
        road). A crop centred on the truck contains *mostly* truck, so its
        per-patch tokens carry cleaner local semantics; fusing the global (1x1)
        scale keeps scene context. This is a different lever from raising the
        resolution (res448) or region pooling (SAM), both of which kept the
        full-image receptive field.
        """
        # Global (1x1) grid sets the fused output resolution G x G.
        base = self._square_grid(image)  # [D, G, G] (already per-cell L2)
        D, G, _ = base.shape
        acc = base.clone()
        cnt = torch.ones((G, G), device=base.device, dtype=base.dtype)

        for n in self.ms_tilings:
            if n == 1:
                continue  # already counted as `base`
            for i in range(n):
                r0 = int(round(i * G / n))
                r1 = max(r0 + 1, int(round((i + 1) * G / n)))
                for j in range(n):
                    c0 = int(round(j * G / n))
                    c1 = max(c0 + 1, int(round((j + 1) * G / n)))
                    crop = self._crop_tile(image, i, j, n)  # [3, hc, wc]
                    grid = self._square_grid(crop)  # [D, G, G] per-cell L2
                    fh, fw = r1 - r0, c1 - c0
                    g_rs = F.interpolate(
                        grid.unsqueeze(0), size=(fh, fw),
                        mode="bilinear", align_corners=False,
                    )[0]  # [D, fh, fw]
                    acc[:, r0:r1, c0:c1] = acc[:, r0:r1, c0:c1] + g_rs
                    cnt[r0:r1, c0:c1] = cnt[r0:r1, c0:c1] + 1.0

        fused = acc / cnt.unsqueeze(0)  # [D, G, G]
        fused = F.normalize(fused, dim=0)  # per-cell L2
        return fused.contiguous()

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Encode a single image into a dense grid of CLIP embeddings (MaskCLIP).

        Parameters
        ----------
        image : Tensor
            ``[3, H, W]`` or ``[1, 3, H, W]`` in ``[0, 1]`` (mirrors how
            ``VGGEncoder`` is fed ``Camera.original_image``).

        Returns
        -------
        Tensor
            Dense CLIP feature map ``[embed_dim, gh, gw]`` (``14×14`` for
            ViT-B/16), each cell L2-normalized. Drop-in replacement for the old
            grid-window GT: same ``[D, gh, gw]`` contract.

            * ``dense_mode == "maskclip"`` (default): the whole image is resized to
              a single ``input_resolution`` square and run through the ViT **once**
              (value-projection dense path).
            * ``dense_mode == "multiscale"``: the value-projection grid is fused
              across crop scales (``ms_tilings``) to suppress global-attention
              contamination — see :meth:`_multiscale_grid`.
        """
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError(f"forward 只接受單張影像，但拿到 batch {tuple(image.shape)}")
            image = image[0]
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"forward 需要 [3, H, W] 或 [1, 3, H, W]，但拿到 {tuple(image.shape)}")

        image = image.to(device=self.device, dtype=self.compute_dtype)
        if self.dense_mode == "multiscale":
            return self._multiscale_grid(image)
        return self._square_grid(image)
