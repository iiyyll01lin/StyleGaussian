"""SigLIP2 dense encoder (MAP-head value-bypass) for the language feature field.

Phase-2 of the stronger-VLM-signal experiment. :class:`scene.clip_encoder.CLIPEncoder`
extracts MaskCLIP per-patch dense tokens from a *native* open_clip ``VisionTransformer``
(CLS-token pooling): in the last attention block it keeps only the value→out projection
of each patch token and drops the query·key pooling, then applies ``ln_post`` + ``proj``.
That recipe assumes the joint image/text embedding is reached by a shared per-token
projection — true for OpenAI/DFN/LAION CLIP, **false** for SigLIP/SigLIP2.

SigLIP(2) in open_clip is a :class:`TimmModel` whose ``trunk`` is a timm
``VisionTransformer`` with **no CLS token** (``num_prefix_tokens == 0``) that pools to
the image embedding through a **MAP head** (timm ``AttentionPoolLatent``): a single
learned latent query attends over the patch tokens, and the pooled vector is the joint
image embedding (``timm_proj='none'`` → the joint space *is* the trunk width, 768 for
ViT-B/16; ``fc_norm`` and ``head`` are ``Identity``).

The MaskCLIP idea transfers cleanly to the MAP head: the pooled image vector is
``proj( Σ_i a_i · v_i ) (+ mlp residual)`` where ``v_i`` is the **value projection** of
patch token ``i`` and ``a_i`` the latent-query softmax weight. We **bypass the
latent-query softmax** (the cross-patch mixing) and the mlp residual, and keep the
per-patch ``proj(v_i)`` — giving each spatial token its *own* location-specific feature
in the same joint space the text tower aligns to. Concretely, with
``kv = attn_pool.kv(tokens)`` (``[B,N,2C]``, ``[k‖v]``) the values are ``kv[..., C:]`` and
the dense map is ``attn_pool.proj(values)`` reshaped to ``[D, gh, gw]``.

Output contract is identical to ``CLIPEncoder`` (``[D, gh, gw]``, each cell
L2-normalized) so ``cameras.extract_clip_features`` / ``train_language.py`` / the field +
``clip_linear`` decoder / relevancy / eval are all unchanged downstream. fp32 / fp16 only
(**never** bf16 on gfx1151). Needs ``open_clip`` + ``transformers`` (SigLIP HF tokenizer);
both imported lazily so importing this module never requires them.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# SigLIP image normalization (inception-style); used as a fallback if the exact
# mean/std cannot be read off the model's preprocess transform.
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


class SigLIPDenseEncoder(nn.Module):
    """Frozen SigLIP2 image/text encoder producing dense grid features (MAP-bypass).

    Parameters
    ----------
    model_name : str
        ``open_clip`` SigLIP(2) architecture, e.g. ``"ViT-B-16-SigLIP2-256"``
        (joint dim 768, 16×16 patch grid @ 256).
    pretrained : str | None
        ``open_clip`` pretrained tag, e.g. ``"webli"``. ``None`` → random weights
        (offline smoke / unit-test; no network).
    device : str | torch.device
        Device to place the model on (use ``"cuda"`` on the gfx1151 box).
    input_resolution : int
        Square resolution each image is resized to before the SigLIP tower. MUST
        match the checkpoint's training resolution (SigLIP2-256 → 256), because the
        timm trunk uses a fixed-size positional embedding (no dynamic resize).
    dtype : torch.dtype
        Compute dtype for the SigLIP tower. ``torch.float32`` (default) or
        ``torch.float16``; **never** ``torch.bfloat16`` on gfx1151.

    Attributes
    ----------
    embed_dim : int
        The SigLIP joint image/text embedding dimension ``D`` (768 for ViT-B/16).
    model : nn.Module
        The underlying frozen ``open_clip`` ``TimmModel`` (registered submodule).
    """

    embed_dim: int

    def __init__(
        self,
        model_name: str = "ViT-B-16-SigLIP2-256",
        pretrained: Union[str, None] = "webli",
        device: Union[str, torch.device] = "cuda",
        input_resolution: int = 256,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if input_resolution < 1:
            raise ValueError(f"input_resolution 必須 >= 1，但拿到 {input_resolution}")
        if dtype == torch.bfloat16:
            raise ValueError("gfx1151 已知 bf16 bug：請改用 fp32 或 fp16 (HANDOFF §11)。")

        self.model_name = model_name
        self.pretrained = pretrained
        self.input_resolution = int(input_resolution)
        self.compute_dtype = dtype

        try:
            import open_clip  # noqa: F401  (lazy on purpose)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SigLIPDenseEncoder 需要 open_clip：請先 `pip install open_clip_torch`。"
            ) from exc

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model = model.to(device=device, dtype=dtype)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        # SigLIP uses an HF (sentencepiece) tokenizer → needs `transformers`
        # (+`sentencepiece`) AND the tokenizer files in the HF cache. Note: under
        # HF_HUB_OFFLINE=1 this raises if the tokenizer was never fetched online.
        try:
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SigLIP tokenizer 需要 `transformers`(+`sentencepiece`)：請先安裝。"
            ) from exc

        # --- locate the timm trunk + MAP head, validate the architecture --------
        visual = model.visual
        trunk = getattr(visual, "trunk", None)
        if trunk is None:
            raise RuntimeError(
                f"{model_name} 不是 TimmModel trunk 架構（visual 無 .trunk）；"
                "SigLIPDenseEncoder 只支援 SigLIP/SigLIP2 的 timm-trunk MAP-head。"
            )
        attn_pool = getattr(trunk, "attn_pool", None)
        if attn_pool is None or not hasattr(attn_pool, "kv") or not hasattr(attn_pool, "proj"):
            raise RuntimeError(
                f"{model_name} 的 trunk 沒有 MAP head（AttentionPoolLatent.kv/proj）；"
                "此 dense 路徑需要 SigLIP 的 attentional-pool head。"
            )
        if int(getattr(trunk, "num_prefix_tokens", 0)) != 0:
            raise RuntimeError(
                "SigLIPDenseEncoder 假設無 CLS/prefix token（SigLIP num_prefix_tokens==0）；"
                f"但 {model_name} 有 {trunk.num_prefix_tokens} 個 prefix token。"
            )
        self._trunk = trunk
        self._attn_pool = attn_pool
        self._pool_dim = int(attn_pool.kv.in_features)  # == joint width C

        self.embed_dim = self._resolve_embed_dim(open_clip, model_name, device)

        mean, std = self._resolve_normalization(preprocess_val)
        self.register_buffer("pixel_mean", torch.tensor(mean, device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std, device=device).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ helpers
    def _resolve_embed_dim(self, open_clip, model_name: str, device) -> int:
        cfg = open_clip.get_model_config(model_name)
        if cfg is not None and "embed_dim" in cfg:
            return int(cfg["embed_dim"])
        with torch.no_grad():
            tokens = self.tokenizer(["x"]).to(device)
            return int(self.model.encode_text(tokens).shape[-1])

    @staticmethod
    def _resolve_normalization(preprocess_val) -> Tuple[Sequence[float], Sequence[float]]:
        transforms = getattr(preprocess_val, "transforms", [])
        for t in transforms:
            if type(t).__name__ == "Normalize":
                return tuple(float(v) for v in t.mean), tuple(float(v) for v in t.std)
        return _SIGLIP_MEAN, _SIGLIP_STD

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _normalize(self, images: Tensor) -> Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    # ----------------------------------------------------------------- encoders
    @torch.no_grad()
    def encode_image(self, images: Tensor) -> Tensor:
        """Encode a batch ``[B, 3, h, w]`` in ``[0,1]`` → ``[B, embed_dim]`` (L2-norm).

        Uses the model's *native* MAP-head pooling (full image embedding), matching
        ``CLIPEncoder.encode_image``'s contract.
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
        emb = self.model.encode_image(images)
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, Sequence[str]]) -> Tensor:
        """Encode text prompt(s) → ``[T, embed_dim]`` (always 2-D, L2-normalized)."""
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        emb = self.model.encode_text(tokens)
        return F.normalize(emb.float(), dim=-1)

    # ------------------------------------------------- MAP-head dense path
    @torch.no_grad()
    def _dense_tokens(self, images: Tensor) -> Tuple[Tensor, int, int]:
        """SigLIP MAP-bypass per-patch dense tokens from the frozen timm trunk.

        Runs ``trunk.forward_features`` (patch-embed + pos-embed + blocks + final
        ``norm``) to get patch tokens ``[B, N, C]`` (no CLS), then bypasses the
        latent-query attention pooling: takes each patch token's **value**
        projection ``v_i`` (the ``v`` half of ``attn_pool.kv``) and applies
        ``attn_pool.proj`` per token, dropping the cross-patch softmax mixing and the
        mlp residual. ``fc_norm`` / ``head`` are ``Identity`` for SigLIP, so the
        result already lives in the joint image/text space.

        Returns ``(tokens, gh, gw)`` with ``tokens`` ``[B, N, C]`` (float32, not yet
        L2-normalized) and ``gh, gw`` the patch grid (``16×16`` for ViT-B/16 @ 256).
        """
        trunk = self._trunk
        feats = trunk.forward_features(images)  # [B, N, C], post final norm, no CLS
        if feats.dim() != 3:
            raise RuntimeError(f"trunk.forward_features 期望 [B,N,C]，但拿到 {tuple(feats.shape)}")
        ap = self._attn_pool
        C = self._pool_dim
        kv = ap.kv(feats)            # [B, N, 2C]  == [k ‖ v] over all heads
        v = kv[..., C:]              # [B, N, C]   value (all heads concatenated)
        tokens = ap.proj(v)          # [B, N, C]   MaskCLIP MAP-bypass per-patch
        tokens = trunk.fc_norm(tokens)  # Identity for SigLIP, kept for generality
        tokens = tokens.float()

        n_patches = tokens.shape[1]
        grid = getattr(trunk.patch_embed, "grid_size", None)
        if grid is not None and int(grid[0]) * int(grid[1]) == n_patches:
            gh, gw = int(grid[0]), int(grid[1])
        else:
            side = int(round(n_patches ** 0.5))
            if side * side != n_patches:
                raise RuntimeError(
                    f"無法把 {n_patches} 個 patch token 還原成方形 grid；請確認輸入為方形。"
                )
            gh = gw = side
        return tokens, gh, gw

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Encode one image ``[3,H,W]`` (or ``[1,3,H,W]``) → ``[embed_dim, gh, gw]``.

        Drop-in for ``CLIPEncoder.forward``: each cell L2-normalized, same
        ``[D, gh, gw]`` contract (``16×16`` for ViT-B/16 SigLIP2 @ 256).
        """
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError(f"forward 只接受單張影像，但拿到 batch {tuple(image.shape)}")
            image = image[0]
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"forward 需要 [3, H, W] 或 [1, 3, H, W]，但拿到 {tuple(image.shape)}")

        image = image.to(device=self.device, dtype=self.compute_dtype)
        res = self.input_resolution
        img = F.interpolate(
            image.unsqueeze(0), size=(res, res),
            mode="bicubic", align_corners=False, antialias=True,
        )  # [1, 3, R, R]
        img = self._normalize(img)
        tokens, gh, gw = self._dense_tokens(img)  # [1, gh*gw, D]
        tokens = F.normalize(tokens[0], dim=-1)   # [gh*gw, D], per-cell L2
        return tokens.t().reshape(self.embed_dim, gh, gw).contiguous()
