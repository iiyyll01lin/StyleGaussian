"""SAM-region-pooled CLIP encoder for the language field (roadmap A1 fallback).

This is the **documented next step after MaskCLIP** for the LangSplat-style CLIP
language field. Where :class:`scene.clip_encoder.CLIPEncoder` (MaskCLIP path)
extracts per-patch dense tokens from the frozen ViT, this encoder instead:

* runs **SAM** (Segment Anything) automatic mask generation on the full image to
  get clean object/region masks;
* CLIP-encodes each region crop (background masked out) with the **frozen
  open_clip image tower** reused from a :class:`CLIPEncoder` (fp32, never bf16);
* paints each region's pooled CLIP embedding into a ``[D, gh, gw]`` grid (same
  ``14×14`` grid as MaskCLIP), so each grid cell carries the semantics of the
  SAM region that dominates it.

The **output contract is identical** to ``CLIPEncoder.forward`` — ``[D, gh, gw]``
with each cell L2-normalized — so ``cameras.extract_clip_features`` /
``train_language.py`` / the per-Gaussian field + ``clip_linear`` decoder /
relevancy / render all stay byte-for-byte the same downstream. The motivation is
*cleaner* object masks than the broad MaskCLIP field (truck↔road corr 0.46,
top-K truck precision 0.112): SAM gives object-shaped regions, so a single region
embedding fills the whole truck instead of bleeding into the road.

Caching
-------
SAM automatic mask generation is the expensive part. The final ``[D, gh, gw]``
GT grid is cached to ``cache_dir`` keyed by a content hash of the image plus the
encoder config, so re-training the field (or re-running with the same settings)
skips both SAM and CLIP.

Import safety
-------------
``segment_anything`` (and ``open_clip``, via ``CLIPEncoder``) are imported
*lazily inside* :meth:`SAMPooledEncoder.__init__`, so simply importing this module
never requires SAM/open_clip to be installed.

ROCm note: SAM + CLIP both run in fp32 on the gfx1151 APU; bf16 is explicitly
rejected (known gfx1151 bug, see HANDOFF §11).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scene.clip_encoder import CLIPEncoder

_SAM_MODEL_TYPES = {"vit_b", "vit_l", "vit_h"}


class SAMPooledEncoder(nn.Module):
    """Frozen SAM + CLIP encoder producing region-pooled dense grid features.

    Parameters
    ----------
    sam_checkpoint : str
        Path to a SAM checkpoint (``sam_vit_b_01ec64.pth`` etc.).
    sam_model_type : str
        SAM architecture matching the checkpoint: ``"vit_b"`` / ``"vit_l"`` /
        ``"vit_h"``.
    clip_model, clip_pretrained : str
        ``open_clip`` architecture / pretrained tag for the region CLIP tower
        (reused via an internal :class:`CLIPEncoder`).
    device : str | torch.device
        Device for both towers (``"cuda"`` on the gfx1151 box).
    grid : int | tuple[int, int]
        Output grid ``(rows, cols)``; defaults to ``(14, 14)`` to match the
        MaskCLIP field for apples-to-apples comparison. Output is ``[D, rows, cols]``.
    dtype : torch.dtype
        Compute dtype (``torch.float32`` default; ``float16`` ok; **never** bf16).
    input_resolution : int
        CLIP image-tower input resolution for region crops (224 for ViT-B/16).
    points_per_side : int
        SAM automatic-mask sampling density. Fewer points -> fewer, larger,
        cleaner regions (16 is a good default for object-level pooling).
    pred_iou_thresh, stability_score_thresh : float
        SAM mask-quality filters (passed through to ``SamAutomaticMaskGenerator``).
        min_region_area_frac : float
        Drop SAM regions smaller than this fraction of the image area (noise).
    per_pixel : bool
        If ``True``, output a **high-resolution per-pixel** GT grid
        ``[D, H//out_stride, W//out_stride]`` instead of the coarse ``grid``:
        each output cell is painted with the masked-crop CLIP embedding of the
        SAM region that *owns* that pixel (largest covering mask), giving sharp
        object-shaped GT instead of the coarse 14×14 max-coverage painting. This
        is roadmap A's ``sam_perpixel`` GT producer. ``False`` keeps the original
        coarse ``[D, grid]`` pooled behaviour (back-compat).
    out_stride : int
        Downsample factor from the full image for the per-pixel output grid
        (``per_pixel=True`` only). ``4`` → ``H/4 × W/4`` (e.g. 546×979 → 136×244).
    mask_bg : str
        How to treat the background of a region crop before CLIP: ``"black"``
        (zero out non-region pixels, object-centric) or ``"none"`` (plain bbox
        crop, keeps a little context).
    cache_dir : str | None
        Directory to cache the final ``[D, gh, gw]`` GT grids. ``None`` disables
        caching.
    chunk_size : int
        Max region crops CLIP-encoded per forward chunk.

    Attributes
    ----------
    embed_dim : int
        CLIP joint embedding dim ``D`` (512 for ViT-B/16) — drives the
        ``clip_linear`` decoder, identical to the MaskCLIP path.
    """

    embed_dim: int

    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        clip_model: str = "ViT-B-16",
        clip_pretrained: Union[str, None] = "openai",
        device: Union[str, torch.device] = "cuda",
        grid: Union[int, Tuple[int, int]] = (14, 14),
        dtype: torch.dtype = torch.float32,
        input_resolution: int = 224,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.90,
        min_region_area_frac: float = 0.0005,
        mask_bg: str = "black",
        cache_dir: Optional[str] = None,
        chunk_size: int = 32,
        per_pixel: bool = False,
        out_stride: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(grid, int):
            grid = (grid, grid)
        if len(grid) != 2 or grid[0] < 1 or grid[1] < 1:
            raise ValueError(f"grid 必須是 (rows, cols) 且都 >= 1，但拿到 {grid}")
        if dtype == torch.bfloat16:
            raise ValueError("gfx1151 已知 bf16 bug：請改用 fp32 或 fp16 (HANDOFF §11)。")
        if sam_model_type not in _SAM_MODEL_TYPES:
            raise ValueError(f"sam_model_type 必須是 {_SAM_MODEL_TYPES}，但拿到 {sam_model_type}")
        if mask_bg not in ("black", "none"):
            raise ValueError(f"mask_bg 必須是 'black' 或 'none'，但拿到 {mask_bg}")
        if not Path(sam_checkpoint).is_file():
            raise FileNotFoundError(
                f"找不到 SAM checkpoint：{sam_checkpoint}（請先下載，見 a1-sam STEP 1）。"
            )

        if int(out_stride) < 1:
            raise ValueError(f"out_stride 必須 >= 1，但拿到 {out_stride}")

        self.grid: Tuple[int, int] = (int(grid[0]), int(grid[1]))
        self.compute_dtype = dtype
        self.input_resolution = int(input_resolution)
        self.min_region_area_frac = float(min_region_area_frac)
        self.mask_bg = mask_bg
        self.per_pixel = bool(per_pixel)
        self.out_stride = int(out_stride)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # --- reuse the existing frozen open_clip tower (encode_image/text) -----
        self.clip = CLIPEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
            grid=self.grid,
            dtype=dtype,
            input_resolution=input_resolution,
            chunk_size=chunk_size,
        )
        self.embed_dim = self.clip.embed_dim

        # --- build SAM (lazy import; fp32) ------------------------------------
        try:
            from segment_anything import (  # noqa: PLC0415
                SamAutomaticMaskGenerator,
                sam_model_registry,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SAMPooledEncoder 需要 segment-anything：請先 `pip install segment-anything` "
                "(只有實際建立 SAMPooledEncoder 時才需要；單純 import 本模組不需要)。"
            ) from exc

        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam = sam.to(device=device)  # SAM runs fp32; never bf16 on gfx1151
        sam.eval()
        sam.requires_grad_(False)
        self.sam = sam
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        mode_tag = f"pp{self.out_stride}" if self.per_pixel else f"g{self.grid[0]}x{self.grid[1]}"
        self._cfg_tag = (
            f"{clip_model}_{clip_pretrained}_{mode_tag}"
            f"_pps{points_per_side}_{mask_bg}_{sam_model_type}"
        )

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.clip.device

    def _cache_path(self, image: Tensor) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Content hash of the image bytes (fp16 to be resolution/dtype-stable)
        # plus the encoder config tag.
        h = hashlib.md5()
        h.update(self._cfg_tag.encode("utf-8"))
        h.update(str(tuple(image.shape)).encode("utf-8"))
        h.update(image.detach().to("cpu", torch.float16).contiguous().numpy().tobytes())
        return self.cache_dir / f"{h.hexdigest()}.pt"

    def _region_crop(self, image: Tensor, seg: Tensor, bbox: Tuple[int, int, int, int]) -> Tensor:
        """Tight bbox crop of ``image`` for a SAM region, resized to R×R.

        ``image`` is ``[3, H, W]`` in [0, 1]; ``seg`` is ``[H, W]`` float in {0,1};
        ``bbox`` is SAM's XYWH. With ``mask_bg == "black"`` the non-region pixels
        are zeroed so CLIP sees an object-centric crop.
        """
        x0, y0, w, h = bbox
        _, H, W = image.shape
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(W, x0 + max(1, int(w))); y1 = min(H, y0 + max(1, int(h)))
        crop = image[:, y0:y1, x0:x1]
        if self.mask_bg == "black":
            crop = crop * seg[y0:y1, x0:x1].unsqueeze(0)
        res = self.input_resolution
        crop = F.interpolate(
            crop.unsqueeze(0), size=(res, res),
            mode="bicubic", align_corners=False, antialias=True,
        ).clamp(0.0, 1.0).squeeze(0)
        return crop

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Encode a single image into a SAM-region-pooled CLIP grid ``[D, gh, gw]``.

        Parameters
        ----------
        image : Tensor
            ``[3, H, W]`` or ``[1, 3, H, W]`` in ``[0, 1]`` (mirrors how
            ``CLIPEncoder.forward`` / ``VGGEncoder`` is fed ``original_image``).

        Returns
        -------
        Tensor
            ``[embed_dim, gh, gw]`` (``14×14`` by default), each cell
            L2-normalized — drop-in replacement for the MaskCLIP GT grid.
        """
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError(f"forward 只接受單張影像，但拿到 batch {tuple(image.shape)}")
            image = image[0]
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"forward 需要 [3, H, W] 或 [1, 3, H, W]，但拿到 {tuple(image.shape)}")

        D = self.embed_dim
        image = image.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        _, H, W = image.shape

        # Output grid: coarse (gh, gw) for the pooled mode, or a high-res
        # (H//out_stride, W//out_stride) per-pixel grid (roadmap A's sam_perpixel).
        if self.per_pixel:
            gh = max(1, H // self.out_stride)
            gw = max(1, W // self.out_stride)
        else:
            gh, gw = self.grid

        cache_path = self._cache_path(image)
        if cache_path is not None and cache_path.is_file():
            grid_emb = torch.load(str(cache_path), map_location=self.device)
            if tuple(grid_emb.shape) == (D, gh, gw):
                return grid_emb.to(device=self.device, dtype=torch.float32)

        img_np = (image.permute(1, 2, 0).contiguous().cpu().numpy() * 255.0).astype("uint8")
        masks = self.mask_generator.generate(img_np)  # list of dicts

        # whole-image CLIP embedding: fallback for grid cells no region covers.
        whole = self.clip.encode_image(image.unsqueeze(0))[0]  # [D], L2-normalized

        # filter tiny noise regions
        min_area = self.min_region_area_frac * float(H * W)
        masks = [m for m in masks if float(m.get("area", 0)) >= min_area]

        if not masks:
            grid_emb = whole.view(D, 1, 1).expand(D, gh, gw).contiguous()
            grid_emb = F.normalize(grid_emb, dim=0)
            if cache_path is not None:
                torch.save(grid_emb.to("cpu", torch.float16), str(cache_path))
            return grid_emb

        # --- per-region: coverage on the (gh, gw) grid + masked-crop CLIP emb ----
        covers: List[Tensor] = []
        crops: List[Tensor] = []
        areas: List[float] = []
        for m in masks:
            seg = torch.from_numpy(m["segmentation"]).to(self.device).float()  # [H, W]
            cov = F.adaptive_avg_pool2d(seg.view(1, 1, H, W), (gh, gw)).view(gh, gw)
            covers.append(cov)
            crops.append(self._region_crop(image, seg, tuple(m["bbox"])))
            areas.append(float(m.get("area", float(seg.sum().item()))))

        cov_stack = torch.stack(covers, dim=0)  # [R, gh, gw]
        region_emb = self.clip.encode_image(torch.stack(crops, dim=0))  # [R, D], L2-norm

        if self.per_pixel:
            # PER-PIXEL assignment (roadmap A): at this high resolution each output
            # cell is tiny, so paint it with the masked-crop embedding of the SAM
            # region that *owns* that pixel. Among masks covering a cell (cov>=0.5)
            # pick the one with the LARGEST area (object-level, not a sub-part), so
            # truck pixels get the whole-truck crop and road pixels the road crop —
            # a sharp object-shaped GT instead of the old coarse 14×14 painting.
            area_t = torch.tensor(areas, device=self.device, dtype=torch.float32)
            covered = cov_stack >= 0.5  # [R, gh, gw]
            score = torch.where(
                covered,
                area_t.view(-1, 1, 1).expand_as(cov_stack),
                torch.full_like(cov_stack, -1.0),
            )
            best_area, best = score.max(dim=0)  # prefer largest covering mask
            # cells with no strong cover -> fall back to highest-coverage region
            max_cov, best_cov = cov_stack.max(dim=0)
            no_strong = best_area < 0.0
            best = torch.where(no_strong, best_cov, best)
        else:
            # each cell -> the region with the highest coverage there (coarse pooled)
            max_cov, best = cov_stack.max(dim=0)  # [gh, gw], [gh, gw]

        grid_emb = region_emb[best.reshape(-1)].reshape(gh, gw, D).permute(2, 0, 1)  # [D, gh, gw]

        # cells no region covers at all -> whole-image embedding (rare at high res)
        uncovered = max_cov <= 1e-6  # [gh, gw]
        if bool(uncovered.any()):
            grid_emb = grid_emb.clone()
            grid_emb[:, uncovered] = whole.view(D, 1)

        grid_emb = F.normalize(grid_emb.contiguous(), dim=0)  # per-cell L2

        if cache_path is not None:
            torch.save(grid_emb.to("cpu", torch.float16), str(cache_path))
        return grid_emb
