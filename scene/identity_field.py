"""Per-Gaussian *identity* embedding field (Gaussian-Grouping / SAGA style).

This is the literature-blessed alternative to the per-pixel CLIP language field
(``sam_perpixel``), which plateaued at an honest-mask precision ~0.31: instead of
distilling CLIP *semantics* into every Gaussian (so "grouping" is entangled with
the broad, fuzzy CLIP direction), we learn a **semantic-free** low-dim identity
embedding per Gaussian and supervise it ONLY with 2D SAM instance masks via a
SAGA-style contrastive objective. Grouping is thereby *decoupled* from CLIP; a
text→group association step (separate, post-gate) re-attaches language later.

Design (fully additive — no existing setup/render path is modified):

* :class:`IdentityField` holds a ``[N, dim]`` embedding (``dim≈16``), mirroring
  the ``training_setup_language`` pattern in ``scene/gaussian_model.py`` (random
  init + Adam). It is attached to a recon ``GaussianModel`` *externally* by the
  trainer, so the per-Gaussian order matches the language / artistic fields 1:1.
* The gsplat feature path already splats arbitrary-dim per-Gaussian fields with
  gradients (``render(..., feature_linear=..., feature_override=...)``). We splat
  the embedding and pass :class:`IdentityRenderHead` as ``feature_linear`` — it
  alpha-normalizes the splatted features (``Σ Tα·e / Σ Tα``) into the per-pixel
  *expected* embedding and stashes the visibility weight ``sum_w`` for fg masking.
* :func:`contrastive_identity_loss` is a minimal **per-view** SAGA objective:
  pull pixels in the SAME SAM instance together, push DIFFERENT instances apart
  (cosine). No cross-view instance-id matching is needed — multi-view 3D
  consistency emerges because the same Gaussians are seen from many views.

Pure ``torch`` only (no SAM / open_clip / CUDA-specific imports), so importing
this module is cheap and unit-testable on CPU. fp32 on gfx1151 (NEVER bf16).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class IdentityRenderHead(nn.Module):
    """Alpha-normalizing render head, drop-in for ``feature_linear(features, sum_w)``.

    The :class:`~stylegaussian_patches.gsplat_backend.FeatureGaussianRasterizer`
    returns the *weighted* splat ``features = Σ_i (T_i α_i) e_i`` ``[D, H, W]`` and
    the per-pixel accumulated alpha ``sum_w = Σ_i T_i α_i`` ``[1, H, W]``. We return
    the per-pixel **expected embedding** ``features / sum_w`` so the result is a
    convex average of the contributing Gaussians' embeddings (background, where
    ``sum_w → 0``, stays near 0). ``sum_w`` is geometry-only (frozen), so dividing
    by it is a constant scale w.r.t. the embedding gradient.

    The last per-pixel ``sum_w`` (squeezed to ``[H, W]``) is cached on
    ``self.last_sum_w`` so the caller can build a foreground mask without a second
    render.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.last_sum_w: Optional[Tensor] = None

    def forward(self, features: Tensor, sum_w: Tensor) -> Tensor:
        sw = sum_w
        if sw.dim() == 3:  # [1, H, W] -> [H, W]
            sw = sw[0]
        self.last_sum_w = sw.detach()
        return features / sw.clamp_min(self.eps).unsqueeze(0)


class IdentityField(nn.Module):
    """Learnable per-Gaussian identity embedding ``[N, dim]`` (semantic-free).

    Parameters
    ----------
    n : int
        Number of Gaussians (must equal the recon ply point count so the field
        aligns 1:1 with the language / artistic per-Gaussian fields).
    dim : int
        Embedding dimensionality (``16`` by default — SAGA/Gaussian-Grouping use
        a small identity dim; this is splatted directly, no decoder).
    device, init_scale, seed :
        Init knobs. Random small-scale init mirrors ``training_setup_language``'s
        ``torch.randn`` field init.
    """

    def __init__(self, n: int, dim: int = 16, device="cuda",
                 init_scale: float = 0.1, seed: Optional[int] = None) -> None:
        super().__init__()
        self.dim = int(dim)
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            emb = torch.randn(n, dim, generator=gen) * init_scale
            emb = emb.to(device)
        else:
            emb = torch.randn(n, dim, device=device) * init_scale
        self.embedding = nn.Parameter(emb.requires_grad_(True))
        self.render_head = IdentityRenderHead()

    @property
    def features(self) -> Tensor:
        """The per-Gaussian field to splat via ``render(feature_override=...)``."""
        return self.embedding


def contrastive_identity_loss(
    emb_map: Tensor,
    label_map: Tensor,
    *,
    n_samples: int = 4096,
    margin: float = 0.0,
    push_weight: float = 1.0,
    valid_mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, dict]:
    """SAGA-style per-view 2D-mask contrastive loss on a rendered embedding map.

    Parameters
    ----------
    emb_map : Tensor
        Per-pixel embeddings ``[D, H, W]`` (alpha-normalized by the render head).
    label_map : Tensor
        Per-pixel SAM **instance** ids ``[H, W]`` (``int``); ``-1`` = unlabeled.
    n_samples : int
        Max pixels sampled per call (the pairwise sim matrix is ``S×S``).
    margin : float
        Different-instance pairs are pushed only while ``cos > margin`` (hinge).
    push_weight : float
        Weight on the inter-instance push term.
    valid_mask : Tensor, optional
        Extra ``[H, W]`` bool gate (e.g. ``sum_w >= thr``) AND-ed with ``label>=0``.

    Returns
    -------
    (loss, stats) : the scalar loss and a small diagnostics dict. ``loss`` keeps a
    graph to ``emb_map``; if too few valid pixels it is ``emb_map.sum()*0`` (zero,
    still differentiable) so the training loop never crashes on empty views.
    """
    D, H, W = emb_map.shape
    lab = label_map.reshape(-1)
    e = emb_map.reshape(D, -1).permute(1, 0)  # [P, D]
    valid = lab >= 0
    if valid_mask is not None:
        valid = valid & valid_mask.reshape(-1)
    idx = valid.nonzero(as_tuple=False).reshape(-1)
    if idx.numel() < 2:
        return emb_map.sum() * 0.0, {"n_valid": int(idx.numel())}
    if idx.numel() > n_samples:
        perm = torch.randperm(idx.numel(), device=emb_map.device, generator=generator)[:n_samples]
        idx = idx[perm]
    es = F.normalize(e[idx], dim=1)  # [S, D]
    ls = lab[idx]                    # [S]
    sim = es @ es.t()                # [S, S] cosine
    same = ls[:, None] == ls[None, :]
    eye = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
    same_off = same & ~eye
    diff = (~same) & ~eye
    pull = (1.0 - sim)[same_off]
    push = F.relu(sim - margin)[diff]
    loss_pull = pull.mean() if pull.numel() else sim.new_zeros(())
    loss_push = push.mean() if push.numel() else sim.new_zeros(())
    loss = loss_pull + push_weight * loss_push
    stats = {
        "n_valid": int(idx.numel()),
        "n_instances": int(torch.unique(ls).numel()),
        "loss_pull": float(loss_pull.detach()),
        "loss_push": float(loss_push.detach()),
    }
    return loss, stats


@torch.no_grad()
def compute_knn_indices(xyz: Tensor, k: int = 8, chunk: int = 2048) -> Tensor:
    """3D K-nearest-neighbor indices for every Gaussian (self excluded).

    A one-time geometric precompute (the recon point cloud is frozen): for each
    Gaussian we find the indices of its ``k`` spatially closest neighbors. Done in
    row-chunks of a brute-force ``cdist`` so the full ``N×N`` distance matrix is
    never materialized; on a single APU ``N≈3.3e5`` with ``chunk=2048`` peaks at a
    few GiB and finishes in seconds. Pure ``torch`` (runs on whatever device ``xyz``
    lives on); the result is cached to disk by the trainer.

    Parameters
    ----------
    xyz : Tensor
        ``[N, 3]`` Gaussian centers.
    k : int
        Neighbors per Gaussian.
    chunk : int
        Rows processed per ``cdist`` block (memory/throughput knob).

    Returns
    -------
    Tensor
        ``[N, k]`` ``long`` neighbor indices (self is masked out via ``+inf``).
    """
    N = int(xyz.shape[0])
    device = xyz.device
    k = max(1, min(int(k), N - 1))
    knn = torch.empty(N, k, dtype=torch.long, device=device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d = torch.cdist(xyz[s:e], xyz)  # [m, N]
        rows = torch.arange(e - s, device=device)
        d[rows, torch.arange(s, e, device=device)] = float("inf")  # mask self
        knn[s:e] = d.topk(k, largest=False).indices
    return knn


def knn_grouping_loss(
    embedding: Tensor,
    knn_idx: Tensor,
    *,
    sample: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    detach_neighbors: bool = False,
) -> Tuple[Tensor, dict]:
    """3D KNN feature-smoothness / grouping regularizer (Gaussian-Grouping ingredient).

    For each (sampled) Gaussian, pull its identity embedding toward its 3D nearest
    neighbors' embeddings in **cosine** geometry (matching
    :func:`contrastive_identity_loss`). Acting directly on the per-Gaussian field
    — not the rendered map — this enforces *spatial* coherence: nearby Gaussians
    share an identity, which suppresses the per-view speckle that fragments a single
    object (the truck) into many clusters. It complements, and is summed with, the
    SAGA per-view contrastive loss (which still does the instance separation).

    Parameters
    ----------
    embedding : Tensor
        The learnable ``[N, dim]`` per-Gaussian field (``IdentityField.embedding``).
    knn_idx : Tensor
        ``[N, k]`` ``long`` neighbor indices from :func:`compute_knn_indices`.
    sample : int, optional
        If set and ``< N``, regularize a random subset of ``sample`` Gaussians per
        call (stochastic, cheap). ``None`` uses all ``N``.
    generator : torch.Generator, optional
        RNG for the subsample (reproducibility).
    detach_neighbors : bool
        If ``True``, stop gradient through the neighbor embeddings (pull only the
        anchor); default pulls both ends symmetrically.

    Returns
    -------
    (loss, stats) : scalar mean ``1 - cos(anchor, neighbor)`` over the sampled
    anchor×k pairs, plus a small diagnostics dict.
    """
    N, K = int(knn_idx.shape[0]), int(knn_idx.shape[1])
    device = embedding.device
    if sample is not None and 0 < sample < N:
        sel = torch.randperm(N, device=device, generator=generator)[:sample]
    else:
        sel = torch.arange(N, device=device)
    anchors = F.normalize(embedding[sel], dim=1)            # [S, D]
    nbr = embedding[knn_idx[sel]]                            # [S, K, D]
    if detach_neighbors:
        nbr = nbr.detach()
    nbr = F.normalize(nbr, dim=2)                            # [S, K, D]
    cos = (anchors.unsqueeze(1) * nbr).sum(dim=2)            # [S, K]
    loss = (1.0 - cos).mean()
    stats = {
        "knn_sample": int(sel.numel()),
        "knn_k": K,
        "knn_cos_mean": float(cos.mean().detach()),
    }
    return loss, stats
