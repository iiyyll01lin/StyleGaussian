"""Language-driven localized stylization: text query → relevancy → soft mask.

Given a per-Gaussian CLIP language field (``[N, Dclip]``, produced by the
Phase-2 feature-field training that distills ``Camera.clip_features``), this
module turns a natural-language query into a soft per-Gaussian mask
``M[N] ∈ [0, 1]``. Because the mask lives on the shared 3D Gaussians, it is
inherently multi-view consistent — that is the whole point of doing this in 3D
rather than per-frame 2D segmentation.

Relevancy follows the LERF / LangSplat convention: instead of a raw cosine
similarity (whose absolute scale is uncalibrated), we score the query against a
small set of *canonical negative* phrases ("object" / "things" / "stuff" /
"texture") and take, per Gaussian, the **pairwise softmax** probability of the
query beating its *hardest* negative. The score is therefore in ``(0, 1)`` and
``> 0.5`` means "more relevant to the query than to that canonical concept".

The blend (owned by a Phase-2 worker in ``render.py``) is then::

    final[n] = M[n] * stylized_rgb[n] + (1 - M[n]) * original_rgb[n]

Import safety
-------------
Only ``torch`` is imported at module scope; the relevancy/mask math is pure
torch and runs on CPU or GPU. The single CLIP touch-point, :func:`encode_text`,
*delegates* to a ``CLIPEncoder`` instance passed by the caller, so importing
this module never requires ``open_clip``.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor

# LERF / LangSplat canonical negatives. The query relevancy is computed relative
# to these generic concepts so its scale is calibrated and comparable.
CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture")


def _l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_relevancy(
    per_gaussian_clip: Tensor,
    text_embed: Tensor,
    canonical_negatives: Tensor,
    softmax_temp: float = 10.0,
) -> Tensor:
    """LERF/LangSplat relevancy of each Gaussian to a text query.

    Parameters
    ----------
    per_gaussian_clip : Tensor
        Per-Gaussian CLIP features ``[N, Dclip]`` (the decoded language field).
        Need not be normalized — this function normalizes internally.
    text_embed : Tensor
        Query CLIP embedding, ``[Dclip]`` or ``[1, Dclip]``.
    canonical_negatives : Tensor
        Negative-phrase CLIP embeddings ``[K, Dclip]`` (e.g. encode
        :data:`CANONICAL_NEGATIVES`). Must have ``K >= 1``.
    softmax_temp : float
        Temperature applied to the cosine similarities before the pairwise
        softmax (LERF uses 10.0). Larger → sharper relevancy.

    Returns
    -------
    Tensor
        Relevancy scores ``[N]`` in ``(0, 1)``. For each Gaussian this is the
        minimum over negatives of ``softmax(temp·[sim_pos, sim_neg_k])[pos]``,
        i.e. the probability of the query beating its hardest negative.
    """
    if per_gaussian_clip.dim() != 2:
        raise ValueError(
            f"per_gaussian_clip 需要 [N, Dclip]，但拿到 {tuple(per_gaussian_clip.shape)}"
        )
    text_embed = text_embed.reshape(-1)
    if text_embed.shape[0] != per_gaussian_clip.shape[1]:
        raise ValueError(
            f"text_embed 維度 {text_embed.shape[0]} 與 per_gaussian_clip 的 "
            f"Dclip={per_gaussian_clip.shape[1]} 不符"
        )
    if canonical_negatives.dim() != 2 or canonical_negatives.shape[0] < 1:
        raise ValueError(
            f"canonical_negatives 需要 [K>=1, Dclip]，但拿到 {tuple(canonical_negatives.shape)}"
        )
    if canonical_negatives.shape[1] != per_gaussian_clip.shape[1]:
        raise ValueError(
            f"canonical_negatives 的 Dclip={canonical_negatives.shape[1]} 與 "
            f"per_gaussian_clip 的 Dclip={per_gaussian_clip.shape[1]} 不符"
        )

    phi = _l2_normalize(per_gaussian_clip.float())          # [N, D]
    q = _l2_normalize(text_embed.float())                   # [D]
    neg = _l2_normalize(canonical_negatives.float())        # [K, D]

    pos_sim = phi @ q                                       # [N]
    neg_sim = phi @ neg.t()                                 # [N, K]
    pos_rep = pos_sim.unsqueeze(1).expand_as(neg_sim)       # [N, K]

    pair = torch.stack([pos_rep, neg_sim], dim=-1)          # [N, K, 2]
    prob_pos = torch.softmax(softmax_temp * pair, dim=-1)[..., 0]  # [N, K]
    # Hardest negative = the one giving the *lowest* positive probability.
    return prob_pos.min(dim=1).values                       # [N]


def relevancy_to_mask(
    scores: Tensor,
    threshold: float = 0.5,
    temperature: float = 10.0,
) -> Tensor:
    """Convert relevancy scores into a soft mask ``M ∈ [0, 1]``.

    ``M = sigmoid(temperature · (scores - threshold))``.

    Parameters
    ----------
    scores : Tensor
        Relevancy scores ``[N]`` (typically from :func:`compute_relevancy`).
    threshold : float
        Decision midpoint; ``M = 0.5`` exactly when ``score == threshold``.
        Higher threshold → smaller (more selective) mask.
    temperature : float
        Sharpness of the soft boundary. ``temperature → ∞`` approaches a hard
        ``score > threshold`` step; small values give a gentle ramp.

    Returns
    -------
    Tensor
        Soft mask ``M`` with the same shape as ``scores``, all entries in
        ``(0, 1)``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必須 > 0，但拿到 {temperature}")
    return torch.sigmoid(temperature * (scores - threshold))


def encode_text(
    prompt: Union[str, Sequence[str]],
    clip_encoder,
) -> Tensor:
    """Encode a text prompt via a ``CLIPEncoder`` (thin convenience wrapper).

    Parameters
    ----------
    prompt : str | Sequence[str]
        Query text, or a list of texts (e.g. the canonical negatives).
    clip_encoder : scene.clip_encoder.CLIPEncoder
        An instantiated encoder (kept as a parameter so this module never
        imports ``open_clip`` itself).

    Returns
    -------
    Tensor
        ``[Dclip]`` for a single ``str`` prompt, or ``[T, Dclip]`` for a list.
        L2-normalized (matching ``CLIPEncoder.encode_text``).
    """
    single = isinstance(prompt, str)
    emb = clip_encoder.encode_text(prompt)  # [T, Dclip], normalized
    return emb[0] if single else emb


def encode_negatives(
    clip_encoder,
    negatives: Sequence[str] = CANONICAL_NEGATIVES,
) -> Tensor:
    """Encode the canonical negative phrases → ``[K, Dclip]`` (normalized)."""
    return clip_encoder.encode_text(list(negatives))
