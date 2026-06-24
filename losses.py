import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

def compute_similarity_label(image_features, text_features, thres_image, thres_text, device='cpu'):
    """
    Computes a binary similarity matrix for image and text features.

    Args:
        image_features (torch.Tensor): Normalized image feature matrix (N, D).
        text_features (torch.Tensor): Normalized text feature matrix (N, D).
        threshold (float): Threshold value for similarity.
        device (str): Device to move the final label tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A binary similarity label matrix of shape (N, N).
    """
    # Compute similarity matrices
    similarity_matrix_image = torch.mm(image_features, image_features.t())
    similarity_matrix_image = (similarity_matrix_image > thres_image).int()

    similarity_matrix_text = torch.mm(text_features, text_features.t())
    similarity_matrix_text = (similarity_matrix_text > thres_text).int()

    # Compute final label
    label = (similarity_matrix_image + similarity_matrix_text > 0).float().to(device)

    return label


def sigmoid_loss(logits_per_image, logits_per_text, label,
                 distance_per_image, distance_per_text, batch_size):
    """
    Computes the Sigmoid-Based Loss for image-text matching.

    Args:
    logits_per_image: Tensor, predicted logits for images.
    logits_per_text: Tensor, predicted logits for text.
    label: Tensor, binary labels (0 or 1).
    distance_per_image: Tensor, precomputed distance for images.
    distance_per_text: Tensor, precomputed distance for text.
    input_images: Tensor, batch of input images.
    input_texts: Tensor, batch of input texts.

    Returns:
    total_loss: Scalar, computed loss for both images and texts.
    """
    # Convert labels from {0,1} to {-1,1}
    label = torch.where(label == 0, -1, 1)

    # Compute sigmoid activations
    z_image = F.sigmoid(label * logits_per_image)
    z_text = F.sigmoid(label * logits_per_text)

    # Compute losses
    image_loss = torch.sum(z_image * distance_per_image) / batch_size
    text_loss = torch.sum(z_text * distance_per_text) / batch_size

    # Total loss (optional: sum or mean)
    total_loss = (image_loss + text_loss) / 2

    return total_loss


def weighted_bce_loss(logits_per_image, logits_per_text, labels,
                      distance_per_image, distance_per_text, batch_size, weights=None, scaling_factor=1.0):
    """
    Computes Weighted Binary Cross-Entropy (BCE) Loss with logits for image-text matching.

    Args:
    logits_per_image: Tensor (batch_size, num_classes), predicted logits for images.
    logits_per_text: Tensor (batch_size, num_classes), predicted logits for texts.
    labels: Tensor (batch_size, num_classes), binary labels (0 or 1).
    distance_per_image: Tensor (batch_size, num_classes), precomputed distances for images.
    distance_per_text: Tensor (batch_size, num_classes), precomputed distances for texts.
    batch_size: Int, number of samples in the batch.
    weights: Tensor (batch_size, num_classes), optional element-wise weights for BCE loss.
    scaling_factor: Float, scales the computed loss (default is 1.0, meaning no scaling).

    Returns:
    loss: Scalar tensor representing the computed weighted BCE loss.
    """

    # Compute BCE loss (without reduction)
    image_loss = F.binary_cross_entropy_with_logits(logits_per_image, labels, weight=distance_per_image,
                                                    reduction="none")
    text_loss = F.binary_cross_entropy_with_logits(logits_per_text, labels, weight=distance_per_text, reduction="none")

    # Apply distance-based weighting
    image_loss = torch.sum(image_loss) / batch_size
    text_loss = torch.sum(text_loss) / batch_size

    # Compute total loss and apply scaling factor
    loss = ((image_loss + text_loss) / 2) * scaling_factor

    return loss


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (Na, d), b: (Nb, d); assume already L2-normalized
    return a @ b.T

@dataclass
class Thresholds:
    p1: float       # Sit >
    p1p: float      # Sit > p′1 (filter for text-text positives)
    p2: float       # Sii >
    p3: float       # Stt >

@torch.no_grad()
def align_sims_k(
    Xf: torch.Tensor,                    # (N_img, d)
    Tf: torch.Tensor,                    # (N_txt, d) with N_txt = k*N_img
    k: Optional[int] = None,
    caption_groups: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns Sit, Sii, Stt all shaped (N_img, N_txt), aligned for k captions per image.
    If texts aren't contiguous groups of size k, pass explicit caption_groups.
    """
    device = Xf.device
    N_img = Xf.size(0)
    N_txt = Tf.size(0)
    if caption_groups is None:
        assert N_txt % N_img == 0, "Cannot infer k; provide caption_groups or ensure N_txt=k*N_img"
        k = N_txt // N_img
        caption_groups = [list(range(i*k, (i+1)*k)) for i in range(N_img)]
    else:
        k = len(caption_groups[0])
        assert len(caption_groups) == N_img
        assert all(len(g)==k for g in caption_groups)
        flat = sorted([j for g in caption_groups for j in g])
        assert flat == list(range(N_txt)), "caption_groups must partition 0..N_txt-1"

    # Sit
    Sit = cosine_sim(Xf, Tf)  # (N_img, N_txt)

    # Sii: (N_img, N_img) -> replicate over k columns per j
    base_ii = cosine_sim(Xf, Xf)  # (N_img, N_img)
    Sii = torch.empty((N_img, N_txt), device=device, dtype=base_ii.dtype)
    for j, group in enumerate(caption_groups):
        Sii[:, group] = base_ii[:, j:j+1]

    # Stt: (N_txt, N_txt) -> for each (i, j) average over i's k × j's k; expand to k cols
    base_tt = cosine_sim(Tf, Tf)  # (N_txt, N_txt)
    Stt = torch.empty((N_img, N_txt), device=device, dtype=base_tt.dtype)
    for i, gi in enumerate(caption_groups):
        mean_over_i = base_tt[gi, :].mean(dim=0)   # (N_txt,)
        for j, gj in enumerate(caption_groups):
            val = mean_over_i[gj].mean()          # scalar
            Stt[i, gj] = val
    return Sit, Sii, Stt

@torch.no_grad()
def build_M(Sit: torch.Tensor, Sii: torch.Tensor, Stt: torch.Tensor, thr: Thresholds) -> torch.Tensor:
    """
    Boolean → {-1,+1}. Shape (N_img, N_txt)
    """
    pos = (Sit > thr.p1) | (Sii > thr.p2) | ((Stt > thr.p3) & (Sit > thr.p1p))
    M = pos.to(Sit.dtype).mul_(2.0).sub_(1.0)  # {0,1}→{-1,+1}
    return M

@torch.no_grad()
def estimate_beta_grid(
    batches,  # iterable of tuples: (Xf, Tf, thr, align_kwargs_dict)
    tau: float,
    beta_grid = tuple([-5.0 + 0.25*i for i in range(21)]),  # [-5, 0]
) -> float:
    best_beta, best_loss = None, float("inf")
    for beta in beta_grid:
        tot = 0.0
        cnt = 0
        for (Xf, Tf, thr, align_kwargs) in batches:
            Sit, Sii, Stt = align_sims_k(Xf, Tf, **align_kwargs)
            M = build_M(Sit, Sii, Stt, thr)
            logits = M * (-Sit / tau + beta)
            loss = F.softplus(logits).mean()
            tot += float(loss); cnt += 1
        avg = tot / max(cnt, 1)
        if avg < best_loss:
            best_loss, best_beta = avg, beta
    return float(best_beta)
