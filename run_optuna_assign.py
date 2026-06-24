"""
python run_optuna.py --study-name retfound_study --n-trials 10
optuna-dashboard --storage sqlite:///retfound_study.db --host 0.0.0.0 --port 5555
python run_optuna.py --study-name retfound_study --n-trials 30 --launch-dashboard --dashboard-port 5555
"""

import gc, os
import math
from copy import deepcopy
import errno
import logging
from datetime import datetime
from omegaconf import OmegaConf
import tempfile

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Subset
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from logger import setup_logger

from dataset import Fundus_RiskFactor_Dataset
from model import Model_Wrapper
from utils import mkdir, get_cosine_schedule_with_warmup
import RETFound_MAE.models_vit as models_vit

import optuna
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def set_seed(seed=42):
    import os
    import random
    import numpy as np
    import torch

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- PATCH: utilities for Sit/Sii/Stt, M, and β ---

@dataclass
class Thresholds:
    p2: float   # Sii >
    p3: float   # Stt >

@torch.no_grad()
def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # assumes a,b row-normalized
    return a @ b.t()

@dataclass
class ThresholdsST:
    p2: float   # threshold for Sii
    p3: float   # threshold for Stt

@torch.no_grad()
def _align_sims_k(
    Xf: torch.Tensor,                # (N_img, d), L2-normalized
    Tf: torch.Tensor,                # (N_txt, d), L2-normalized, N_txt = k * N_img
    k: Optional[int] = None,
    caption_groups: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build ONLY Sii and Stt aligned to (N_img, N_txt).
    - Sii: replicate image-image sims across each caption group of image j
    - Stt: average k×k text-text sims between image i's captions and image j's captions
    Returns:
        Sii, Stt with shape (N_img, N_txt)
    """
    device = Xf.device
    N_img = Xf.size(0)
    N_txt = Tf.size(0)

    if caption_groups is None:
        assert N_txt % N_img == 0, "Cannot infer k; pass caption_groups or ensure N_txt = k * N_img"
        k = N_txt // N_img
        caption_groups = [list(range(i*k, (i+1)*k)) for i in range(N_img)]
    else:
        k = len(caption_groups[0])
        assert len(caption_groups) == N_img
        assert all(len(g) == k for g in caption_groups)
        flat = sorted([idx for g in caption_groups for idx in g])
        assert flat == list(range(N_txt)), "caption_groups must partition 0..N_txt-1"

    # base sims
    base_ii = Xf @ Xf.T           # (N_img, N_img) — Xf is already L2-normalized
    base_tt = Tf @ Tf.T           # (N_txt, N_txt)

    # Sii: replicate over each caption group j
    Sii = torch.empty((N_img, N_txt), device=device, dtype=base_ii.dtype)
    for j, group in enumerate(caption_groups):
        Sii[:, group] = base_ii[:, j:j+1]

    # Stt: average k×k for each (i,j), then broadcast across group j
    Stt = torch.empty((N_img, N_txt), device=device, dtype=base_tt.dtype)
    for i, gi in enumerate(caption_groups):
        mean_over_i = base_tt[gi, :].mean(dim=0)  # (N_txt,)
        for j, gj in enumerate(caption_groups):
            val = mean_over_i[gj].mean()
            Stt[i, gj] = val

    return Sii, Stt

@torch.no_grad()
def build_M(Sii: torch.Tensor, Stt: torch.Tensor, thr: ThresholdsST) -> torch.Tensor:
    """
    Build M using ONLY Sii and Stt:
        M = (Sii > p2) ∨ (Stt > p3)
    Output in {-1, +1}.
    """
    pos = (Sii > thr.p2) | (Stt > thr.p3)
    return pos.to(Sii.dtype).mul_(2.0).sub_(1.0)

class MultiPositiveBCELoss(torch.nn.Module):
    """
    ℓ = mean softplus( m_ij * ( -s_ij/τ + β ) ), where m_ij∈{-1,+1}, s_ij = Sit
    """
    def __init__(self, beta_init: float = -2.0, learn_beta: bool = True):
        super().__init__()
        if learn_beta:
            self.beta = torch.nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta_init)))

    def forward(self, Sit: torch.Tensor, M: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        logits = M * (-Sit / tau + self.beta)
        return F.softplus(logits).mean()

    @torch.no_grad()
    def set_beta(self, value: float):
        if isinstance(self.beta, torch.nn.Parameter):
            self.beta.data.fill_(float(value))
        else:
            self.beta.copy_(torch.tensor(float(value), device=self.beta.device))

@torch.no_grad()
def estimate_beta_grid(
    batches,                 # iterable of tuples: (Xf, Tf, thr_st, align_kwargs)
    tau_scalar: float,
    beta_grid = tuple([-5.0 + 0.25*i for i in range(21)]),  # [-5, 0]
) -> float:
    """
    Grid-search β for the Sii/Stt-only mask:
      - Sit is used only inside the loss (s_ij)
      - M is built from Sii and Stt only
    Each item in `batches` is:
      (Xf, Tf, thr_st: ThresholdsST, align_kwargs: {"k": k} or {"caption_groups": ...})
    """
    best_beta, best_loss = None, float("inf")
    tau = torch.tensor(float(tau_scalar))

    for beta in beta_grid:
        tot, cnt = 0.0, 0
        for (Xf, Tf, thr_st, align_kwargs) in batches:
            # s_ij for the loss
            Sit = Xf @ Tf.t()  # (N_img, N_txt) — Xf, Tf assumed L2-normalized already

            # build M from Sii/Stt only
            Sii, Stt = _align_sims_k(Xf, Tf, **align_kwargs)
            M = build_M(Sii, Stt, thr_st)

            logits = M * (-Sit / tau + beta)
            loss = F.softplus(logits).mean().item()
            tot += loss; cnt += 1
        avg = tot / max(cnt, 1)
        if avg < best_loss:
            best_loss, best_beta = avg, beta
    return float(best_beta)


set_seed(42)


config = OmegaConf.create({
    "saved_checkpoints" : './saved_checkpoints/use_GACL_true',
    "image_feature_file" : './data/UKB/Macular_Measurement.csv',
    "train_json" : "./data/UKB/captions_train.json",
    "val_json": "./data/UKB/captions_val.json",
    "test_json" : "./data/UKB/captions_test2.json",
    "logs" : 'logs',
    "logs_name": "training_logs_assign.txt",
    "use_distance": False,
    "use_assign": True,
    "vision_model_checkpoint":'./RETFound_MAE/RETFound_cfp_weights.pth', 
    "vision_model_output_dim": 1024,
    "text_model_output_dim": 1024, # change this into 1024
    "context_length": 512, 
    "proj_dim": 1024,
    "dropout": 0,
    "num_train_epochs": 10, 
    "batch_size": 64,
    "gradient_accumulation_steps" : 1,
    "loss": 'mutual',
    "optimizer": {
        "params":{
            'eps': 8.605285922082583e-07, 
            'lr': 0.00024276918679026895, 
            'weight_decay': 0.023262062576234904}
    },
    "n_gpu":1,
    "logging_steps" : 10,
    "save_steps" : 500,
    "val_steps":500
                          })

logger = setup_logger("optuna_finetuning", config.logs, 0, filename=config.logs_name)

# --- helpers: build loaders, evaluate, one-epoch train ---
def build_dataloaders(train_ds, val_ds, batch_size: int, subset_size: int = None, val_ratio: float = 0.1, num_workers: int = 0):
    """
    Creates train/val DataLoaders. If subset_size is provided, uses a random subset first (as in your script).
    """
    
    n_train = len(train_ds)
    n_val = len(val_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, n_train, n_val

import torch
import torch.nn.functional as F

def contrastive_info_nce_loss(image_embeds: torch.Tensor,
                              text_embeds:  torch.Tensor,
                              labels:       torch.Tensor,
                              temperature:  float = 0.07,
                              classification_logits_image: torch.Tensor = None,
                              classification_logits_text:  torch.Tensor = None,
                              multi_label:  bool  = False,
                              lambda_class: float = 0) -> torch.Tensor:
    """
    Combined InfoNCE‐style contrastive loss + optional classification loss.
    
    Args:
        image_embeds: (B, D) normalized image embeddings
        text_embeds:  (B, D) normalized text embeddings
        labels:       (B, B) binary label matrix (if multi_label) 
                      or class indices (if single label).
        temperature:  scaling factor for contrastive logits.
        classification_logits_image: (B, C) logits for image→class (optional)
        classification_logits_text:  (B, C) logits for text→class (optional)
        multi_label:  whether classification is multi‐label (True) or single label (False)
        lambda_class: weight for classification loss component.
    
    Returns:
        loss: scalar tensor
    """
    # 1) Contrastive part (InfoNCE)
    # Compute logits = similarity(image, text) / temperature
    logits_it = torch.matmul(image_embeds, text_embeds.t()) / temperature  # (B, B)
    logits_ti = torch.matmul(text_embeds, image_embeds.t()) / temperature  # (B, B)
    
    # Define target for contrastive: for single‐label / one‐to‐one scenario:
    #   ground_truth = torch.arange(B, device=device)
    # But for multi‐label, maybe labels is binary matrix.
    B = image_embeds.size(0)
    device = image_embeds.device
    
    if not multi_label:
        target = torch.arange(B, dtype=torch.long, device=device)
        contrastive_loss_i2t = F.cross_entropy(logits_it, target)
        contrastive_loss_t2i = F.cross_entropy(logits_ti, target)
        contrastive_loss = (contrastive_loss_i2t + contrastive_loss_t2i) / 2.0
    else:
        # Multi‐label: flatten and use BCE with logits
        # For efficiency, treat logits_it and labels as matching pairs matrix
        loss_i2t = F.binary_cross_entropy_with_logits(logits_it, labels)
        loss_t2i = F.binary_cross_entropy_with_logits(logits_ti, labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
    
    # 2) Classification part (optional)
    class_loss = torch.tensor(0.0, device=device)
    if classification_logits_image is not None and classification_logits_text is not None:
        if not multi_label:
            # single‐label classification: labels should be class indices
            class_loss_i = F.cross_entropy(classification_logits_image, labels)
            class_loss_t = F.cross_entropy(classification_logits_text,  labels)
            class_loss = (class_loss_i + class_loss_t) / 2.0
        else:
            # multi‐label: labels should be (B, C) binary matrix
            class_loss_i = F.binary_cross_entropy_with_logits(classification_logits_image, labels)
            class_loss_t = F.binary_cross_entropy_with_logits(classification_logits_text,  labels)
            class_loss = (class_loss_i + class_loss_t) / 2.0
    
    # 3) Combine
    total_loss = contrastive_loss + lambda_class * class_loss
    return total_loss


def evaluate(model, dataset, val_loader, config, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    use_amp = (device == "cuda")
    for batch in val_loader:
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            input_images, input_texts, image_features = batch
            input_images = input_images.to(device)
            input_texts = dataset.tokenizer.batch_encode_plus(  # unwrap Subset→Subset→Fundus dataset
                input_texts, return_tensors='pt', padding='max_length', max_length=config.context_length, truncation=True
            ).to(device)
            image_features = image_features.to(device)

            # forward (matches your train path)
            # text CLS features for distance/assign (you used it during training)
            text_features = model.get_text_embeddings(input_texts).last_hidden_state[:, 0, :].to(device)

            image_embeds, text_embeds = model(input_images, input_texts)
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            text_embeds  = text_embeds  / text_embeds.norm(dim=1, keepdim=True)

            text_features_norm  = text_features / text_features.norm(dim=1, keepdim=True)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.logit_scale.exp()
                thres_image = model.thres_image
                thres_text  = model.thres_text
            else:
                logit_scale = model.module.logit_scale.exp()
                thres_image = model.module.thres_image
                thres_text  = model.module.thres_text

            logits_per_image = logit_scale * image_embeds @ text_embeds.t()
            logits_per_text  = logit_scale * text_embeds  @ image_embeds.t()

            # in validation, image-text similarity from single subject is evaluated, because that's the goal. 
            label = torch.eye(len(image_embeds), device=device)
            loss = contrastive_info_nce_loss(image_embeds, 
                                             text_embeds,
                                             label,
                                             classification_logits_image=logits_per_image,
                                             classification_logits_text=logits_per_text,
                                             temperature = 0.07,
                                             multi_label=True)

            if config.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            n_batches  += 1

        model.train()
    return total_loss / max(1, n_batches)

def make_model_and_optimizer(config, device, thres_image_init: float, thres_text_init: float, lr: float, eps: float, weight_decay: float):
    """
    Builds a fresh model per trial, sets thresholds from Optuna (frozen by default), and returns optimizer/scheduler.
    """
    model = Model_Wrapper(config)
    model.freeze_model()
    unfreeze_vision = True
    unfreeze_text = False

    if unfreeze_vision:
        for child in model.vision_model.children():
            for p in child.parameters():
                p.requires_grad = True
    if unfreeze_text:
        for child in model.text_model.children():
            for p in child.parameters():
                p.requires_grad = True

    # thresholds as *hyperparameters* (freeze to let Optuna tune them)
    model.thres_image = nn.Parameter(torch.tensor([thres_image_init], dtype=torch.float32, device=device), requires_grad=False)
    model.thres_text  = nn.Parameter(torch.tensor([thres_text_init],  dtype=torch.float32, device=device), requires_grad=False)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)

    return model, optimizer

def train_one_epoch(model, dataset, train_loader, config, loss_mpbce, optimizer, scheduler, device, logger, global_state):
    
    use_amp = (device == "cuda")
    """
    One epoch over train_loader. Uses your exact loss path. Returns updated global_state.
    """
    for step, batch in enumerate(train_loader):
        with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
            input_images, input_texts, image_features = batch
            input_images = input_images.to(device)
            input_texts = dataset.tokenizer.batch_encode_plus(
                input_texts, return_tensors='pt', padding='max_length', max_length=config.context_length, truncation=True
            ).to(device)
            image_features = image_features.to(device)

            text_features = model.get_text_embeddings(input_texts).last_hidden_state[:, 0, :].to(device)

            image_embeds, text_embeds = model(input_images, input_texts)
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            text_embeds  = text_embeds  / text_embeds.norm(dim=1, keepdim=True)

            text_features_norm  = text_features / text_features.norm(dim=1, keepdim=True)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.logit_scale.exp()
                thres_image = model.thres_image
                thres_text  = model.thres_text
            else:
                logit_scale = model.module.logit_scale.exp()
                thres_image = model.module.thres_image
                thres_text  = model.module.thres_text

            logits_per_image = logit_scale * image_embeds @ text_embeds.t()
            logits_per_text  = logit_scale * text_embeds  @ image_embeds.t()

            # Compute loss based on the selected loss function
            if config.loss == 'mutual':
                # --- PATCH: compute Sit/Sii/Stt, M, and multi-positive BCE loss ---    
                Sit = image_embeds @ text_embeds.t()

                # CLIP-style temperature tau = exp(-logit_scale)
                if config.n_gpu == 1:
                    tau = torch.exp(-model.logit_scale)
                else:
                    tau = torch.exp(-model.module.logit_scale)

                # Use the normalized projected features from your forward()
                # image_embeds : (N_img, d), text_embeds : (N_txt, d)
                # If you do batch text augmentation, N_txt = k * N_img.
                N_img = image_embeds.size(0)
                N_txt = text_embeds.size(0)

                # If texts are contiguous per image, infer k; else pass caption_groups from dataloader.
                align_kwargs = {}
                if N_txt % N_img == 0:
                    align_kwargs["k"] = N_txt // N_img
                if hasattr(dataset, "caption_groups_in_batch") and dataset.caption_groups_in_batch is not None:
                    align_kwargs = {"caption_groups": dataset.caption_groups_in_batch(step)}

                # Build similarity matrices aligned to (N_img, N_txt)
                Sii, Stt = _align_sims_k(image_features_norm, text_features_norm, **align_kwargs)

                # Thresholds for M; start conservative and adjust if needed
                thr = Thresholds(
                    p2 = model.thres_image,   # Sii >
                    p3 = model.thres_text    # Stt >
                )

                # Assignment matrix M in {-1,+1}
                M = build_M(Sii, Stt, thr)

                # tau from your learnable logit_scale (CLIP-style)
                tau = torch.exp(-model.logit_scale) if config.n_gpu == 1 else torch.exp(-model.module.logit_scale)

                # Multi-positive BCE (Eq. 2)
                loss = loss_mpbce(Sit, M, tau=tau)

            if config.n_gpu > 1:
                loss = loss.mean()
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            global_state["global_loss"] += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_state["global_step"] += 1
                optimizer.step()

                # clamp logit_scale per CLIP
                if config.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                else:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

                if scheduler:
                    scheduler.step()
                model.zero_grad()

                if global_state["global_step"] % config.logging_steps == 0:
                    logger.info(
                        f"Epoch: {global_state['epoch']}  step: {global_state['global_step']}  "
                        f"lr: {optimizer.param_groups[0]['lr']:.6g}  "
                        f"loss: {loss.item():.4f}  ({global_state['global_loss']/max(1,global_state['global_step']):.4f})"
                    )
    return global_state

# --- Optuna objective ---
def objective(trial: optuna.Trial):
    try:
        # Suggest hyperparameters
        num_epochs   = trial.suggest_int("num_train_epochs", 1, 6)
        lr           = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
        eps          = trial.suggest_float("eps", 1e-9, 1e-6, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        thres_image0 = trial.suggest_float("thres_image", 0.2853, 0.9949)
        thres_text0  = trial.suggest_float("thres_text", 0.9548, 0.9979)
        beta = trial.suggest_float("beta", -5, 0)

        # Fresh config per trial (don’t mutate global)
        cfg = deepcopy(config)
        cfg.num_train_epochs = int(num_epochs)
        cfg.optimizer.params.lr = float(lr)
        cfg.optimizer.params.eps = float(eps)
        cfg.optimizer.params.weight_decay = float(weight_decay)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- PATCH: loss object (keep optimizer line as-is since it already includes model params) ---
        loss_mpbce = MultiPositiveBCELoss(beta_init=-2.0, learn_beta=True).to(device)

        # DataLoaders (uses your global `dataset` and `subset_size`)
        train_dataset = Fundus_RiskFactor_Dataset(cfg, 'train')
        val_dataset = Fundus_RiskFactor_Dataset(cfg, 'val')
        train_loader, val_loader, n_train, n_val = build_dataloaders(
            train_dataset, val_dataset, batch_size=cfg.batch_size, val_ratio=0.1, num_workers=0
        )

        # Scheduler warmup based on *this trial’s* total steps
        t_total = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_train_epochs

        # Build model/optimizer for this trial
        model, optimizer = make_model_and_optimizer(
            cfg, device, thres_image_init=thres_image0, thres_text_init=thres_text0,
            lr=cfg.optimizer.params.lr, eps=cfg.optimizer.params.eps, weight_decay=cfg.optimizer.params.weight_decay
        )
        num_warmup_steps = int(0.2 * t_total) if t_total > 0 else 0
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
        
        # --- PATCH: optional β estimation from a few dry batches (Original paper implemented this, but the beta values were found by optuna in our application.) ---
        with torch.no_grad():
            dry_batches = []
            for _ in range(3):
                input_images_, input_texts_list_, _ = next(iter(train_loader))
                input_images_ = input_images_.to(device)
                tokenized_texts_ = train_dataset.tokenizer.batch_encode_plus(
                    input_texts_list_, return_tensors='pt', padding='max_length',
                    max_length=config.context_length, truncation=True
                )
                tokenized_texts_ = {k: v.to(device) for k, v in tokenized_texts_.items()}

                # Forward through your model (embeds are L2-normalized in your forward)
                Xf_, Tf_ = model(input_images_, tokenized_texts_)

                # figure out k (contiguous grouping) or pass caption_groups if interleaved
                if Tf_.size(0) % Xf_.size(0) != 0:
                    raise RuntimeError("N_txt must be a multiple of N_img or pass caption_groups.")
                align_kwargs = {"k": Tf_.size(0) // Xf_.size(0)}

                # thresholds ONLY for Sii / Stt
                thr_st = ThresholdsST(p2=model.thres_image, p3=model.thres_text)

                dry_batches.append((Xf_, Tf_, thr_st, align_kwargs))

            tau0 = torch.exp(-model.logit_scale.detach()).item()
            best_beta = estimate_beta_grid(dry_batches, tau_scalar=tau0)
            loss_mpbce.set_beta(beta)
            logger.info(f"[beta-init] set β to {best_beta:.3f}")

        # Logging
        logger.info(f"[Trial {trial.number}] epochs={cfg.num_train_epochs} lr={cfg.optimizer.params.lr:.2e} "
                    f"eps={cfg.optimizer.params.eps:.1e} wd={cfg.optimizer.params.weight_decay:.2e} "
                    f"thres_image={thres_image0:.3f} thres_text={thres_text0:.3f} | steps={t_total} warmup={num_warmup_steps}")

        # Train with early stopping + pruning
        best_val = float("inf")
        best_state = None
        patience = 3
        no_improve = 0

        global_state = {"global_step": 0, "global_loss": 0.0, "epoch": 0}

        for epoch in range(cfg.num_train_epochs):
            global_state["epoch"] = epoch
            train_one_epoch(model, train_dataset, train_loader, cfg, loss_mpbce, optimizer, scheduler, device, logger, global_state)
            val_loss = evaluate(model, val_dataset, val_loader, cfg, device)

            # Optuna reporting/pruning
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping on val loss
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    "model": deepcopy(model.state_dict() if cfg.n_gpu == 1 else model.module.state_dict()),
                    "optimizer": deepcopy(optimizer.state_dict()),
                    "epoch": epoch
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"[Trial {trial.number}] Early stopping at epoch {epoch+1} (best val={best_val:.4f})")
                    break

        # (optional) Save best checkpoint per trial (comment out if not desired to save every trial)
        if best_state is not None:
            fn = os.path.join(cfg.saved_checkpoints, f"best_trial{trial.number}_val{best_val:.4f}.pt")
            torch.save(best_state, fn)

        return best_val
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"[Trial {trial.number}] OOM — pruning.")
            raise optuna.TrialPruned()
        raise
    finally:
        # hard cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

def run_optuna(n_trials: int = 25, study_name: str = "retfundus_tuning", direction: str = "minimize"):
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler, pruner=pruner)
    logger.info(f"Starting Optuna study: {study_name} | trials={n_trials}")
    best = study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best trial: {study.best_trial.number}  value (val loss)={study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    return study

study = run_optuna(n_trials=30)
print("Best params:", study.best_params)


# -----------------------------
# Optuna Dashboard/Jupyter Helper
# -----------------------------
if __name__ == "__main__":
    import argparse, subprocess, sys, shutil
    import os
    
    parser = argparse.ArgumentParser(description="Run Optuna with dashboard-friendly storage.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", type=str, default="optuna_study", help="Optuna study name.")
    parser.add_argument("--storage", type=str, default=None, help="RDB storage URL (e.g., sqlite:///optuna.db). If not set, uses sqlite:///<study-name>.db")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch optuna-dashboard in a subprocess.")
    parser.add_argument("--dashboard-port", type=int, default=5555, help="Port for optuna-dashboard.")
    parser.add_argument("--dashboard-host", type=str, default="127.0.0.1", help="Host for optuna-dashboard.")
    parser.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"], help="Optimization direction.")
    args, unknown = parser.parse_known_args()
    
    storage = args.storage or f"sqlite:///{args.study_name}.db"
    
    try:
        import optuna
        if 'run_optuna' in globals():
            try:
                study = run_optuna(n_trials=args.n_trials, study_name=args.study_name, direction=args.direction)
            except TypeError:
                print("[info] Custom run_optuna signature. Falling back to Study.optimize with RDB.")
                if 'objective' in globals():
                    study = optuna.create_study(
                        study_name=args.study_name,
                        direction=args.direction,
                        storage=storage,
                        load_if_exists=True
                    )
                    callbacks = []
                    try:
                        from optuna.integration import TQDMCallback
                        callbacks.append(TQDMCallback(leave=False))
                    except Exception:
                        pass
                    study.optimize(objective, n_trials=args.n_trials, callbacks=callbacks)
                else:
                    print("[error] No objective(trial) found.")
                    sys.exit(1)
        else:
            print("[warning] No 'run_optuna' found. Trying objective(trial).")
            if 'objective' in globals():
                study = optuna.create_study(
                    study_name=args.study_name,
                    direction=args.direction,
                    storage=storage,
                    load_if_exists=True
                )
                study.optimize(objective, n_trials=args.n_trials)
            else:
                print("[error] Could not find run_optuna() or objective(). Exiting.")
                sys.exit(1)
    except Exception as e:
        print(f"[fatal] Failed to start Optuna: {e}")
        sys.exit(1)
    
    print(f"\n[ok] Study '{args.study_name}' ready.")
    print(f"[ok] Storage: {storage}")
    
    if args.launch_dashboard:
        dash_cmd = shutil.which("optuna-dashboard")
        if dash_cmd is None:
            print("\n[warn] 'optuna-dashboard' not found. Install with: pip install optuna-dashboard")
        else:
            print(f"\n[ok] Launching optuna-dashboard on http://{args.dashboard_host}:{args.dashboard_port}")
            try:
                proc = subprocess.Popen(
                    [dash_cmd, "--host", args.dashboard_host, "--port", str(args.dashboard_port), "--storage", storage],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                # Print first lines so user sees the URL
                for _ in range(5):
                    line = proc.stdout.readline()
                    if not line:
                        break
                    print(line.strip())
                print("\n[tip] Keep this process running to keep the dashboard active. Press Ctrl+C to stop.")
                proc.wait()
            except KeyboardInterrupt:
                print("\n[info] Dashboard stopped by user.")
            except Exception as e:
                print(f"[error] Failed to launch dashboard: {e}")
    
    print("\n[How to view the dashboard]")
    print(f"  optuna-dashboard --storage {storage} --host 0.0.0.0 --port 5555")
    print("  # If remote, SSH tunnel: ssh -N -L 5555:127.0.0.1:5555 your_user@remote_host")
    print("  # Then open http://127.0.0.1:5555 in your browser.")