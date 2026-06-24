# Import necessary libraries
import os
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel
from transformers import CLIPTokenizer, CLIPTextModel
from RETFound_MAE import models_vit
from RETFound_MAE.models_vit import vit_large_patch16, VisionTransformer
from functools import partial
from scipy.stats import wasserstein_distance


def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def save_checkpoint(config, epoch, global_step, model, optimizer):
    """
    Saves a checkpoint of the model and optimizer.

    Args:
        config (object): Configuration object containing paths and settings.
        epoch (int): Current training epoch.
        global_step (int): Current training step.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.

    Notes:
        - Saves model and optimizer `state_dict()` along with epoch and training steps.
        - Handles multi-GPU setups by saving `model.module.state_dict()` when necessary.
        - Retries saving up to 10 times to handle potential I/O issues.
    """
    now = datetime.now()
    current_time_str = now.strftime("%H:%M:%S")
    checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_{global_step}_{current_time_str}.pt')
    save_num = 0

    while save_num < 10:
        try:
            state_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict() if config.n_gpu > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state_dict, checkpoint_path)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
            break
        except:
            save_num += 1

    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 attempts.")


def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    """
    Loads a pre-trained Vision Transformer (ViT) model from a checkpoint.

    Args:
        chkpt_dir (str): Path to the model checkpoint.
        arch (str): Model architecture to use (default: 'vit_large_patch16').

    Returns:
        VisionTransformer: Loaded Vision Transformer model.
    """
    # Instantiate the model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=False
    )

    # Load model weights from the checkpoint
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)

    return model



class Model_Wrapper(torch.nn.Module):
    """
    A wrapper class for a multi-modal model integrating Vision and Text models.
    """
    def __init__(self, config):
        """
        Initializes the Model_Wrapper with a vision and text model.

        Args:
            config (object): Configuration object containing model paths and settings.
        """
        super(Model_Wrapper, self).__init__()

        self.config = config

        # Load the pre-trained Vision Transformer model
        self.chkpt_dir = self.config.vision_model_checkpoint
        self.vision_model = prepare_model(self.chkpt_dir, 'vit_large_patch16')
        self.vision_model.head = torch.nn.Identity()  # Remove classification head

        # Projection layer for visual embeddings
        self.visual_proj = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.vision_model_output_dim, self.config.proj_dim),
        )

        # Load the pre-trained Text model (GatorTron)
        self.text_model = AutoModel.from_pretrained('UFNLP/gatortronS')
        #self.text_model = AutoModel.from_pretrained("bert-base-uncased")

        # Projection layer for text embeddings
        self.text_proj = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.text_model_output_dim, self.config.proj_dim),
        )

        # Define the logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Define learnable threshold parameters for similarity calculations
        self.beta = nn.Parameter(torch.ones([])) * -1
        self.thres_image = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.thres_text = nn.Parameter(torch.Tensor([0.97]), requires_grad=True)
        
    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def freeze_model(self):
        """
        Freezes the parameters of the Vision and Text models to prevent updates.
        """
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward_visual(self, images):
        """
        Forward pass for image embeddings.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Visual embeddings.
        """
        visual_proj = self.visual_proj(self.vision_model(images))
        return visual_proj

    def forward_text(self, texts):
        """
        Forward pass for text embeddings.

        Args:
            texts (dict): Tokenized text inputs.

        Returns:
            torch.Tensor: Text embeddings.
        """
        out = self.text_model(**texts)  # last_hidden_state, pooler_output (maybe)
        text_cls = self._mean_pool(out.last_hidden_state, texts['attention_mask'])
        text_proj = self.text_proj(text_cls)
        return text_proj

    def get_text_embeddings(self, texts):
        """
        Returns text embeddings without projection.

        Args:
            texts (dict): Tokenized text inputs.

        Returns:
            torch.Tensor: Raw text embeddings.
        """
        return self.text_model(**texts)

    def get_image_embeddings(self, images):
        """
        Returns image embeddings without projection.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Raw image embeddings.
        """
        return self.vision_model(images)

    def forward(self, images, texts):
        """
        Forward pass for both image and text inputs.

        Args:
            images (torch.Tensor): Input images.
            texts (dict): Tokenized text inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Projected image and text features.
        """
        image_features = self.forward_visual(images)
        text_features = self.forward_text(texts)

        return image_features, text_features

