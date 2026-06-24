import os
import math
import torch
import errno
import logging
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

def mkdir(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): The directory path to create.

    Notes:
        - If `path` is an empty string, the function does nothing (avoids creating `./`).
        - Uses `os.makedirs()` to create directories, handling cases where the directory
          already exists.
        - Raises an error for any other type of OSError.
    """
    # If the path is empty (i.e., current directory), do nothing.
    if path == '':
        return
    try:
        os.makedirs(path)  # Creates the directory and any missing parent directories
    except OSError as e:
        if e.errno != errno.EEXIST:  # Ignore "directory already exists" errors
            raise  # Raise other OS errors


def convert_models_to_fp32(model):
    """
    Converts all parameters and gradients of a PyTorch model to 32-bit floating point.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.

    Notes:
        - This function explicitly sets the `.data` attribute of model parameters and
          gradients to `float32` to ensure consistency in precision.
        - Useful when models are trained in mixed-precision but need to be evaluated
          in full 32-bit precision.
    """
    for p in model.parameters():
        p.data = p.data.float()  # Convert model parameters to float32
        if p.grad:
            p.grad.data = p.grad.data.float()  # Convert gradients to float32


def torch_version_str_compare_lessequal(version1, version2):
    """
    Compares two PyTorch version strings to check if `version1` is less than or equal to `version2`.

    Args:
        version1 (str): The first PyTorch version string (e.g., "1.8.1+cu102").
        version2 (str): The second PyTorch version string (e.g., "1.9.0").

    Returns:
        bool: True if `version1` <= `version2`, False otherwise.

    Notes:
        - PyTorch version strings can contain additional information after a `+` (e.g., CUDA version).
        - This function extracts the main version (`X.Y.Z` format) before comparison.
        - Ensures that both versions follow a standard three-part format (`1.X.Y`).
    """
    # Extract the major.minor.patch version numbers, ignoring any build metadata (e.g., "+cu102")
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]

    # Ensure both versions are in the expected format (three-part versioning)
    assert len(v1) == 3, f"Cannot parse the version of your installed PyTorch! ({version1})"
    assert len(v2) == 3, f"Illegal version specification ({version2}). Should be in 1.X.Y format."

    # Sort versions and check if v1 is the lesser or equal version
    return sorted([v1, v2])[0] == v1        


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Creates a cosine learning rate schedule with warmup.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)