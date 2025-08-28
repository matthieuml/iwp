import logging
import os
import random
import shutil

import numpy as np
import torch

logger = logging.getLogger("iwp")


def make_dirs(
    experiment_path: str,
) -> tuple[str, str, str]:
    """Create directories for the experiment.

    Args:
        experiment_path (str): path to the experiment

    Returns:
        tuple: A tuple containing the experiment path and the visuals path.
    """
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        parent, folder_name = os.path.split(experiment_path)
        counter = 1
        while True:
            new_folder_name = f"{folder_name}_{counter}"
            experiment_path = os.path.join(parent, new_folder_name)
            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)
                break
            counter += 1

    visuals_path = os.path.join(experiment_path, "visuals")
    os.makedirs(visuals_path, exist_ok=True)
    logger.debug(f"Created directories for the experiment at {visuals_path}")

    return experiment_path, visuals_path


def copy_file(source: str, destination: str) -> None:
    """
    Copies a file from the source path to the destination path.

    Args:
        source (str): Path to the source file.
        destination (str): Path to the destination file or directory.

    Returns:
        None
    """
    try:
        shutil.copy(source, destination)
        logger.debug(f"Copied file from {source} to {destination}")
    except Exception as e:
        logger.error(f"Failed to copy file from {source} to {destination}")


def set_seed(seed: int) -> None:
    """
    Set the seed for Python's random module, numpy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Python's built-in random module
    random.seed(seed)
    logger.debug(f"Set seed for random module to {seed}")

    # Numpy
    np.random.seed(seed)
    logger.debug(f"Set seed for numpy to {seed}")

    # PyTorch
    torch.manual_seed(seed)
    logger.debug(f"Set seed for PyTorch to {seed}")
    torch.cuda.manual_seed_all(seed)  # Also covers multi-GPU environments
    logger.debug(f"Set seed for all CUDA GPUs to {seed}")

    # Ensure deterministic behavior in PyTorch (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Set PyTorch backend to deterministic mode")

    logger.info(f"Set seed for all random number generators to seed {seed}")
