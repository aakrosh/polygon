"""Model loading utilities for POLYGON VAE models."""

import logging
import torch

from polygon.vae.vae_model import VAE


def load_vae_model(model_path, device):
    """Load VAE model, detecting bow_weight from checkpoint.

    This handles models trained with different configurations by inspecting
    the checkpoint keys before creating the model. Specifically, it detects
    if the model was trained with Bag-of-Words (BoW) auxiliary loss enabled.

    Note: The delta parameter (for delta-VAE) does not create additional
    layers, so it cannot be detected from checkpoint keys. However, delta
    only affects training (KL divergence target), not inference behavior.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: torch.device or string ('cpu', 'cuda:0', etc.)

    Returns:
        VAE model in eval mode
    """
    # Load checkpoint to inspect keys
    checkpoint = torch.load(model_path, map_location='cpu')

    # Detect if model was trained with BoW (bow_fc layer present)
    has_bow = any(k.startswith('bow_fc.') for k in checkpoint.keys())

    # Create model with appropriate parameters
    model_params = {}
    if has_bow:
        # Use a non-zero bow_weight to create the bow_fc layer
        # Actual value doesn't matter for inference, just needs to be > 0
        model_params['bow_weight'] = 1.0
        logging.info("Detected BoW-enabled model (bow_fc layer present)")

    model = VAE(**model_params).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model
