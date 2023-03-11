from typing import Tuple

import torch
from monai.networks.nets import SwinUNETR

from common import logutils

logger = logutils.get_logger(__name__)


def get_model(
    img_size: Tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    feature_size: int = 24,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """Retrieves a SwinUNETR model configured with the given parameters.

    Args:
        img_size: Dimension of the input image.
        in_channels: Dimension of the input channels.
        out_channels: Dimension of the output channels.
        feature_size: Dimension of network feature size.
        use_checkpoint: Use gradient checkkpointing for reduced memory usage
            with the cost of a small increase in the computation time.

    Returns:
        SwinUNETR model configured with the given parameters.
    """
    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
    logger.info("SwinUNETR model created.")
    return model


def set_cudnn_benchmark():
    """Causes cuDNN to benchmark multiple convolution algorithms and select the
    fastest."""
    logger.info("Enabling cuDNN benchmark")
    torch.backends.cudnn.benchmark = True
