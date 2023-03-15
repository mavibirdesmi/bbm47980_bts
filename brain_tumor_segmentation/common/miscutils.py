import os
from typing import Generic, Optional, TypeVar, Union

import torch
import yaml

from brain_tumor_segmentation.common import logutils

logger = logutils.get_logger(__name__)

K = TypeVar("K", bound=str)
V = TypeVar("V")


class DotConfig(Generic[K, V]):
    """A simple configuration class with dot notation support."""

    def __init__(self, config: dict):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, DotConfig(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"DotConfig({vars(self)})"

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def __len__(self) -> int:
        return len(vars(self))


def load_hyperparameters(path: Optional[str] = None) -> DotConfig:
    """Reads and retrieves hyperparameters accessible using the dot notation.

    Args:
        path: Path to hyperparameter configuration file. The file in the path given
        should be a valid yaml file. If not specified script will look for a
        ``hyperparameters.yaml`` file in the same directory the script is located in.

    Returns:
        DotConfig object containing hyperparameters.
    """

    if not path:
        project_dir = logutils.get_project_dir()
        path = os.path.join(project_dir, "hyperparameters.yaml")

    with open(path) as fp:
        hyperparams = yaml.safe_load(fp)

    logger.info("Hyperparameters loaded: %s", hyperparams)
    return DotConfig(hyperparams)


class AverageMeter(object):
    """Computes and stores the average and current value on the fly."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: Union[int, float, torch.Tensor] = 0
        self.avg: Union[int, float, torch.Tensor] = 0
        self.sum: Union[int, float, torch.Tensor] = 0
        self.count: Union[int, float, torch.Tensor] = 0

    def update(self, val, n=1):
        self.val: Union[int, float, torch.Tensor] = val
        self.sum: Union[int, float, torch.Tensor] = self.sum + val * n
        self.count: Union[int, float, torch.Tensor] = self.count + n
        self.avg: Union[int, float, torch.Tensor] = self.sum / self.count
