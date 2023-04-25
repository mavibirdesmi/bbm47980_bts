import os
import random
from typing import Any, Dict, Generic, Optional, TypeVar, Union

import numpy as np
import torch
import yaml

from bts.common import logutils

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

    def to_dict(self) -> Dict[Any, Any]:
        dict_repr = vars(self).copy()
        for key, value in dict_repr.items():
            if isinstance(value, DotConfig):
                dict_repr[key] = value.to_dict()

        return dict_repr


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
        self.avg: Union[int, float, torch.Tensor] = np.where(
            self.count > 0, self.sum / self.count, self.sum
        )


def load_checkpoint(
    model: Union[torch.nn.Module, torch.nn.parallel.DataParallel], checkpoint_path: str
) -> Union[torch.nn.Module, torch.nn.parallel.DataParallel]:
    """Loads the model checkpoint and returns it.

    Args:
        model_path: Path to the model.

    Returns:
        Checkpoint loaded model.
    """
    if not checkpoint_path.endswith("model.pt"):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")

    state_dict = torch.load(checkpoint_path)

    if isinstance(model, torch.nn.parallel.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def save_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
):
    """Save model checkpoint in the given directory, with `model.pt` name.

    Args:
        model: Trained model.
        save_dir: Directory to save the model checkpoint.
        epoch: Epoch number. Stored to the model state dictionary.
            Defaults to `None`.
        best_acc: Best accuracy. Stored to the model state dictionary.
            Defaults to `None`.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, torch.nn.parallel.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    checkpoint_path = os.path.join(save_dir, "model.pt")
    logger.info(f"Saving checkpoint to {checkpoint_path}")

    torch.save(state_dict, checkpoint_path)


def seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    logger.info("Everything is seeded.")
