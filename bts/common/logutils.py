import logging
import os
import sys
from typing import Dict, Optional, Union

import torch
from tqdm.auto import tqdm


def get_project_dir() -> str:
    """Returns the project directory from the main executable.

    Examples:
        $ echo $HOME
            /home/my-user
        $ python path1/path2/main.py
            /home/my-user/path1/path2

    Returns:
        Project directory of the script which the python is executed.
    """
    script_path = sys.argv[0]
    script_dir = os.path.dirname(script_path)
    return script_dir


def get_logger(logger_name: str) -> logging.Logger:
    """Returns a logger with the info level.

    Args:
        logger_name: Name of the logger.
    """
    project_dir = get_project_dir()
    project_dir = project_dir.replace("/", ":")

    logger = logging.getLogger(logger_name)
    # If propogate is True, each message will also be propagated to root handler
    # causing a message to appear more than once.
    logger.propagate = False
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"[{project_dir}:%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class etqdm(tqdm):
    """A simple wrapper around the tqdm."""

    def __init__(self, *args, epoch: Optional[int] = None, **kwargs):
        """Returns a progress bar with description set to Epoch.

        Args:
            epoch: Epoch number. Defaults to None.

        Raises:
            AssertionError: If the epoch parameter is not an integer.
        """
        if epoch:
            assert isinstance(epoch, int), "Epoch must be an integer value!"
        desc = None
        self.disable = None
        if epoch:
            desc = f"Epoch {epoch}"
        super().__init__(*args, **kwargs, desc=desc)

    def log_metrics(self, metrics: Dict[str, Union[int, float, torch.Tensor]]):
        """Log metrics on the progress bar.

        Args:
            metrics: Dictionary of the metrics.
        """
        self.set_postfix(
            metrics,
        )
