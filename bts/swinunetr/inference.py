import argparse
import os
from functools import partial
from os.path import join
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from torch.utils.data import DataLoader

from bts.common import logutils, miscutils
from bts.common.miscutils import DotConfig
from bts.data.dataset import get_test_dataset
from bts.data.utils import save_prediction_as_nrrd
from bts.swinunetr import model as smodel

logger = logutils.get_logger(__name__)


def get_args() -> argparse.Namespace:
    """Retrieve arguments passed to the script.

    Returns:
        Namespace object where each argument can be accessed using the dot notation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model.",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        help=(
            "Training hyperparameters file path. The file in the path given should be "
            "a valid yaml file. If not specified script will look for a "
            "``hyperparameters.yaml`` file in the same directory the script is located "
            "in."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help=(
            "Path to save the the predictions. If not set, will create a directory "
            "named `predictions, and save the prediction into it."
        ),
    )

    return parser.parse_args()


def test(
    model: torch.nn.Module,
    loader: DataLoader,
    roi_size: int,
    sw_batch_size: int,
    overlap: int,
    labels: DotConfig[str, int],
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Tests the given model.

    Args:
        model: Model to test.
        loader: Data loader.
            The batch data should be a dictionary containing "image" and "label" keys.
        roi_size: The spatial window size for inferences.
        sw_batch_size: The batch size to run window slices.
        overlap: Amount of overlap between scans.
        labels: Label key-values configured with DotConfig.
            labels should have `BRAIN` and `TUMOR` keys.
        epoch: Epoch number. Only used in the progress bar to display the current epoch.
        device: Device to load the model and data into. Defaults to None. If set to None
            will be set to ``cuda`` if it is available, else will be set to ``cpu``.
        output_dir: Path to save the predictions into. If not set, will create and use
            the ``predictions`` in the working directory.

    Raises:
        AssertionError: If labels does not have either of `BRAIN` and `TUMOR` keys.

    Returns:
        A dictionary containing statistics about the model validation process.
        Keys and values available in the dictionary are as follows:
            ``Mean Brain Acc.``: Mean accuracy value for the brain segmentation
            ``Mean Tumor Acc.``: Mean accuracy value for the tumor segmentation
            ``Mean Loss``: Mean validation loss value for the whole segmentation.
    """
    for expected_label in ["BRAIN", "TUMOR"]:
        assert expected_label in labels, f"labels should have a {expected_label} key!"

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

    if not output_dir:
        output_dir = "predictions"

    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )

    post_softmax = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=0.5, dtype="int8")

    with torch.no_grad(), logutils.etqdm(loader) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["img"].to(device)
            info: Dict[str, Any] = batch_data["info"]
            meta_dict: Dict[str, Any] = batch_data["img_meta_dict"]

            logits = model_inferer(image)

            seg = post_pred(post_softmax(logits[0])).detach().cpu().numpy()
            seg = one_hot_to_discrete(seg, labels)

            save_prediction_as_nrrd(
                seg,
                0,
                join(
                    output_dir,
                    f"{info['patient_name'][0]}_prediction.nrrd",
                ),
                meta_dict=meta_dict,
            )


def one_hot_to_discrete(
    target: Union[torch.Tensor, np.ndarray], labels: DotConfig[str, str]
) -> np.ndarray:
    """Transform tensor in the one hot form to the discrete form.

    Args:
        target: Tensor to transform. Expected shape is (L, H, W, D) where L is the
            number of labels.
        labels: Labels to use in transformation

    Returns:
        Discrete transformed array, with `int8` dtype.
    """
    L, H, W, D = target.shape
    target_discrete = np.ones((H, W, D), dtype=np.int8) * labels.GROUND

    target_discrete[target[0] == 1] = labels.BRAIN
    target_discrete[target[1] == 1] = labels.TUMOR

    return target_discrete


def main():
    args = get_args()

    hyperparams = miscutils.load_hyperparameters(args.hyperparameters)

    hyperparams.BATCH_SIZE = 1

    assert hyperparams.BATCH_SIZE == 1, "Inference only works with batch size of one!"

    smodel.set_cudnn_benchmark()

    model = smodel.get_model(
        img_size=hyperparams.ROI,
        in_channels=hyperparams.IN_CHANNELS,
        out_channels=hyperparams.OUT_CHANNELS,
        feature_size=hyperparams.FEATURE_SIZE,
        use_checkpoint=hyperparams.GRADIENT_CHECKPOINT,
    )

    model = torch.nn.DataParallel(model)

    model = miscutils.load_checkpoint(model, args.model)

    dataset = get_test_dataset(args.data_dir)

    loader = DataLoader(dataset=dataset, batch_size=1)

    logger.info("Starting test.")

    test(
        model,
        loader=loader,
        roi_size=hyperparams.ROI,
        sw_batch_size=hyperparams.SW_BATCH_SIZE,
        overlap=hyperparams.INFER_OVERLAP,
        labels=hyperparams.LABELS,
        device=hyperparams.DEVICE,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
