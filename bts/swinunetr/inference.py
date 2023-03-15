import argparse
from functools import partial
from typing import Dict, Union

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from torch.utils.data import DataLoader

from bts.common import logutils, miscutils
from bts.common.miscutils import DotConfig
from bts.data.dummydataset import get_dataloader
from bts.swinunetr import model as smodel

logger = logutils.get_logger(__name__)


def get_args() -> argparse.Namespace:
    """Retrieve arguments passed to the script.

    Returns:
        Namespace object where each argument can be accessed using the dot notation.
    """
    parser = argparse.ArgumentParser()
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
        "--output", type=str, required=True, help="Path to save the trained model."
    )

    return parser.parse_args()


def test(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    roi_size: int,
    sw_batch_size: int,
    overlap: int,
    labels: DotConfig[str, DotConfig[str, int]],
    device: torch.device = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Tests the given model.

    Args:
        model: Model to test.
        loader: Data loader.
            The batch data should be a dictionary containing "image" and "label" keys.
        loss_function: Loss function to measure the loss during the validation.
        roi_size: The spatial window size for inferences.
        sw_batch_size: The batch size to run window slices.
        overlap: Amount of overlap between scans.
        labels: Label key-values configured with DotConfig.
            labels should have `BRAIN` and `TUMOR` keys.
        epoch: Epoch number. Only used in the progress bar to display the current epoch.
        device: Device to load the model and data into. Defaults to None. If set to None
            will be set to ``cuda`` if it is available, else will be set to ``cpu``.

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

    model = model.to(device)
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    test_accuracy = miscutils.AverageMeter()
    test_loss = miscutils.AverageMeter()

    post_softmax = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True)

    with torch.no_grad(), logutils.etqdm(loader) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)

            logits = model_inferer(image)

            batch_labels = decollate_batch(label)
            outputs = decollate_batch(logits)
            preds = torch.stack(
                [
                    post_pred(post_softmax(val_pred_tensor))
                    for val_pred_tensor in outputs
                ]
            )

            loss: torch.Tensor = loss_function(logits, preds)
            loss = loss.item()

            test_loss.update(loss, image.size(0))

            dice_metric.reset()
            dice_metric(y_pred=preds, y=batch_labels)

            accuracy = dice_metric.aggregate()
            test_loss.update(accuracy, image.size(0))

            num_gt_labels = len(labels)
            num_pred_labels = accuracy.numel()
            assert num_gt_labels == num_pred_labels, (
                f"Number of labels should match with the number of prediction labels. "
                f"Found num labels: '{num_gt_labels}' != "
                f"num prediction labels: '{num_pred_labels}' "
            )

            metrics = {
                "Mean Test Brain Acc.": accuracy[labels.BRAIN],
                "Mean Test Tumor Acc.": accuracy[labels.TUMOR],
                "Mean Test Loss": test_loss,
            }

            pbar.log_metrics(metrics)

    history = {
        "Mean Brain Acc.": test_accuracy.avg[labels.BRAIN],
        "Mean Tumor Acc.": test_accuracy.avg[labels.TUMOR],
        "Mean Loss": test_accuracy.avg,
    }

    return history


def main():
    args = get_args()

    hyperparams = miscutils.load_hyperparameters(args.hyperparameters)

    smodel.set_cudnn_benchmark()

    model = smodel.get_model(
        img_size=hyperparams.ROI,
        in_channels=hyperparams.IN_CHANNELS,
        out_channels=hyperparams.OUT_CHANNELS,
        feature_size=hyperparams.FEATURE_SIZE,
        use_checkpoint=hyperparams.GRADIENT_CHECKPOINT,
    )

    model = torch.nn.DataParallel(model)

    # dice_loss = DiceLoss(to_onehot_y=False, softmax=True)
    dice_loss = DiceLoss(include_background=True, to_onehot_y=False, softmax=True)

    test_loader = get_dataloader()

    logger.info("Starting test")

    history = test(
        model,
        loader=test_loader,
        loss_function=dice_loss,
        roi_size=hyperparams.ROI,
        sw_batch_size=hyperparams.SW_BATCH_SIZE,
        overlap=hyperparams.INFER_OVERLAP,
        labels=hyperparams.LABELS,
        device=hyperparams.DEVICE,
    )

    loss = history["Mean Loss"]
    brain_acc = history["Mean Brain Acc."]
    tumor_acc = history["Mean Tumor Acc."]

    logger.info(
        f"Mean Val Loss: {loss} "
        f"Mean Val Brain Acc.: {brain_acc} "
        f"Mean Val Tumor Acc.: {tumor_acc} "
    )


if __name__ == "__main__":
    main()
