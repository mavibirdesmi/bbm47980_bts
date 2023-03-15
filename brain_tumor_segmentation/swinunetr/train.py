import argparse
from functools import partial
from typing import Dict, Optional, Union

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from torch.utils.data import DataLoader

from brain_tumor_segmentation.common import logutils, miscutils
from brain_tumor_segmentation.common.miscutils import DotConfig
from brain_tumor_segmentation.data.dummydataset import get_dataloader
from brain_tumor_segmentation.swinunetr import model as smodel

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


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Trains the given model for one epoch based on the optimizer given. The training
    progress is displayed with a custom progress bar. At the end of the each batch the
    mean of the batch loss is displayed within the progress bar.

    Args:
        model: Model to train.
        loader: Data loader.
            The batch data should be a dictionary containing "image" and "label" keys.
        loss_function: Loss function to measure the loss during the training.
        optimizer: Optimizer to optimize the loss.
        epoch: Epoch number. Only used in the progress bar to display the current epoch.
        device: Device to load the model and data into. Defaults to None. If set to None
            will be set to ``cuda`` if it is available, else will be set to ``cpu``.

    Returns:
        A dictionary containing statistics about the model training process.
        Keys and values available in the dictionary are as follows:
            ``Mean Loss``: Mean validation loss value for the whole segmentation.
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

    model = model.to(device)
    model.train()

    train_loss = miscutils.AverageMeter()

    with logutils.etqdm(loader, epoch=epoch) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)

            optimizer.zero_grad()

            logits = model(image)
            loss: torch.Tensor = loss_function(logits, label)

            loss.backward()
            optimizer.step()

            loss = loss.item()

            train_loss.update(loss, image.size(0))

            metrics = {
                "Mean Train Loss": loss,
            }

            pbar.log_metrics(metrics)

    history = {
        "Mean Loss": train_loss.avg,
    }

    return history


def val_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    roi_size: int,
    sw_batch_size: int,
    overlap: int,
    labels: DotConfig[str, DotConfig[str, int]],
    epoch: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Evaluates the given model.

    Args:
        model: Model to evaluate.
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

    val_accuracy = miscutils.AverageMeter()
    val_loss = miscutils.AverageMeter()

    post_softmax = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True)

    with torch.no_grad(), logutils.etqdm(loader, epoch=epoch) as pbar:
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

            val_loss.update(loss, image.size(0))

            dice_metric.reset()
            dice_metric(y_pred=preds, y=batch_labels)

            accuracy = dice_metric.aggregate()
            val_accuracy.update(accuracy, image.size(0))

            num_gt_labels = len(labels)
            num_pred_labels = accuracy.numel()
            assert num_gt_labels == num_pred_labels, (
                f"Number of labels should match with the number of prediction labels. "
                f"Found num labels: '{num_gt_labels}' != "
                f"num prediction labels: '{num_pred_labels}' "
            )

            metrics = {
                "Mean Val Brain Acc.": accuracy[labels.BRAIN],
                "Mean Val Tumor Acc.": accuracy[labels.TUMOR],
                "Mean Val Loss": val_loss,
            }

            pbar.log_metrics(metrics)

    history = {
        "Mean Brain Acc.": val_accuracy.avg[labels.BRAIN],
        "Mean Tumor Acc.": val_accuracy.avg[labels.TUMOR],
        "Mean Loss": val_loss.avg,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.LEARNING_RATE,
        weight_decay=hyperparams.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparams.EPOCHS
    )

    train_loader = get_dataloader()
    val_loader = get_dataloader()

    for epoch in range(1, hyperparams.EPOCHS + 1):
        logger.info(f"Epoch {epoch} is starting.")

        train_history = train_epoch(
            model,
            loader=train_loader,
            loss_function=dice_loss,
            optimizer=optimizer,
            epoch=epoch,
            device=hyperparams.DEVICE,
        )

        logger.info(f'Mean Train Loss: {train_history["Mean Loss"]}')

        if epoch % 100 == 0:
            val_history = val_epoch(
                model,
                loader=val_loader,
                loss_function=dice_loss,
                roi_size=hyperparams.ROI,
                sw_batch_size=hyperparams.SW_BATCH_SIZE,
                overlap=hyperparams.INFER_OVERLAP,
                labels=hyperparams.LABELS,
                epoch=epoch,
                device=hyperparams.DEVICE,
            )

            val_loss = val_history["Mean Loss"]
            val_brain_acc = val_history["Mean Brain Acc."]
            val_tumor_acc = val_history["Mean Tumor Acc."]

            logger.info(
                f"Mean Val Loss: {val_loss} "
                f"Mean Val Brain Acc.: {val_brain_acc} "
                f"Mean Val Tumor Acc.: {val_tumor_acc} "
            )

        logger.info(f"Epoch {epoch} finished.\n")

        scheduler.step()


if __name__ == "__main__":
    main()
