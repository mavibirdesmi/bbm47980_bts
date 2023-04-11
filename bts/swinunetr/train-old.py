import os
import tempfile
from functools import partial
from typing import Dict, Optional, Union

import torch
from monai import data
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from torch.utils.data import DataLoader

import wandb
from bts.common import logutils, miscutils
from bts.common.miscutils import DotConfig
from bts.data.dataset import get_train_dataset

logger = logutils.get_logger(__name__)


wandb.init(name="Old code")


batch_size = 2
shuffle = True

roi = (128, 128, 128)


# create transformations
dataset = get_train_dataset("/home/vedatb/senior-project/data/btsed_dataset")

dataloader = data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


directory = os.environ.get("MONAI_DATA_DIRECTORY")
directory = "/home/vedatb/senior-project/bbm47980_bts/old-model-checkpoints"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


device = "cuda"
device = torch.device(device)


model = SwinUNETR(
    img_size=roi,
    in_channels=1,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
)
model = torch.nn.DataParallel(model)
model = model.to(device)


batch_size = 2
sw_batch_size = 2
infer_overlap = 0.6
max_epochs = 5000
val_every = 250


torch.backends.cudnn.benchmark = True


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


model_inferer = partial(
    sliding_window_inference,
    roi_size=roi,
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)


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
    model = model.train()

    train_loss = miscutils.AverageMeter()

    with logutils.etqdm(loader, epoch=epoch) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["img"].to(device)
            label = batch_data["label"].to(device)

            optimizer.zero_grad()

            logits = model(image)
            loss: torch.Tensor = loss_function(logits, label)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            train_loss.update(loss_val, image.size(0))

            metrics = {
                "Mean Loss": loss_val,
            }

            pbar.log_metrics(metrics)

    history = {
        "Mean Train Loss": train_loss.avg.item(),
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

    dice_metric = DiceMetric(
        include_background=True, reduction="mean_batch", get_not_nans=True
    )

    val_accuracy = miscutils.AverageMeter()
    val_loss = miscutils.AverageMeter()

    post_pred = AsDiscrete(threshold=0.5, dtype="bool")
    post_sigmoid = Activations(sigmoid=True)

    with torch.no_grad(), logutils.etqdm(loader, epoch=epoch) as pbar:
        for batch_data in pbar:
            batch_data: Dict[str, torch.Tensor]
            image = batch_data["img"].to(device)
            label = batch_data["label"].to(device)

            logits = model_inferer(image)

            preds = post_pred(post_sigmoid(logits))

            loss: torch.Tensor = loss_function(logits, preds)

            loss_val = loss.item()
            val_loss.update(loss_val, image.size(0))

            dice_metric.reset()
            dice_metric(y=label, y_pred=preds)

            accuracy, not_nans = dice_metric.aggregate()

            val_accuracy.update(accuracy.cpu().numpy(), n=not_nans.cpu().numpy())

            # `GROUND` label is excluded
            metrics = {
                "Mean Brain Acc": accuracy[labels.BRAIN - 1].item(),
                "Mean Tumor Acc": accuracy[labels.TUMOR - 1].item(),
                "Mean Loss": loss_val,
            }

            pbar.log_metrics(metrics)

    # `GROUND` label is excluded
    history = {
        "Mean Val Brain Acc": val_accuracy.avg[labels.BRAIN - 1],
        "Mean Val Tumor Acc": val_accuracy.avg[labels.TUMOR - 1],
        "Mean Val Acc": val_accuracy.avg.mean(),
        "Mean Val Loss": val_loss.avg.item(),
    }

    return history


hyperparams = miscutils.load_hyperparameters(
    "/home/vedatb/senior-project/bbm47980_bts/bts/swinunetr/hyperparameters.yaml"
)


start_epoch = 0


def main():
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    for epoch in range(start_epoch, max_epochs):
        train_history = train_epoch(
            model=model,
            loader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            loss_function=dice_loss,
        )

        logger.info(f'Mean Train Loss: {round(train_history["Mean Train Loss"], 2)}')
        wandb.log(train_history, step=epoch)

        if (epoch + 1) % val_every == 0 or epoch == 0:
            val_history = val_epoch(
                model,
                loader=dataloader,
                loss_function=dice_loss,
                roi_size=hyperparams.ROI,
                sw_batch_size=hyperparams.SW_BATCH_SIZE,
                overlap=hyperparams.INFER_OVERLAP,
                labels=hyperparams.LABELS,
                epoch=epoch,
                device=hyperparams.DEVICE,
            )

            # val_loss = val_history["Mean Val Loss"]
            # val_brain_acc = val_history["Mean Val Brain Acc"]
            # val_tumor_acc = val_history["Mean Val Tumor Acc"]
            # val_mean_acc = val_history["Mean Val Tumor Acc"]

            wandb.log(val_history, step=epoch)

            wandb.log({"Learning Rate": scheduler.get_lr()[0]}, step=epoch)
            scheduler.step()


if __name__ == "__main__":
    main()
