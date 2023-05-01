import argparse
import warnings
from functools import partial
from typing import Tuple

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bts.common import miscutils
from bts.common.miscutils import DotConfig
from bts.data.dataset import get_train_dataset, get_val_dataset

warnings.filterwarnings("ignore")


class SwinUNETRModel(LightningModule):
    def __init__(
        self,
        img_size: Tuple[int, int, int],
        classes: DotConfig[str, int],
        in_channels: int = 1,
        max_epochs: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.max_epochs = max_epochs
        self.classes = classes

        # Exclude `GROUND`
        num_classes = len(classes) - 1

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            use_checkpoint=True,
        )
        self.loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)

        self.metric_fn = DiceMetric(
            include_background=True, reduction="mean_batch", get_not_nans=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        pred = self(img)
        loss = self.loss_fn(pred, label)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_inferer = partial(
            sliding_window_inference,
            roi_size=self.img_size,
            sw_batch_size=2,
            predictor=self,
            overlap=0.6,
        )

        post_pred = AsDiscrete(threshold=0.5, dtype="bool")
        post_sigmoid = Activations(sigmoid=True)

        img, label = batch["img"], batch["label"]
        logits = model_inferer(img)

        pred = post_pred(post_sigmoid(logits))

        loss = self.loss_fn(pred, label)
        self.metric_fn(pred, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        dice, not_nans = self.metric_fn.aggregate()

        self.log(
            "val_dice_brain",
            dice[self.classes.BRAIN - 1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_dice_tumor",
            dice[self.classes.TUMOR - 1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.metric_fn.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]


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


def main():
    args = get_args()

    seed_everything(42)

    # Read the hyperparameters from the yaml file
    hyperparameters = miscutils.load_hyperparameters(args.hyperparameters)

    # Get the ROI size, batch size and overlap size
    image_size = hyperparameters.IMAGE_SIZE

    # Get the data loaders for training, validation and test sets
    train_dataset = get_train_dataset(args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_dataset = get_val_dataset(args.data_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize the SwinUNETR model
    model = SwinUNETRModel(
        img_size=image_size,
        classes=hyperparameters.LABELS,
    )

    # Initialize the wandb logger
    wandb_logger = WandbLogger(project="bbm47980_bts")

    # Initialize the model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice_tumor",
        dirpath=args.output,
        filename="model-{epoch:02d}-{val_dice_tumor:.2f}",
        save_top_k=1,
        mode="max",
    )

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        logger=wandb_logger,
        max_epochs=hyperparameters.EPOCHS,
        deterministic=True,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=100,
        strategy="ddp",
        precision=16,
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
