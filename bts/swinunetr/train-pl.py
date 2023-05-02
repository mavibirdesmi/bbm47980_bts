import os
import warnings

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from bts.common import miscutils
from bts.data.dataset import get_test_dataset, get_train_dataset, get_val_dataset
from bts.swinunetr.common import get_args
from bts.swinunetr.model import SwinUNETRModel

warnings.filterwarnings("ignore")


if __name__ == "__main__":
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
        batch_size=hyperparameters.BATCH_SIZE,
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

    test_dataset = get_test_dataset(args.data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize the SwinUNETR model
    model = SwinUNETRModel(
        img_size=image_size,
        classes=hyperparameters.LABELS,
        output_dir="predictions",
        max_epochs=hyperparameters.EPOCHS,
    )

    # Initialize the wandb logger
    wandb_logger = WandbLogger(project="bbm47980_bts")

    # Initialize the model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice_tumor",
        dirpath=args.output,
        filename="best-checkpoint",
        save_top_k=1,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        logger=wandb_logger,
        max_epochs=hyperparameters.EPOCHS,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor],
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

    trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=os.path.join(args.output, "best-checkpoint.ckpt"),
    )
