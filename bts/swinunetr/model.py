import os
from functools import partial
from typing import Any, Dict, Tuple, Union

import nrrd
import numpy as np
import pytorch_lightning as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from torch.optim.lr_scheduler import CosineAnnealingLR

from bts.common.miscutils import DotConfig


class SwinUNETRModel(pl.LightningModule):
    def __init__(
        self,
        img_size: Tuple[int, int, int],
        classes: DotConfig[str, int],
        output_dir: str,
        in_channels: int = 1,
        max_epochs: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.max_epochs = max_epochs
        self.classes = classes
        self.output_dir = output_dir

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

        self.val_model_inferer = partial(
            sliding_window_inference,
            roi_size=self.img_size,
            sw_batch_size=4,
            predictor=self,
            overlap=0.5,
        )

        self.test_model_inferer = partial(
            sliding_window_inference,
            roi_size=self.img_size,
            sw_batch_size=1,
            predictor=self,
            overlap=0.6,
        )

        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        self.post_sigmoid = Activations(sigmoid=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        pred = self(img)
        loss = self.loss_fn(pred, label)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch["img"], batch["label"]
        logits = self.val_model_inferer(img)

        pred = self.post_pred(self.post_sigmoid(logits))

        loss = self.loss_fn(pred, label)
        self.metric_fn(pred, label)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        dice, not_nans = self.metric_fn.aggregate()
        self.log(
            "val_dice_brain",
            dice[self.classes.BRAIN - 1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_dice_tumor",
            dice[self.classes.TUMOR - 1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.metric_fn.reset()

    def test_step(self, batch: Dict[str, Union[torch.Tensor, Any]], batch_idx):
        image = batch["img"]
        info = batch["info"]
        label_meta_dict = batch["label_meta_dict"]
        headers = nrrd.read_header(label_meta_dict["filename_or_obj"][0])

        logits = self.test_model_inferer(image)

        pred = self.post_pred(self.post_sigmoid(logits[0]))
        pred = pred.detach().cpu().numpy()

        seg = self.one_hot_to_discrete(pred)

        save_path = os.path.join(
            self.output_dir,
            f"{info['patient_name'][0]}_prediction.seg.nrrd",
        )

        nrrd.write(
            file=save_path,
            data=seg,
            header=headers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs),
            "interval": "epoch",
            "frequency": 100,
        }
        return [optimizer], [scheduler]

    def one_hot_to_discrete(
        self, target: Union[torch.Tensor, np.ndarray]
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
        target_discrete = np.ones((H, W, D), dtype=np.int8) * self.classes.GROUND

        target_discrete[target[0] == 1] = self.classes.BRAIN
        target_discrete[target[1] == 1] = self.classes.TUMOR

        return target_discrete
