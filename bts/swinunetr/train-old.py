#!/usr/bin/env python
# coding: utf-8

# Inspired from https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

# In[1]:


import time

import matplotlib
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from bts.common import logutils, miscutils
from bts.common.miscutils import DotConfig
from bts.data.dataset import get_test_dataset, get_train_dataset

# In[2]:


logger = logutils.get_logger(__name__)


wandb.init(name="Old code -r")


# In[3]:


import json
import os
from typing import Dict, Optional, Union

import torch
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, Transform
from monai.utils.enums import TransformBackends

# In[4]:


root_path = "../../data"


# In[5]:


batch_size = 2
shuffle = True

roi = (128, 128, 128)


# In[6]:


from monai import data, transforms
from monai.data.utils import collate_meta_tensor, list_data_collate

"""

    
"""

# create transformations
dataset = get_train_dataset("/home/vedatb/senior-project/data/btsed_dataset")

dataloader = data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    # collate_fn=collate_fn,
    pin_memory=True,
)


# ### Check data shape and visualize from data loader


# ### Check data shape and visualize from nrrd filess


# In[10]:

import os
import tempfile
from functools import partial

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

# ### Setup data directory

# In[11]:


directory = os.environ.get("MONAI_DATA_DIRECTORY")
directory = "/home/vedatb/senior-project/bbm47980_bts/old-model-checkpoints"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# ### Setup average meter, fold reader, checkpoint save

# In[12]:


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


# In[13]:


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


# ### Setup device

# In[14]:


device = "cuda"
device = torch.device(device)


# ### SwinUNETR Model

# In[15]:


model = SwinUNETR(
    img_size=roi,
    in_channels=1,  # t1 images,
    out_channels=2,  # brain and tumor
    feature_size=48,
    use_checkpoint=True,
)
model = torch.nn.DataParallel(model)
model = model.to(device)


# ### Optimizer and Loss Function

# In[16]:


batch_size = 2
sw_batch_size = 2
infer_overlap = 0.6
max_epochs = 5000
val_every = 250


# In[17]:


# Causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
torch.backends.cudnn.benchmark = True


# In[18]:


dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(
    include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


# In[19]:


model_inferer = partial(
    sliding_window_inference,
    roi_size=roi,
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)


# ### Define Train and Validation Epoch

# In[20]:


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


# In[21]:


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
# In[22]:


def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_brain = []
    dices_tumor = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []

    for epoch in range(start_epoch, max_epochs):
        train_history = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_function=loss_func,
        )

        logger.info(f'Mean Train Loss: {round(train_history["Mean Train Loss"], 2)}')
        wandb.log(train_history, step=epoch)

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_history["Mean Train Loss"])
            trains_epoch.append(int(epoch))

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

            val_loss = val_history["Mean Val Loss"]
            val_brain_acc = val_history["Mean Val Brain Acc"]
            val_tumor_acc = val_history["Mean Val Tumor Acc"]
            val_mean_acc = val_history["Mean Val Tumor Acc"]

            wandb.log(val_history, step=epoch)

            dices_brain.append(val_brain_acc)
            dices_tumor.append(val_tumor_acc)
            dices_avg.append(val_mean_acc)
            if val_tumor_acc > val_acc_max:
                print(
                    "new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_tumor_acc)
                )
                val_acc_max = val_tumor_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )

            wandb.log({"Learning Rate": scheduler.get_lr()[0]}, step=epoch)
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_brain,
        dices_tumor,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


# In[23]:


start_epoch = 0

(
    val_acc_max,
    dices_brain,
    dices_tumor,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=dataloader,
    val_loader=dataloader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
)


# In[ ]:

print(f"train completed, best average dice: {val_acc_max:.4f} ")


# ## Plot the Losses and Metrics


# ## Test the Model

# ### Read test json

# In[ ]:


# ### Create test dataset and dataloader

# In[ ]:

test_ds = get_test_dataset("/home/vedatb/senior-project/data/btsed_dataset")

test_loader = data.DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ### Load the best saved checkpoint and perform inference
# We select a single case from the validation set and perform inference to compare the model segmentation output with the corresponding label

# In[ ]:


model.load_state_dict(torch.load(os.path.join(root_dir, "model.pt"))["state_dict"])
model.to(device)
model.eval()

model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)


with torch.no_grad():
    for idx, batch_data in enumerate(test_loader):
        image = batch_data["img"].cuda()
        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[0] == 1] = 1
        seg_out[seg[1] == 1] = 2

        nrrd.write(
            file=f"/home/vedatb/senior-project/bbm47980_bts/old-predictions/{idx}.nrrd",
            data=seg_out,
        )
