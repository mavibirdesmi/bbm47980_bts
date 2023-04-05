#!/usr/bin/env python
# coding: utf-8

# Inspired from https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb

# In[1]:


import time

import matplotlib
import matplotlib.pyplot as plt
import nrrd
import numpy as np
from tqdm.auto import tqdm

import wandb
from bts.data.dataset import get_test_dataset, get_train_dataset

# In[2]:


wandb.init(name="Old code")


# In[3]:


import json
import os
from typing import Dict

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
    model,
    loader,
    optimizer,
    epoch,
    loss_function,
):
    model.train()
    run_loss = AverageMeter()
    start_time = time.time()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["img"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()

        logits = model(data)

        loss = loss_function(logits, target)

        loss.backward()
        optimizer.step()

        run_loss.update(loss.item(), n=batch_size)

        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "Loss: {:.4f}".format(run_loss.avg),
            "Time {:.2f}s".format(time.time() - start_time),
        )

        start_time = time.time()

    wandb.log({"Mean Train Loss": run_loss.avg})

    return run_loss.avg


# In[21]:


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["img"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_brain = run_acc.avg[0]
            dice_tumor = run_acc.avg[1]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_brain:",
                dice_brain,
                ", dice_tumor:",
                dice_tumor,
            )

    wandb.log(
        {
            "Mean Val Brain Acc": run_acc.avg[0],
            "Mean Val Tumor Acc": run_acc.avg[1],
            "Mean Val Acc": run_acc.avg.mean(),
        }
    )

    return run_acc.avg


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
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_function=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "Loss: {:.4f}".format(train_loss),
            "Time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_brain = val_acc[0]
            dice_tumor = val_acc[1]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_brain:",
                dice_brain,
                ", dice_tumor:",
                dice_tumor,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_brain.append(dice_brain)
            dices_tumor.append(dice_tumor)
            dices_avg.append(val_avg_acc)
            if dice_tumor > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, dice_tumor))
                val_acc_max = dice_tumor
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )
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
