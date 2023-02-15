import monai

from monai.networks.nets import SwinUNETR

from monai import data
from monai.data.image_reader import NrrdReader

from monai import transforms
from monai import losses

import os
import json
from dotenv import load_dotenv

from functools import partial

from brain_tumor_segmentation.bts_dataset import generate_sample_paths

PROJECT_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

img_size = [128, 128, 128]
model = partial(
    SwinUNETR,
    img_size = img_size,
    in_channels = 1,
    out_channels = 1
    # use default parameters
)
## transforms
img_transforms = transforms.compose.Compose(
    [
        transforms.LoadImaged(
            keys=["image", "label"],
            reader=NrrdReader()
        ),
        
    ]
)
## create dataloaders
paths = generate_sample_paths(os.environ["DATASET_PATH"])