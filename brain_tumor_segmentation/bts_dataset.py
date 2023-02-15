import os

import json
from typing import Dict

from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends

import torch

def generate_sample_paths (
    root_path : str
):
    config_path = os.path.join(
        root_path,
        "data.json"
    )
    with open(config_path, "r") as file:
        config : Dict = json.load(file)
    
    sample_paths = []
    for sample_id, sample_content in config.items():
        mri_path = os.path.join(
            root_path,
            sample_id,
            sample_content["image"]
        )
        label_path = os.path.join(
            root_path,
            sample_id,
            sample_content["label"]
        )
        sample_paths.append({
            "image" : mri_path,
            "label" : label_path
        })
    
    return sample_paths

class ConvertToMultiChannelBasedOnBtsClasses (Transform):
    backend = [TransformBackends.TORCH]

    def __call__ (self, img : torch.Tensor):
        # expected ndim is 3 

