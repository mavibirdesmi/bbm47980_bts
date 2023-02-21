import os

import json
from typing import Dict

from monai.transforms.transform import Transform, MapTransform
from monai.config import KeysCollection
from monai.utils.enums import TransformBackends

import torch

def generate_sample_paths_from_json (
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
    """
    Converts 3 dimensional label to 4 dimensional based on the Brain Tumor
    Classification (BTS) dataset

    label 1 is the brain (gray matter)
    label 2 is the tumour
    """
    backend = [TransformBackends.TORCH]

    def __call__ (self, img : torch.Tensor):
        # expected ndim is 3 
        assert img.ndim == 3, "Image is expected to be 3 dimensional"

        result = [img == 1, img == 2]
        return torch.stack(result, dim=0)


class ConvertToMultiChannelBasedOnBtsClassesd (MapTransform):
    """
    Dictionary based wrapper of ConvertToMultiChannelBasedOnBtsClasses

    Converts 3 dimensional label to 4 dimensional based on the Brain Tumor
    Classification (BTS) dataset

    label 1 is the brain (gray matter)
    label 2 is the tumour
    """

    backend = ConvertToMultiChannelBasedOnBtsClasses.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBtsClasses()
    
    def __call__(self, data):
        data_dict = dict(data)
        for key in self.key_iterator(data_dict):
            data_dict[key] = self.converter(data_dict[key])
        
        return data_dict