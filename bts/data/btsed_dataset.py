import json
import os
from typing import Callable, Dict, List, Optional

import torch
from monai.config import KeysCollection
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, MapTransform, Transform
from monai.utils.enums import TransformBackends


class JsonTransform(MapTransform):
    def __call__(self, data):
        data = dict(data)
        for item_key in self.keys:
            file_path = data[item_key]
            with open(file_path, "r") as file:
                file_as_dict = json.load(file)
                data[item_key] = file_as_dict

        return data


class ConvertToMultiChannelBasedOnBtsClasses(Transform):
    """
    Converts 3 dimensional label to 4 dimensional based on the Brain Tumor
    Classification (BTS) dataset

    label 1 is the brain (gray matter)
    label 2 is the tumour
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, img: torch.Tensor):
        # expected ndim is 3
        assert img.ndim == 3, "Image is expected to be 3 dimensional"

        result = [img == 1, img == 2]
        return torch.stack(result, dim=0)


class ConvertToMultiChannelBasedOnBtsClassesd(MapTransform):
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


class BrainTumourSegmentationEchidnaDataset(Dataset):
    """BTS Echidna Dataset that contains 8 t1w annotated images

    This dataset is constructed to be a demo dataset for the initial starting
    point of the project. It contains annotated t1w images of 8 patients. Each
    sample has 3 classes (with their class indexes):
        0. Background
        1. Brain (not a specfic part, includes gray and white matter)
        2. Tumour

    *The resolution of samples are not consistent.

    Image Shape: (H,W,D)
    Label Shape: (H,W,D)

    Example:
    ```py
        dataset = BrainTumourSegmentationEchidnaDataset(
            dataset_root_path="/path/to/dataset"
        )
        d_loader = DataLoader(dataset=dataset)
        for sample in d_loader:
            img, label, info = sample["img"], sample["label"], sample["info"]
            # do magic
    ```
    """

    def __init__(
        self, dataset_root_path: str, transform: Optional[Callable] = None, **kwargs
    ) -> None:
        self.root_path = dataset_root_path
        self.n_samples = 8
        self.n_classes = 3

        if not os.path.isdir(dataset_root_path):
            raise ValueError(f"{dataset_root_path} is not a valid dataset path")

        data_paths = self._generate_data_paths()

        if not transform:
            transform = Compose(
                [
                    LoadImaged(["img", "label"], reader="NrrdReader"),
                    ConvertToMultiChannelBasedOnBtsClassesd(["label"]),
                    JsonTransform(["info"]),
                ]
            )

        Dataset.__init__(self, data=data_paths, transform=transform, **kwargs)

    def get_num_classes(self) -> int:
        """Get number of classes

        Returns:
            int: Number of classes
        """
        return self.n_classes

    def _generate_data_paths(self) -> List[Dict[str, str]]:
        """Generate data paths for image, label and info files

        Returns:
            List[Dict[str, str]]: List of sample file paths, keys for each file
            are img, label and info
        """
        data_paths = []

        for folder_index in range(self.n_samples):
            sample_folder_path = os.path.join(self.root_path, str(folder_index))

            sample_data_paths = {
                "img": os.path.join(sample_folder_path, "image.nrrd"),
                "label": os.path.join(sample_folder_path, "label.nrrd"),
                "info": os.path.join(sample_folder_path, "info.json"),
            }
            data_paths.append(sample_data_paths)

        return data_paths
