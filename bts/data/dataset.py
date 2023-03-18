import json
import os
from typing import Callable, Dict, List, Optional

import torch
from monai.config import KeysCollection
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, MapTransform, Transform
from monai.utils.enums import TransformBackends
import yaml


from bts.common.miscutils import DotConfig


def read_local_labels() -> DotConfig[str, int]:
    file_dir = os.path.dirname(__file__)
    labels_path = os.path.join(file_dir, "labels.yaml")

    with open(labels_path) as fp:
        labels = DotConfig(yaml.safe_load(fp))
    return labels

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
    """Converts 3 dimensional label to 4 dimensional based on the Brain Tumor
    Classification (BTS) dataset.

    Note that we do not explicitly make use of the ground label, since we can
    infer it by using the predictions.

    label 1 is the brain
    label 2 is the tumour
    """

    backend = [TransformBackends.TORCH]

    def __init__(self):
        self.labels = read_local_labels()

    def __call__(self, img: torch.Tensor):
        # expected ndim is 3
        assert img.ndim == 3, "Image is expected to be 3 dimensional"

        brain_label = self.labels.BRAIN
        tumor_label = self.labels.TUMOR
        result = [
            img == brain_label,
            img == tumor_label,
        ]
        return torch.stack(result, dim=0)


class ConvertToMultiChannelBasedOnBtsClassesd(MapTransform):
    """Dictionary based wrapper of ConvertToMultiChannelBasedOnBtsClasses.

    Converts 3 dimensional label to 4 dimensional based on the Brain Tumor
    Classification (BTS) dataset.

    Label 1 is the brain and label 2 is the tumour.
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
    """BTS Echidna Dataset that contains 8 t1w annotated images.

    This dataset is constructed to be a demo dataset for the initial starting
    point of the project. It contains annotated t1w images of 8 patients. Each
    sample has 3 classes (with their class indexes):
        0. Background
        1. Brain
        2. Tumour

    Note: The resolution of samples are not consistent.

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
        self.labels = read_local_labels()
        self.n_classes = len(self.labels)

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
        """Get number of classes.

        Returns:
            Number of classes.
        """
        return self.n_classes

    def _generate_data_paths(self) -> List[Dict[str, str]]:
        """Generate data paths for image, label and info files.

        Returns:
            List of samples structured as a dictionary. Each sample have the following
            keys:
                img: Path to image nrrd file.
                label: Path to the labels nrrd file.
                info: Path to the metadata json file.
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
