from monai.config import KeysCollection
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    Resized,
    Transform,
)
from monai.utils.enums import TransformBackends
from torch.utils.data import DataLoader

import json

from typing import Callable, Optional, List, Dict
import os

MODAL2INDEX = {
    "flair" : 0,
    "t1" : 1,
    "t1c" : 2,
    "t2" : 3
}

LABEL2INDEX = {
    "background" : 0,
    "brain" : 1,
    "tumour" : 2
}

class PorcupineDataset(Dataset):
    """BTS Porcupine Dataset that contains t1, t1c, t2 annotated images.

    This dataset is constructed to be a demo dataset for the initial starting
    point of the project. Each sample has 3 classes (with their class indexes):
        0. Background
        1. Brain
        2. Tumour

    Note: The resolution of samples are not consistent. Resizing is needed if they will
    be used as batch inputs.

    Image Shape: (H,W,D)
    Label Shape: (H,W,D)
    """

    def __init__(
        self, dataset_root_path: str, transform: Optional[Callable] = None, **kwargs
    ) -> None:
        self.root_path = dataset_root_path
        self.n_classes = 3

        if not os.path.isdir(dataset_root_path):
            raise ValueError(f"{dataset_root_path} is not a valid dataset path")

        data_paths = self._generate_data_paths()
        self.n_samples = len(data_paths)

        if not transform:
            transform = Compose([
                LoadImaged(["image"], reader="NrrdReader"),
                LoadImaged(["label"], reader="NumpyReader")
            ])

        Dataset.__init__(self, data=data_paths, transform=transform, **kwargs)

    def get_num_classes(self) -> int:
        """Get number of classes.

        Returns:
            Number of classes.
        """
        return self.n_classes
    
    def __len__ (self) -> int:
        return self.n_samples

    def _generate_data_paths(self) -> List[Dict[str, str]]:
        """Generate data paths for image, label and info files.

        Each sample should have the following files in its directory:
            image.nrrd: Image.
            label.nrrd: Segmentation labels.
            info.json: Sample metadata.

        Returns:
            List of samples structured as a dictionary. Each sample have the following
            keys:
                img: Path to the image nrrd file.
                label: Path to the label nrrd file.
                info: Path to the metadata json file.
        """
        data_paths = []

        with open(os.path.join(self.root_path, "map.json"), "r") as f:
            info_json = json.load(f)
        
        for patient_info in info_json["files"]:
            patient_fp = os.path.join(self.root_path, patient_info["index"])

            fmask = [False, False, False, False]
            fmask[MODAL2INDEX[patient_info["modality"]]] = True
            
            patient_data_paths = {
                "image" : os.path.join(patient_fp, "image.nrrd"),
                "label" : os.path.join(patient_fp, "transform_label.npy"),
                "info" : {
                    "index" : patient_info["index"],
                    "modality" : patient_info["modality"]
                }
            }
            data_paths.append(patient_data_paths)

        return data_paths
    
if __name__ == "__main__":
    fpath = "/home/desmin/data/porcupine_dataset"
    dataset = PorcupineDataset(dataset_root_path=fpath)

    loader = DataLoader(dataset=dataset, batch_size=1)
    for item in loader:
        print(item["image"].shape)
        print(item["label"].shape)
        print(item["info"])
        # print(img.shape)
        # print(label.shape)
        # print(fmask)
        # print(info)
        print("==============")
    