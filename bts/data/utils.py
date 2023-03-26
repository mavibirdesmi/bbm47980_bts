import os
from typing import IO, Any, Dict, List, Optional, Union

import nrrd
import torch
from monai.transforms import MapTransform, Transform

from bts.common import logutils

logger = logutils.get_logger(__name__)


class UnsqueezeData(Transform):
    def __call__(self, data: torch.Tensor):
        return torch.unsqueeze(data, 0)


class UnsqueezeDatad(MapTransform):
    def __init__(self, keys: List, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = UnsqueezeData()

    def __call__(self, data):
        data_dict = dict(data)
        for key in self.key_iterator(data_dict):
            data_dict[key] = self.converter(data_dict[key])

        return data_dict


def save_prediction_as_nrrd(
    prediction: torch.Tensor,
    index_in_batch: int,
    file: Union[str, IO],
    meta_dict: Optional[Dict[str, Any]] = None,
):
    """Saves the tensor prediction as nrrd file.

    Args:
        prediction: Prediction from the model.
        index_in_batch: Index of the sample to be saved, this is needed due
            to how meta_dict is constructed while being collated.
        file: File path or IO to be used for saving the prediction.
        meta_dict: If given, a header is created from the meta_dict values. This option
            is suggested if you want to view predictions in applications
            (e.g. 3D Slicer) with the original image. Defaults to None.

    Example:
    ```py
    >>  prediction = model(sample['img'])
    >>  for i, sample_pred in enumerate(prediction):
            save_prediction_as_nrrd(
                sample_pred,
                i,
                f"prediction_{i}.nrrd",
                sample['img_meta_dict']
            )
    ```
    """
    dirname = os.path.dirname(file)

    os.makedirs(dirname, exist_ok=True)

    # convert meta dict to suitable header
    prediction_header = {
        "type": meta_dict["type"][index_in_batch],
        "dimension": meta_dict["dimension"][index_in_batch].item(),
        "space": meta_dict["space"][index_in_batch],
        "sizes": prediction.shape,
        "space directions": meta_dict["affine"][index_in_batch, :3, :3].numpy(),
        "space origin": meta_dict["affine"][index_in_batch, :3, -1].numpy(),
    }
    nrrd.write(file=file, data=prediction, header=prediction_header)


def collate_fn(batch: Any) -> Any:
    """Collate list of data.

    Args:
        batch: List of data.

    Returns:
        Collated data.
    """
    elem = batch[0]
    coll = {}

    for key in elem:
        if key in ["img", "label"]:
            data_for_batch = tuple(torch.as_tensor(d[key]) for d in batch)
            coll[key] = torch.stack(data_for_batch, dim=0)
        else:
            coll[key] = torch.stack(data_for_batch, dim=0)

    return coll
