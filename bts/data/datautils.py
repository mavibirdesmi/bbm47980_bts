from typing import IO, Any, Dict, Union

import nrrd
import torch


def save_prediction_as_nrrd(
    prediction: torch.Tensor,
    index_in_batch: int,
    file: Union[str, IO],
    meta_dict: Dict[str, Any] = None,
):
    """_summary_

    Args:
        prediction (torch.Tensor): _description_
        file (IO): _description_
        meta_dict (Dict[str, Any], optional): _description_. Defaults to None.

    Example:
    ```
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
    if prediction.is_cuda:
        prediction = prediction.cpu()

    prediction_numpy = prediction.numpy()

    # convert meta dict to suitable header
    prediction_header = {
        "type": meta_dict["type"][index_in_batch],
        "dimension": meta_dict["dimension"][index_in_batch].item(),
        "space": meta_dict["space"][index_in_batch],
        "sizes": prediction.shape,
        "space directions": meta_dict["affine"][index_in_batch, :3, :3].numpy(),
        "space origin": meta_dict["affine"][index_in_batch, :3, -1].numpy(),
    }
    nrrd.write(file=file, data=prediction_numpy, header=prediction_header)
