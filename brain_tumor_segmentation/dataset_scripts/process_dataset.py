## This script aims to convert an nrrd image to a certain format. Specially
## designed for manipulating label files

import argparse
import os
import pathlib
from typing import Dict

import nrrd
import numpy as np

parser = argparse.ArgumentParser(
    prog="Nrrd Label Converter for BTS Dataset sent by Dr. Şahin",
    description="""
    Converts labels to the desired format.
    Shape: HxWxD
    Brain Label: 1
    Tumour Label: 2
    """,
)
parser.add_argument("--path2image", type=str, required=True, help="Path to the image")
parser.add_argument(
    "--label_brain", type=int, required=True, help="Label used for brain in the file"
)
parser.add_argument(
    "--label_tumour", type=int, required=True, help="Label used for tumour in the file"
)
# only required if ndim is 4
parser.add_argument(
    "--brain_0th_axis_idx", type=int, help="Label used for brain in the file"
)
parser.add_argument(
    "--tumour_0th_axis_idx", type=int, help="Label used for tumour in the file"
)
args = parser.parse_args()


def transform_3d_nrrd(
    data: np.ndarray, label_brain: int, label_tumour: int
) -> np.ndarray:
    """Transform function for 3 dimensional NRRD images

    Args:
        data (np.ndarray): Data to be transformed, expected to have .ndim=4
        label_brain (int): The integer that refers to brain in the data
        label_tumour (int): The integer that refers to tumour in the data

    Returns:
        np.ndarray: Transformed data
    """
    new_data = np.zeros_like(data, dtype=np.uint8)
    # convert brain labels to 1
    new_data[data == label_brain] = 1
    # convert tumour labels to 2
    new_data[data == label_tumour] = 2

    return new_data


def transform_4d_nrrd(
    data: np.ndarray,
    label_brain: int,
    brain_0th_axis_idx: int,
    label_tumour: int,
    tumour_0th_axis_idx: int,
) -> np.ndarray:
    """Transform function for 4 dimensional NRRD images

    Args:
        data (np.ndarray): Data to be transformed, expected to have .ndim=4
        label_brain (int): The integer that refers to brain in the data
        brain_0th_axis_idx (int): In 0th axis, the channel has the brain labels
        label_tumour (int): The integer that refers to tumour in the data
        tumour_0th_axis_idx (int): In 0th axis, the channel has the tumour labels

    Returns:
        np.ndarray: Transformed data
    """
    brain_mask = data[brain_0th_axis_idx] == label_brain
    tumour_mask = data[tumour_0th_axis_idx] == label_tumour

    has_intersection = (brain_mask & tumour_mask).sum()
    print(f"Is there any intersection: {has_intersection > 0}")

    new_data = np.zeros(data.shape[1:], dtype=np.uint8)
    new_data[brain_mask] = 1
    new_data[tumour_mask] = 2

    return new_data


if __name__ == "__main__":
    filename = os.path.basename(args.path2image)
    path = pathlib.PurePath(args.path2image)
    print(f"Processing {path.parent.name}/{filename}...")
    print(path.parent)
    data, header = nrrd.read(filename=args.path2image)
    if data.ndim == 3:
        transformed_data = transform_3d_nrrd(data, args.label_brain, args.label_tumour)
    elif data.ndim == 4:
        try:
            args.brain_0th_axis_idx
            args.tumour_0th_axis_idx
        except AttributeError as e:
            print(e)
            print(
                (
                    "It seems like you gave an image with four dimensions but "
                    "haven't provided which indexes correspond to which labels. "
                    "Please set these values with `brain_0th_axis_idx` and "
                    "`brain_tumour_0th_axis_idx`."
                )
            )
            exit(1)
        transformed_data = transform_4d_nrrd(
            data,
            args.label_brain,
            args.brain_0th_axis_idx,
            args.label_tumour,
            args.tumour_0th_axis_idx,
        )

    print(f"Finished processing, writing to {path.parent}/transformed_label.nrrd")
    transformed_data
    nrrd.write(
        os.path.join(path.parent.__str__(), "transformed_label.nrrd"),
        transformed_data,
        header=header,
    )
