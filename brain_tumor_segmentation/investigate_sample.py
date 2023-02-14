from scipy.ndimage import affine_transform
import numpy as np

import torch

import monai
from monai import transforms
from monai.data.image_reader import NrrdReader

import matplotlib.pyplot as plt
import os
import json

from dotenv import load_dotenv
PROJECT_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

path_to_dataset = os.environ["DATASET_PATH"]

available_samples = []
for dir_name in os.listdir(path_to_dataset):
    if os.path.isdir(os.path.join(
        path_to_dataset,
        dir_name
    )):
        available_samples.append(dir_name)

for idx, sample_name in enumerate(available_samples):
    print(f"{idx} : {sample_name}")

idx = int(input(
    "Please type the number for the sample you want to investigate: ")
)
patient_name = available_samples[idx]
path_to_data = os.path.join(path_to_dataset, patient_name)

path_to_nrrd_image = None
path_to_nrrd_image_label = None
for file_name in os.listdir(path_to_data):
    if not os.path.isfile(
        os.path.join(path_to_data, file_name)
    ) or os.path.splitext(file_name)[1] != ".nrrd":
        continue

    if not path_to_nrrd_image_label and '.seg' in file_name:
        path_to_nrrd_image_label = os.path.join(
            path_to_data,
            file_name
        )
    elif not path_to_nrrd_image:
        path_to_nrrd_image = os.path.join(
            path_to_data,
            file_name
        )

    # found image and label
    if path_to_nrrd_image_label and path_to_nrrd_image:
        break

assert path_to_nrrd_image and path_to_nrrd_image_label, ("Couldn't find image "
f"and label in {path_to_data}")
print("\n")

print(f"{bcolors.OKGREEN}Using the following files:{bcolors.ENDC}")
print(f"{bcolors.OKBLUE}{bcolors.BOLD}Image file: {bcolors.ENDC}" 
      f"{path_to_nrrd_image}")
print(f"{bcolors.OKBLUE}{bcolors.BOLD}Label file: {bcolors.ENDC}" 
      f"{path_to_nrrd_image_label}")
print()

reader = NrrdReader()
image, image_data = transforms.LoadImage(
    reader = reader
)(path_to_nrrd_image)

label, label_data = transforms.LoadImage(
    reader = reader
)(path_to_nrrd_image_label)

print(f"{bcolors.BOLD}Image Properties{bcolors.ENDC}---------------")
print("Shape:", image.shape)
print("Orientation:", image_data["space"])
print()
print(f"{bcolors.BOLD}Label Properties{bcolors.ENDC}---------------")
print("Shape", label.shape)
print("Orientation:", label_data["space"])
if label.ndim == 3:
    print("Unique Numbers:", np.unique(label.numpy()))
else:
    print("unique vals in 0th dim", np.unique(label[0,:,:,:].numpy()), sep="\n")
    print("unique vals in 1th dim", np.unique(label[1,:,:,:].numpy()), sep="\n")