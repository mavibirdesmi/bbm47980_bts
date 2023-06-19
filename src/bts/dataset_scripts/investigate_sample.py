import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from monai import transforms
from monai.data.image_reader import NrrdReader

PROJECT_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


path_to_dataset = os.environ["DATASET_PATH"]

available_samples = []
for dir_name in os.listdir(path_to_dataset):
    if os.path.isdir(os.path.join(path_to_dataset, dir_name)):
        available_samples.append(dir_name)

for idx, sample_name in enumerate(available_samples):
    print(f"{idx} : {sample_name}")

idx = int(input("Please type the number for the sample you want to investigate: "))
patient_name = available_samples[idx]
path_to_data = os.path.join(path_to_dataset, patient_name)

path_to_nrrd_image = None
path_to_nrrd_image_label = None
for file_name in os.listdir(path_to_data):
    if (
        not os.path.isfile(os.path.join(path_to_data, file_name))
        or os.path.splitext(file_name)[1] != ".nrrd"
    ):
        continue

    if not path_to_nrrd_image_label and ".seg" in file_name:
        path_to_nrrd_image_label = os.path.join(path_to_data, file_name)
    elif not path_to_nrrd_image:
        path_to_nrrd_image = os.path.join(path_to_data, file_name)

    # found image and label
    if path_to_nrrd_image_label and path_to_nrrd_image:
        break

assert path_to_nrrd_image and path_to_nrrd_image_label, (
    "Couldn't find image " f"and label in {path_to_data}"
)
print("\n")

print(f"{bcolors.OKGREEN}Using the following files:{bcolors.ENDC}")
print(
    f"{bcolors.OKBLUE}{bcolors.BOLD}Image file: {bcolors.ENDC}" f"{path_to_nrrd_image}"
)
print(
    f"{bcolors.OKBLUE}{bcolors.BOLD}Label file: {bcolors.ENDC}"
    f"{path_to_nrrd_image_label}"
)
print()

reader = NrrdReader(index_order="F")
image, image_data = transforms.LoadImage(reader=reader)(path_to_nrrd_image)

label, label_data = transforms.LoadImage(reader=reader)(path_to_nrrd_image_label)

print(f"{bcolors.BOLD}Image Properties{bcolors.ENDC}---------------")
print("Shape:", image.shape)
print("Orientation:", image_data["space"])
print()
print(f"{bcolors.BOLD}Label Properties{bcolors.ENDC}---------------")
print("Shape", label.shape)
print("Orientation:", label_data["space"])
if label.ndim == 3:
    print("Unique Numbers:", np.unique(label.numpy()))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image[:, :, 50], cmap="gray")
    axs[0].title.set_text("Image")
    axs[1].imshow(label[:, :, 50], cmap="gray")
    axs[1].title.set_text("Label")
else:
    print("unique vals in 0th dim", np.unique(label[0, :, :, :].numpy()), sep="\n")
    print("unique vals in 1th dim", np.unique(label[1, :, :, :].numpy()), sep="\n")

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(image[:, :, 60], cmap="gray")
    axs[0].title.set_text("Image")
    axs[1].imshow(label[0, :, :, 60], cmap="gray")
    axs[1].title.set_text("Label 0th Dim")
    axs[2].imshow(label[1, :, :, 60], cmap="gray")
    axs[2].title.set_text("Label 1th Dim")

plt.savefig(os.path.join(PROJECT_DIR, "..", "assets", f"{patient_name}.png"))