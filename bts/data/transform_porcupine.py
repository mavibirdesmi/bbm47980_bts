import nrrd
import numpy as np
import os
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm

import json

BRAIN_LABEL_NAMES = set(["brain"])
TUMOR_LABEL_NAMES = set(["tumor", "tumr", "Segment_3", "Segment_5", "tissue"])

BRAIN_LABEL = 1
TUMOUR_LABEL = 2

def get_class_indexes (head : OrderedDict):

    segment0_name = head["Segment0_Name"] if "Segment0_Name" in head.keys() else None
    segment1_name = head["Segment1_Name"] if "Segment1_Name" in head.keys() else None

    brain_class_index, tumor_class_index = None, None
    if segment0_name == None and segment1_name == None:
        pass # bad data

    if segment1_name in BRAIN_LABEL_NAMES:
        brain_class_index = 2
        tumor_class_index = 1
    elif segment1_name in TUMOR_LABEL_NAMES:
        tumor_class_index = 2
        brain_class_index = 1
    elif segment0_name in BRAIN_LABEL_NAMES:
        brain_class_index = 1
        tumor_class_index = 2
    elif segment0_name in TUMOR_LABEL_NAMES:
        tumor_class_index = 1
        brain_class_index = 2

    return brain_class_index, tumor_class_index

def convert_4d_label (label : np.ndarray, brain_idx : int, tumour_idx : int):

    N, H, W, D = label.shape
    label_tr = np.zeros((H, W, D), dtype=np.uint8)

    label_tr[label[brain_idx - 1] == 1] = BRAIN_LABEL
    label_tr[label[tumour_idx - 1] == 1] = TUMOUR_LABEL

    return label_tr

def convert_3d_label (label : np.ndarray, brain_idx : int, tumour_idx : int):

    H, W, D = label.shape
    label_tr = np.zeros((H, W, D), dtype=np.uint8)

    label_tr[label == brain_idx] = BRAIN_LABEL
    label_tr[label == tumour_idx] = TUMOUR_LABEL

    return label_tr

def add_modality_info (json_path : str):

    t1 = set(["64", "3", "113", "118", "58", "52"])
    t2 = set(["79"])
    # t1c remaining

    with open(json_path, "r") as f:
        info_json = json.load(f)

    for patient_info in info_json["files"]:

        if patient_info["index"] in t1:
            patient_info["modality"] = "t1"
        elif patient_info["index"] in t2:
            patient_info["modality"] = "t2"
        else:
            patient_info["modality"] = "t1c"

    with open(json_path, "w") as f:
        f.write(json.dumps(info_json, indent=4))


if __name__ == "__main__":
    path = "/home/desmin/data/porcupine_dataset"
    new_dataset_path = "/home/desmin/data/porcupine_dataset_label_transforms"
    json_path = "/home/desmin/data/porcupine_dataset/map.json"

    add_modality_info(json_path)

    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "label.nrrd":
                fpath = os.path.join(root, file)
                img, head = nrrd.read(fpath)
                
                brain_idx, tumour_idx = get_class_indexes(head)
                if img.ndim == 3:
                    label_tr = convert_3d_label(img, brain_idx, tumour_idx)
                elif img.ndim == 4:
                    label_tr = convert_4d_label(img, brain_idx, tumour_idx)

                assert label_tr.ndim == 3 and 2 <= len(np.unique(label_tr)) <= 3, "Transformation failed"
                
                file_idx = os.path.basename(os.path.normpath(root))
                np.save(
                    os.path.join(root, f"transform_label.npy"),
                    label_tr
                )
                print(f"Saved {os.path.join(root, f'transform_label.npy')}")
                
