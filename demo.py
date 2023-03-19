from functools import partial
from os.path import join as osjoin
from typing import Any, Dict, List

import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    Transform,
)

from bts.data import (
    BrainTumourSegmentationEchidnaDataset,
    ConvertToMultiChannelBasedOnBtsClassesd,
    JsonTransform,
    save_prediction_as_nrrd,
)
from bts.swinunetr.model import get_model

SPATIAL_SIZE = (128, 128, 128)
DATASET_PATH = "/home/desmin/grad_project/echidna_dataset/btsed_dataset"
BATCH_SIZE = 1
DEVICE = "cuda"
PREDICTION_DIR = "/home/desmin/grad_project/predictions"


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


dataset = BrainTumourSegmentationEchidnaDataset(
    dataset_root_path=DATASET_PATH,
    transform=Compose(
        [
            LoadImaged(["img", "label"], reader="NrrdReader"),
            UnsqueezeDatad(["img"]),
            ConvertToMultiChannelBasedOnBtsClassesd(["label"]),
            # for training
            # RandSpatialCropd(
            # ["img", "label"], roi_size=SPATIAL_SIZE, random_size=False),
            JsonTransform(["info"]),
        ]
    ),
)

train_dataset = dataset[:4]
test_dataset = dataset[4:]

print(
    f"Train dataset length: {len(train_dataset)}\n"
    "Test dataset length: {len(test_dataset)}"
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_dataset = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

model = get_model(
    img_size=SPATIAL_SIZE,
    in_channels=1,
    out_channels=3,
    feature_size=12,
    use_checkpoint=True,
)
model = model.to(DEVICE)

model = torch.nn.DataParallel(model)
model = model.eval()

model_inferer = partial(
    sliding_window_inference,
    roi_size=SPATIAL_SIZE,
    sw_batch_size=BATCH_SIZE,
    predictor=model,
)

post_softmax = Activations(softmax=True)
post_pred = AsDiscrete(argmax=True)

with torch.no_grad():
    for idx, sample in enumerate(train_loader):
        img, label, info = sample["img"], sample["label"], sample["info"]

        img: torch.Tensor
        label: torch.Tensor
        info: Dict[str, Any]

        img_meta_data = sample["img_meta_dict"]

        img = img.to(DEVICE)
        label = label.to(DEVICE)

        # logits
        logits = model_inferer(img)

        batch_labels = decollate_batch(label)

        # scalar
        preds = torch.stack(
            [post_pred(post_softmax(val_pred_tensor)) for val_pred_tensor in logits]
        )

        for sample_idx, sample_pred in enumerate(preds):
            save_prediction_as_nrrd(
                sample_pred[0],
                sample_idx,
                osjoin(
                    PREDICTION_DIR,
                    f"{info['patient_name'][sample_idx]}_prediction.nrrd",
                ),
                meta_dict=img_meta_data,
            )
