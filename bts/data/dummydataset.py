import uuid
from typing import Dict, Tuple

import torch
from monai.data import DataLoader, Dataset


def generate_dummy_data(
    n: int = 10, sample_shape: Tuple[int, int, int] = (128, 128, 128)
):
    data = {}

    # 1 channel: T1 images
    NUM_CHANNELS = 1

    # Brain & Tumor
    NUM_LABELS = 2

    for _ in range(n):
        random_patient_id = uuid.uuid4()

        data[random_patient_id] = {
            "image": torch.rand(NUM_CHANNELS, *sample_shape, dtype=torch.float32),
            "label": torch.randint(
                low=0, high=2, size=(NUM_LABELS, *sample_shape), dtype=torch.bool
            ),
        }

    return data


class DummyDataset(Dataset):
    def __init__(self, data: Dict[str, Dict[str, str]]):
        self.data = data
        self.keys = list(data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[self.keys[index]]


def get_dataloader(batch_size: int = 2):
    DUMMY_DATA = generate_dummy_data()

    dataset = DummyDataset(DUMMY_DATA)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    return dataloader
