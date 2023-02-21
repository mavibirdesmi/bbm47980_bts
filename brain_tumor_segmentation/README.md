# Brain Tumor Segmentation
<hr>

Constructing dataloader
```py
from brain_tumor_segmentation.bts_dataset import ( 
    ConvertToMultiChannelBasedOnBtsClassesd,
    generate_sample_paths_from_json
)

from monai import data
from monai import transforms

# get paths by using the json file
sample_paths = generate_sample_paths_from_json(DATASET_PATH)

# create transformations
img_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.ConvertToMultiChannelBasedOnBtsClassesd(keys=["label"])
    ## image alternation transformations
])

dataset = data.Dataset(
    data=sample_paths,
    transforms=img_transforms
)

dataloader = data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
)
```