import os
from PIL import Image
from torch.utils.data import Dataset, Subset
import numpy as np
from torchvision import transforms

class FireEyeDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.classes = ["Smoke", "fire", "non fire"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []

        for cls in self.classes:
            class_dir = os.path.join(img_dir, cls)

            for file in os.listdir(class_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    path = os.path.join(class_dir, file)
                    label = self.class_to_idx[cls]

                    self.samples.append([path, label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

import copy

def create_datasets(
    train_dir,
    test_dir,
    transform=None,
    train_ratio=0.85,
    val_ratio=0.15,
    seed=13062026,
):

    # Dataset original de train
    full_train_dataset = FireEyeDataset(
        train_dir,
        transform=transform
    )

    # Dataset de test separado
    test_dataset = FireEyeDataset(
        test_dir,
        transform=transform
    )

    size = len(full_train_dataset)

    # Mezclar índices
    indices = np.random.RandomState(seed).permutation(size)

    # Punto de división
    train_end = int(train_ratio * size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    # Crear copias
    train_dataset = copy.deepcopy(full_train_dataset)
    val_dataset = copy.deepcopy(full_train_dataset)

    # Filtrar samples
    train_dataset.samples = [
        full_train_dataset.samples[i]
        for i in train_indices
    ]

    val_dataset.samples = [
        full_train_dataset.samples[i]
        for i in val_indices
    ]

    return train_dataset, val_dataset, test_dataset


DIR_TRAIN = r"FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET\train"
DIR_TEST = r"FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET\test"

MEAN_DATASET = [0.4414869,  0.41371821, 0.38738546]
STD_DATASET = [0.29328358, 0.29236583, 0.30662562]

transform_fireeye = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop((400, 225)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN_DATASET,
        std=STD_DATASET
    )
])

fireeye_train, fireeye_val, fireeye_test = create_datasets(
    DIR_TRAIN,
    DIR_TEST,
    transform=transform_fireeye
)
