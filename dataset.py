import os
from PIL import Image
from torch.utils.data import Dataset

DIR_TRAIN = r"FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET\train"
DIR_TEST = r"FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET\test"

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