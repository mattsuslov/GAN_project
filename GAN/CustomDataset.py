import torchvision.transforms as tt
from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import pandas as pd
import random

class CustomDataset(Dataset):
    def __init__(self, features_path, index_col, images_path, transform):
        self.images_path = images_path
        self.features_df = pd.read_csv(features_path, index_col=index_col)
        self.transform = transform
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, index):
        name = self.features_df.index[index]
        image_path = os.path.join(self.images_path, name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        features1 = torch.tensor(self.features_df.iloc[index])
        features2 = torch.tensor(self.features_df.iloc[random.randint(0, len(self.features_df) - 1)])
        features3 = torch.tensor(self.features_df.iloc[random.randint(0, len(self.features_df) - 1)])
        return image, features1, features2, features3