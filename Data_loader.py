import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':20,
    'SEED':41
}


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        # PIL 이미지로 불러오기
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = torch.tensor(self.label_list[index], dtype=torch.float32)
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)
    
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_labels(df):
    return df.iloc[:,2:].values

def get_data_loader(val_df, train_df):
    train_labels = get_labels(train_df)
    val_labels = get_labels(val_df)
    
    train_dataset = CustomDataset(train_df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=2)

    val_dataset = CustomDataset(val_df['img_path'].values, val_labels, test_transform)
    val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)

    return train_loader, val_loader

def get_test_loader(test):
    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    return test_loader
