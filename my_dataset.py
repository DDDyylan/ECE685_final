import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageSet(Dataset):
    def __init__(self, image_paths, image_labels, transforms, root_adj = ''):
        super().__init__()
        self.images_paths = image_paths
        self.transforms = transforms
        self.image_labels = image_labels
        self.root_adj = root_adj # adjust path based on your folder structure

    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.root_adj + self.images_paths[idx])
        label = self.image_labels[idx]
        image = self.transforms(image)
        bbox = torch.zeros(2)
        if label == 1:
            bbox = bbox + image.shape[1]
        return image, label, bbox
