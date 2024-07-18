import os

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, root_dir_data):
        self.root_data = root_dir_data
        self.rgb_files = sorted([p for p in os.listdir(self.root_data) if p.endswith('r.png')])
        self.depth_files = sorted([p for p in os.listdir(self.root_data) if p.endswith('d.npy')])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_name = self.rgb_files[idx]
        depth_name = self.depth_files[idx]
        bgr_img = cv2.imread(os.path.join(self.root_data, rgb_name), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        depth_img = np.load(os.path.join(self.root_data, depth_name))
        return rgb_img, depth_img