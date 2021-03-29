# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import h5py
import cv2
import glob
from time import sleep

class CrowdDataset(Dataset):
    def __init__(self, root_path, transform=None):
        # get all images to be tested
        root = glob.glob(os.path.join(root_path, 'test_data/images/*.jpg'))

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # get the image path
        img_path = self.lines[index]

        img, target = load_data(img_path)
        # perform data augumentation
        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)
        target = torch.Tensor(target)

        return img, target

def load_data(img_path):
    # get the path of the ground truth
    gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
    # open the image
    img = Image.open(img_path).convert('RGB')
    # load the ground truth
    while True:
        try:
            gt_file = h5py.File(gt_path)
            break
        except:
            sleep(2)
    target = np.asarray(gt_file['density'])

    return img, target