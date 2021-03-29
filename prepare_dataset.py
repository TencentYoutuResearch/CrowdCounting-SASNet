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
import h5py
import scipy.io as io
import numpy as np
import os
import glob
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import cv2
import scipy.spatial

# define the argparser
def get_args_parser():
    parser = argparse.ArgumentParser('Data Preprocess', add_help=False)
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    
    return parser

# the function to generate the density map, with provided points
def generate_density_map(shape=(5, 5), points=None, f_sz=15, sigma=4):
    """
    generate density map given head coordinations
    """
    im_density = np.zeros(shape[0:2])
    h, w = shape[0:2]
    if len(points) == 0:
        return im_density
    # iterate over all the points
    for j in range(len(points)):
        # create the gaussian kernel
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        # limit the bound
        x = np.minimum(w, np.maximum(1, np.abs(np.int32(np.floor(points[j, 0])))))
        y = np.minimum(h, np.maximum(1, np.abs(np.int32(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        # get the rect around each head
        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))
        dx1 = 0
        dy1 = 0
        dx2 = 0
        dy2 = 0
        change_H = False
        if x1 < 1:
            dx1 = np.abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dy1 = np.abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dx1
        y1h = 1 + dy1
        x2h = f_sz - dx2
        y2h = f_sz - dy2
        if change_H: 
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)
        # attach the gaussian kernel to the rect of this head
        im_density[y1 - 1:y2, x1 - 1:x2] = im_density[y1 - 1:y2, x1 - 1:x2] + H
    return im_density


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Data Preprocess', parents=[get_args_parser()])
    args = parser.parse_args()

    # get all image of the dataset
    img_paths = []
    for root, dirs, files in os.walk(args.data_path):
        for img_path in files:
            # only jpg image 
            if img_path.endswith('.jpg'):
                img_paths.append(os.path.join(root, img_path))
     # iterate over all images
    for img_path in img_paths:
        print(img_path)
        # get the path of the GT
        gt_path = img_path.replace('.jpg', '.txt')
        gt = []
        # read gt line by line
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                gt.append([x, y])
        # load the image
        image = cv2.imread(img_path)
        # generate the density map
        positions = generate_density_map(shape=image.shape, points=np.array(gt), f_sz=15, sigma=4)
        # save the density map
        with h5py.File(img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = positions