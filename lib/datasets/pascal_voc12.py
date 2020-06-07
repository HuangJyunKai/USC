# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset
from tqdm import tqdm

class Pascal_voc12(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=21,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=520, 
                 crop_size=(480, 480), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],):
    
        super(Pascal_voc12, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)
        _mask_dir = os.path.join(root, 'SegmentationClass')
        _image_dir = os.path.join(root, 'JPEGImages')
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        _splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        # prepare data
        if 'train'in self.split:
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif 'val' in self.split:
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif 'test' in self.split:
            self.images = []
            return
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        
        self.label_mapping = {0: 0, 14: 1, 
                              19: 2, 33: 3, 
                              37: 4, 38: 5, 
                              52: 6, 57: 7, 
                              72: 8, 75: 9, 89: 10, 
                              94: 11, 108: 12, 112: 13, 
                              113: 14, 128: 15, 132: 16, 
                              147: 17, 150: 18, 220: 19}

        # generate masks
        '''
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        '''
        #self._key = np.array(range(len(self._mapping))).astype('uint8')
        
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if 'test' not in self.split:
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if 'test' not in self.split:
            assert (len(self.images) == len(self.masks))
        '''
        total = []
        for i in range(len(self.masks)):
            target = cv2.imread(self.masks[i],cv2.IMREAD_GRAYSCALE)
            total.append(np.unique(target).tolist())
        def flatmatrix(matrix):
            result = []
            for i in range(len(matrix)):
                result.extend(matrix[i])
            return result
        total = flatmatrix(total)
        print(np.unique(np.array(total)))
        '''
            
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    def __getitem__(self, index):
        #item = self.files[index]
        #name = item['file_name']
        #img_id = item['image_id']

        #image = cv2.imread(os.path.join(self.detail.img_folder,name),
        #                   cv2.IMREAD_COLOR)
        name = ""
        image = cv2.imread(self.images[index],cv2.IMREAD_COLOR)
        target = cv2.imread(self.masks[index],cv2.IMREAD_GRAYSCALE)
        #label = np.asarray(self.masks[img_id],dtype=np.int)
        size = image.shape
        target = self.convert_label(target)
        if self.split == 'val':
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            label = cv2.resize(target, self.crop_size, 
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif self.split == 'testval':
            # evaluate model on val dataset
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(target)
        else:
            image, label = self.gen_sample(image, target, 
                                self.multi_scale, self.flip)
        #label[ label == 255] = 0
        #print(np.unique(label))                       
        return image.copy(), label.copy(), np.array(size), name
    
    def __len__(self):
        return len(self.images)
