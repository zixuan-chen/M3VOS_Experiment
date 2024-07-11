from __future__ import division
import os
import shutil
import json
import cv2
from PIL import Image

import numpy as np
from torch.utils.data import Dataset

from methods.RMem.aot_plus.utils.image import _palette


class VOSTest(Dataset):
    def __init__(self,
                 image_root,
                 label_root,
                 seq_name,
                 images: list,
                 labels: list,
                 rgb=True,
                 transform=None,
                 single_obj=False,
                 resolution=None):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb
        self.single_obj = single_obj
        self.resolution = resolution

        self.obj_nums = []
        self.obj_indices = []

        curr_objs = [0]
        for img_name in self.images:
            self.obj_nums.append(len(curr_objs) - 1)
            current_label_name = img_name.split('.')[0] + '.png'
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                curr_obj = list(np.unique(current_label))
                for obj_idx in curr_obj:
                    if obj_idx not in curr_objs:
                        curr_objs.append(obj_idx)
            self.obj_indices.append(curr_objs.copy())

        self.obj_nums[0] = self.obj_nums[1]

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        if isinstance(self.images[idx], str):
            img_name = self.images[idx]
            img_path = os.path.join(self.image_root, self.seq_name, img_name)
            img = cv2.imread(img_path)
            img = np.array(img, dtype=np.float32)
            if self.rgb:
                img = img[:, :, [2, 1, 0]]
            return img
        else:
            return self.images[idx]

    def read_label(self, label_name, squeeze_idx=None):
        if label_name is None:
            return None
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        if self.single_obj:
            label = (label > 0).astype(np.uint8)
        elif squeeze_idx is not None:
            squeezed_label = label * 0
            for idx in range(len(squeeze_idx)):
                obj_id = squeeze_idx[idx]
                if obj_id == 0:
                    continue
                mask = label == obj_id
                squeezed_label += (mask * idx).astype(np.uint8)
            label = squeezed_label
        return label

    def insert_frames(self, step, num_frames, frame):
        for i in range(num_frames):
            self.images.insert(step, frame)
            self.labels.insert(step, None)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(
                float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        if isinstance(img_name, str):
            current_label_name = img_name.split('.')[0] + '.png'
        else:
            current_label_name = None
        obj_num = self.obj_nums[idx]
        obj_idx = self.obj_indices[idx]

        if current_label_name and current_label_name in self.labels:
            current_label = self.read_label(current_label_name, obj_idx)
            sample = {
                'current_img': current_img,
                'current_label': current_label
            }
        else:
            sample = {'current_img': current_img}

        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': self.num_frame,
            'obj_num': obj_num,
            'current_name': img_name,
            'height': height,
            'width': width,
            'flip': False,
            'obj_idx': obj_idx
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample