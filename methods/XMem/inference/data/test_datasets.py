import os
from os import path
import json
import cv2
import shutil
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from util.image import _palette
from inference.data.video_reader import VideoReader


class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames
    
    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=True
            )

    def __len__(self):
        return len(self.vid_list)

class VOSTTestDataset:
    def __init__(self, data_root, split, size=-1):

        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        seq_names = []
        self.size = size
        with open(os.path.join(data_root, 'ImageSets', split + '.txt')) as f:
            seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seq_names.extend(seqs_tmp)
        self.seqs = list(np.unique(seq_names))
    
    def __len__(self):
        return len(self.seqs)
    
    def get_datasets(self):
        for video in self.seqs:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
            )
   