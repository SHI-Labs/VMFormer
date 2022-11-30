# ------------------------------------------------------------------------
# VMFormer data loader
# ------------------------------------------------------------------------
# Modified from RVM (https://github.com/PeterL1n/RobustVideoMatting)
# Copyright (c) 2021 ByteDance Inc. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import datasets.transforms as T
import os
from PIL import Image
from random import randint
import cv2
import random
import math
import time

class TrainFrameSampler:
    def __init__(self, speed=[0.5, 1, 2, 3, 4, 5]):
        self.speed = speed
    
    def __call__(self, seq_length):
        frames = list(range(seq_length))
        
        # Speed up
        speed = random.choice(self.speed)
        frames = [int(f * speed) for f in frames]
        
        # Shift
        shift = random.choice(range(seq_length))
        frames = [f + shift for f in frames]
        
        # Reverse
        if random.random() < 0.5:
            frames = frames[::-1]

        return frames

class VideoMatteDataset(Dataset):
    def __init__(self,
                 imagematte_dir,
                 lr_videomatte_dir,
                 hr_videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.imagematte_dir = imagematte_dir
        self.imagematte_files = os.listdir(os.path.join(imagematte_dir, 'fgr'))
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
        self.lr_videomatte_dir = lr_videomatte_dir
        self.lr_videomatte_clips = sorted(os.listdir(os.path.join(lr_videomatte_dir, 'fgr')))
        self.lr_videomatte_frames = [sorted(os.listdir(os.path.join(lr_videomatte_dir, 'fgr', clip))) 
                                  for clip in self.lr_videomatte_clips]
        self.lr_videomatte_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.lr_videomatte_clips)) 
                               for frame_idx in range(0, len(self.lr_videomatte_frames[clip_idx]), seq_length)]
        
        self.hr_videomatte_dir = hr_videomatte_dir
        self.hr_videomatte_clips = sorted(os.listdir(os.path.join(hr_videomatte_dir, 'fgr')))
        self.hr_videomatte_frames = [sorted(os.listdir(os.path.join(hr_videomatte_dir, 'fgr', clip))) 
                                  for clip in self.hr_videomatte_clips]
        self.hr_videomatte_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.hr_videomatte_clips)) 
                               for frame_idx in range(0, len(self.hr_videomatte_frames[clip_idx]), seq_length)]
        
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.lr_videomatte_idx)

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        if random.random() < 0.2:
            fgrs, phas = self._get_imagematte()
        elif random.random() < 0.6:
            fgrs, phas = self._get_lr_videomatte(idx)
        else:
            fgrs, phas = self._get_hr_videomatte(idx)

        if self.transform is not None:
            fgrs, phas, bgrs = self.transform(fgrs, phas, bgrs)
            imgs = []
            bgr_phas = []
            for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
                img = fgr * pha + bgr * (1 - pha)
                imgs.append(img)
                bgr_phas.append(1.0 - pha)
        target = {'masks': torch.cat(phas, dim=0), 'bgr_masks': torch.cat(bgr_phas, dim=0)}
        return torch.cat(imgs, dim=0), target
        #### torch.cat(imgs,dim=0) [self.seq_lengthx3, H, W]
        #### torch.cat(phas,dim=0) [self.seq_lengthx1, H, W]
    
    def _get_imagematte(self):
        img_file = random.choice(self.imagematte_files)
        with Image.open(os.path.join(self.imagematte_dir, 'fgr', img_file)) as fgr, \
             Image.open(os.path.join(self.imagematte_dir, 'pha', img_file)) as pha:
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))

        fgrs = [fgr] * self.seq_length
        phas = [pha] * self.seq_length
        return fgrs, phas

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_lr_videomatte(self, idx):
        clip_idx, frame_idx = self.lr_videomatte_idx[idx]
        clip = self.lr_videomatte_clips[clip_idx]
        frame_count = len(self.lr_videomatte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.lr_videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.lr_videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.lr_videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _get_hr_videomatte(self, idx):
        clip_idx, frame_idx = self.hr_videomatte_idx[idx]
        clip = self.hr_videomatte_clips[clip_idx]
        frame_count = len(self.hr_videomatte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.hr_videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.hr_videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.hr_videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.MotionBlur(prob=0.1),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomMotionAffine(prob=0.3),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=768),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=768),
                ])
            ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.vm_path)
    assert root.exists(), f'provided VM path {root} does not exist'
    
    if args.dataset_file == 'vm':
        print('use VideoMatting dataset')
        dataset = VideoMatteDataset(imagematte_dir='/home/jiachenl/data/Matting/ImageMatte/train',
                                    lr_videomatte_dir='/home/jiachenl/data/Matting/VideoMatte240K_JPEG_SD/train',
                                    hr_videomatte_dir='/home/jiachenl/data/Matting/VideoMatte240K_JPEG_HD/train',
                                    background_image_dir='/home/jiachenl/data/Matting/BG20k/train',
                                    background_video_dir='/home/jiachenl/data/Matting/BackgroundVideos/BackgroundVideosTrain/train',
                                    size=512,
                                    seq_length=args.num_frames,
                                    seq_sampler=TrainFrameSampler(),
                                    transform=make_coco_transforms(image_set))

    return dataset
