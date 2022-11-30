# ------------------------------------------------------------------------
# VMFormer Infernce 
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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import datasets
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import sys
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--version', default='v1', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone', default='mv3', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'temporal'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_out_stride', default=4, type=int)
    parser.add_argument('--query_temporal', type=str, default=None,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--fpn_temporal', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--l1_loss_coef', default=1, type=float)
    parser.add_argument('--lap_loss_coef', default=1, type=float)
    parser.add_argument('--temporal_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--img_path', default='../data/Matting/videomatte_512x288/')
    parser.add_argument('--dataset_file', default='vm')
    parser.add_argument('--model', default='vm')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_frames', default=20, type=int, help='number of frames')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    mad = MetricMAD()
    mse = MetricMSE()
    grad = MetricGRAD()
    conn = MetricCONN()

    pha_mads = []
    pha_mses = []
    pha_grads = []
    pha_conns = []

    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        model.eval()

        root_folder = args.img_path
        clip_paths = []

        for dataset in sorted(os.listdir(root_folder)):
            if os.path.isdir(os.path.join(root_folder, dataset)):
                dataset_path = os.path.join(root_folder, dataset)
                for clip in sorted(os.listdir(dataset_path)):
                    clip_path = os.path.join(dataset_path, clip)
                    clip_paths.append(clip_path)

        for single_clip_path in clip_paths:
            pha_mad = []
            pha_mse = []
            pha_grad = []
            pha_conn = []
            pha_dtssd = []
            print('processing %s\n'%(single_clip_path))
            files = []
            single_clip_path_com = os.path.join(single_clip_path,'com')
            for single in sorted(os.listdir(single_clip_path_com)):
                files.append(os.path.join(single_clip_path_com, single))
            img_set = []
            img_index_set = []

            for k in range(len(files)):
                im = Image.open(files[k])
                w, h = im.size
                sizes = torch.as_tensor([int(h), int(w)])
                img_set.append(transform(im).unsqueeze(0).cuda())
                img_index_set.append(k)
                if (k+1) % args.num_frames == 0:
                    img = torch.cat(img_set,0)
                    model.num_frames=img.shape[0]
                    #### img.shape [5, 3, 288, 512]
                    outputs = model.inference(img, img.shape[-1], img.shape[-2])
                    for (j,mask) in enumerate(outputs):
                        mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False)
                        pred_pha = mask[0][0].sigmoid().cpu().detach().numpy().astype(np.float32)
                        #### to make fair comparisons to RVM's evaluation script
                        pred_pha = pred_pha * 255
                        pred_pha = pred_pha.astype(np.uint8)
                        pred_pha = pred_pha / 255
                        #### 
                        pha_gt_file = files[img_index_set[j]].replace('com','pha')
                        true_pha = cv2.imread(pha_gt_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                        pha_mad.append(mad(pred_pha, true_pha))
                        pha_mse.append(mse(pred_pha, true_pha))
                        pha_grad.append(grad(pred_pha, true_pha))
                        pha_conn.append(conn(pred_pha, true_pha))
                    img_set = []
                    img_index_set = []
            
            print('pha_mad:%.2f\n'%(np.mean(pha_mad)))
            print('pha_mse:%.2f\n'%(np.mean(pha_mse)))
            print('pha_grad:%.2f\n'%(np.mean(pha_grad)))
            print('pha_conn:%.2f\n'%(np.mean(pha_conn)))
            pha_mads.append(np.mean(pha_mad))
            pha_mses.append(np.mean(pha_mse))
            pha_grads.append(np.mean(pha_grad))
            pha_conns.append(np.mean(pha_conn))
        
        print('pha_mad:%.2f\n'%(np.mean(pha_mads)))
        print('pha_mse:%.2f\n'%(np.mean(pha_mses)))
        print('pha_grad:%.2f\n'%(np.mean(pha_grads)))
        print('pha_conn:%.2f\n'%(np.mean(pha_conns)))

class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3

class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3

class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
    
    def __call__(self, pred, true):
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2


class MetricCONN:
    def __call__(self, pred, true):
        step=0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        connectivity_error = np.sum(np.abs(true_phi - pred_phi))
        return connectivity_error / 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VMFormer inference script on RVM test set', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
