# ------------------------------------------------------------------------
# VMFormer
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

import torch.utils.data

from .vm import build as build_vm

def build_dataset(image_set, args):
    if args.dataset_file == 'vm':
        return build_vm(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


