# ------------------------------------------------------------------------
# Transforms and data augmentation for sequence level images, bboxes and masks.
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

import random
import easing_functions as ef

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh, box_iou
from util.misc import interpolate
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2

def random_easing_fn():
    if random.random() < 0.2:
        return ef.LinearInOut()
    else:
        return random.choice([
            ef.BackEaseIn,
            ef.BackEaseOut,
            ef.BackEaseInOut,
            ef.BounceEaseIn,
            ef.BounceEaseOut,
            ef.BounceEaseInOut,
            ef.CircularEaseIn,
            ef.CircularEaseOut,
            ef.CircularEaseInOut,
            ef.CubicEaseIn,
            ef.CubicEaseOut,
            ef.CubicEaseInOut,
            ef.ExponentialEaseIn,
            ef.ExponentialEaseOut,
            ef.ExponentialEaseInOut,
            ef.ElasticEaseIn,
            ef.ElasticEaseOut,
            ef.ElasticEaseInOut,
            ef.QuadEaseIn,
            ef.QuadEaseOut,
            ef.QuadEaseInOut,
            ef.QuarticEaseIn,
            ef.QuarticEaseOut,
            ef.QuarticEaseInOut,
            ef.QuinticEaseIn,
            ef.QuinticEaseOut,
            ef.QuinticEaseInOut,
            ef.SineEaseIn,
            ef.SineEaseOut,
            ef.SineEaseInOut,
            Step,
        ])()

class Step:
    def __call__(self, value):
        return 0 if value < 0.5 else 1

def lerp(a, b, percentage):
    return a * (1 - percentage) + b * percentage

def crop(fgrs, phas, bgrs, region):
    cropped_fgrs = []
    cropped_phas = []
    cropped_bgrs = []
    for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
        cropped_fgrs.append(F.crop(fgr, *region))
        cropped_phas.append(F.crop(pha, *region))
        cropped_bgrs.append(F.crop(bgr, *region))
    return cropped_fgrs, cropped_phas, cropped_bgrs


def hflip(fgrs, phas, bgrs):
    flipped_fgrs = []
    flipped_phas = []
    flipped_bgrs = []

    for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
        flipped_fgrs.append(F.hflip(fgr))
        flipped_phas.append(F.hflip(pha))
        flipped_bgrs.append(F.hflip(bgr))

    return flipped_fgrs, flipped_phas, flipped_bgrs

def vflip(fgrs, phas, bgrs):
    flipped_fgrs = []
    flipped_phas = []
    flipped_bgrs = []

    for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
        flipped_fgrs.append(F.vflip(fgr))
        flipped_phas.append(F.vflip(pha))
        flipped_bgrs.append(F.vflip(bgr))

    return flipped_fgrs, flipped_phas, flipped_bgrs

def resize(fgrs, phas, bgrs, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(fgrs[0].size, size, max_size)
    rescaled_fgrs = []
    rescaled_phas = []
    rescaled_bgrs = []
    for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
        rescaled_fgrs.append(F.resize(fgr, size))
        rescaled_phas.append(F.resize(pha, size))
        rescaled_bgrs.append(F.resize(bgr, size))

    return rescaled_fgrs, rescaled_phas, rescaled_bgrs


def pad(clip, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = []
    for image in clip:
        padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[0].size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, fgrs, phas, bgrs):
        w = random.randint(self.min_size, min(fgrs[0].width, self.max_size))
        h = random.randint(self.min_size, min(fgrs[0].height, self.max_size))
        region = T.RandomCrop.get_params(fgrs[0], [h, w])
        return crop(fgrs, phas, bgrs, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, fgr, pha, bgr):
        
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            fgr *= alpha
            bgr *= alpha
        return fgr, pha, bgr

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, fgr, pha, bgr):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            fgr += delta
            bgr += delta
        return fgr, pha, bgr

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, fgr, pha, bgr):
        if rand.randint(2):
            fgr[:, :, 1] *= rand.uniform(self.lower, self.upper)
            bgr[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return fgr, pha, bgr

class RandomHue(object): #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, fgr, pha, bgr):
        if rand.randint(2):
            fgr[:, :, 0] += rand.uniform(-self.delta, self.delta)
            fgr[:, :, 0][fgr[:, :, 0] > 360.0] -= 360.0
            fgr[:, :, 0][fgr[:, :, 0] < 0.0] += 360.0
            bgr[:, :, 0] += rand.uniform(-self.delta, self.delta)
            bgr[:, :, 0][bgr[:, :, 0] > 360.0] -= 360.0
            bgr[:, :, 0][bgr[:, :, 0] < 0.0] += 360.0
        return fgr, pha, bgr

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, fgr, pha, bgr):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            fgr, pha, bgr = shuffle(fgr, pha, bgr)
        return fgr, pha, bgr

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, fgr, pha, bgr):
        if self.current == 'BGR' and self.transform == 'HSV':
            fgr = cv2.cvtColor(fgr, cv2.COLOR_BGR2HSV)
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            fgr = cv2.cvtColor(fgr, cv2.COLOR_HSV2BGR)
            bgr = cv2.cvtColor(bgr, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return fgr, pha, bgr

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, fgr, pha, bgr):
        fgr = fgr[:, :, self.swaps]
        bgr = bgr[:, :, self.swaps]
        return fgr, pha, bgr

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    
    def __call__(self, fgrs, phas, bgrs):
        fgrs_new = []
        phas_new = []
        bgrs_new = []
        for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
            fgr = np.asarray(fgr).astype('float32')
            pha = np.asarray(pha).astype('float32')
            bgr = np.asarray(bgr).astype('float32')
            fgr, pha, bgr = self.rand_brightness(fgr, pha, bgr)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            fgr, pha, bgr = distort(fgr, pha, bgr)
            fgr, pha, bgr = self.rand_light_noise(fgr, pha, bgr)
            fgrs_new.append(Image.fromarray(fgr.astype('uint8')))
            phas_new.append(Image.fromarray(pha.astype('uint8')))
            bgrs_new.append(Image.fromarray(bgr.astype('uint8')))
        return fgrs_new, phas_new, bgrs_new

#NOTICE: if used for mask, need to change
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, clip, target):
        if rand.randint(2):
            return clip,target
        imgs = []
        masks = []
        image = np.asarray(clip[0]).astype('float32')
        height, width, depth = image.shape
        ratio = rand.uniform(1, 4)
        left = rand.uniform(0, width*ratio - width)
        top = rand.uniform(0, height*ratio - height)
        for i in range(len(clip)):
            image = np.asarray(clip[i]).astype('float32')
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),int(left):int(left + width)] = image
            imgs.append(Image.fromarray(expand_image.astype('uint8')))
            expand_mask = torch.zeros((int(height*ratio), int(width*ratio)),dtype=torch.uint8)
            expand_mask[int(top):int(top + height),int(left):int(left + width)] = target['masks'][i]
            masks.append(expand_mask)
        boxes = target['boxes'].numpy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = torch.tensor(boxes)
        target['masks']=torch.stack(masks)
        return imgs, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, fgrs, phas, bgrs):
        if random.random() < self.p:
            return hflip(fgrs, phas, bgrs)
        return fgrs, phas, bgrs

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, fgrs, phas, bgrs):
        if random.random() < self.p:
            return vflip(fgrs, phas, bgrs)
        return fgrs, phas, bgrs


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, fgrs, phas, bgrs):
        size = random.choice(self.sizes)
        return resize(fgrs, phas, bgrs, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, fgrs, phas, bgrs):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(fgrs, phas, bgrs, (pad_x, pad_y))

class MotionPause(object):
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, *imgs):
        if random.random() < self.prob:
            frames = len(imgs[0])
            pause_frame = random.choice(range(frames - 1))
            pause_length = random.choice(range(frames - pause_frame))
            for img in imgs:
                img[pause_frame + 1 : pause_frame + pause_length] = img[pause_frame]
            return imgs
        else:
            return imgs

class MotionBlur(object):
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, *imgs):
        if random.random() < self.prob:
            blurA = random.random() * 10
            blurB = random.random() * 10

            frames = len(imgs[0])
            easing = random_easing_fn()
            for t in range(frames):
                percentage = easing(t / (frames - 1))
                blur = max(lerp(blurA, blurB, percentage), 0)
                if blur != 0:
                    kernel_size = int(blur * 2)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    for img in imgs:
                        img[t] = F.gaussian_blur(img[t], kernel_size, sigma=blur)
            return imgs
        else:
            return imgs

class RandomMotionAffine(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, *imgs):
        if random.random() < self.prob:
            config = dict(degrees=(-10, 10), translate=(0.1, 0.1),
                      scale_ranges=(0.9, 1.1), shears=(-5, 5), img_size=imgs[0][0].size)
            angleA, (transXA, transYA), scaleA, (shearXA, shearYA) = T.RandomAffine.get_params(**config)
            angleB, (transXB, transYB), scaleB, (shearXB, shearYB) = T.RandomAffine.get_params(**config)
            frames = len(imgs[0])
            easing = random_easing_fn()
            for t in range(frames):
                percentage = easing(t / (frames - 1))
                angle = lerp(angleA, angleB, percentage)
                transX = lerp(transXA, transXB, percentage)
                transY = lerp(transYA, transYB, percentage)
                scale = lerp(scaleA, scaleB, percentage)
                shearX = lerp(shearXA, shearXB, percentage)
                shearY = lerp(shearYA, shearYB, percentage)
                for img in imgs:
                    img[t] = F.affine(img[t], angle, (transX, transY), scale, (shearX, shearY), F.InterpolationMode.BILINEAR)
            return imgs
        else:
            return imgs

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, fgrs, phas, bgrs):
        if random.random() < self.p:
            return self.transforms1(fgrs, phas, bgrs)
        return self.transforms2(fgrs, phas, bgrs)


class ToTensor(object):
    def __call__(self, fgrs, phas, bgrs):
        fgrs_new = []
        phas_new = []
        bgrs_new = []
        for (fgr, pha, bgr) in zip(fgrs, phas, bgrs):
            fgrs_new.append(F.to_tensor(fgr))
            phas_new.append(F.to_tensor(pha))
            bgrs_new.append(F.to_tensor(bgr))
        return fgrs_new, phas_new, bgrs_new


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, fgrs, phas, bgrs):
        return self.eraser(fgrs), self.eraser(phas), self.eraser(bgrs)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, fgrs, phas, bgrs):
        fgrs_new = []
        bgrs_new = []
        for (fgr, bgr) in zip(fgrs, bgrs):
            fgrs_new.append(F.normalize(fgr, mean=self.mean, std=self.std))
            bgrs_new.append(F.normalize(bgr, mean=self.mean, std=self.std))
        return fgrs_new, phas, bgrs_new

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, fgrs, phas, bgrs):
        for t in self.transforms:            
            fgrs, phas, bgrs = t(fgrs, phas, bgrs)
        return fgrs, phas, bgrs

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string