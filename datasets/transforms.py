import random
import numpy as np
from PIL import ImageEnhance, ImageFilter

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_utils import xyxy2xywh, box_iou


def crop(image, box, region):
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    return cropped_image, cropped_box


def resize_according_to_side(img, box, size):
    h, w = img.height, img.width
    ratio_h = float(size / float(h))
    ratio_w = float(size / float(w))
    new_w, new_h = round(w* ratio_w), round(h * ratio_h)
    img = F.resize(img, (new_h, new_w))
    box[0] = box[0] * ratio_w
    box[1] = box[1] * ratio_h
    box[2] = box[2] * ratio_w
    box[3] = box[3] * ratio_h
    
    return img, box


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img
        

class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img
        

class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation
    
    def __call__(self, img):
        saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast   = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, input_dict):
        if random.random() < 0.8:
            image = input_dict['img']
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    image = self.rand_brightness(image)
                elif func_id == 1:
                    image = self.rand_contrast(image)
                elif func_id == 2:
                    image = self.rand_saturation(image)
            input_dict['img'] = image

        return input_dict


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.
    
    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img

        return input_dict


class RandomHorizontalFlip(object):
    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict['img']
            box = input_dict['box']
            text = input_dict['text']

            img = F.hflip(img)
            text = text.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            h, w = img.height, img.width
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

            input_dict['img'] = img
            input_dict['box'] = box
            input_dict['text'] = text

        return input_dict


class RandomResize(object):
    def __init__(self, sizes, resize_long_side=True):
        self.sizes = sizes
        self.resize_long_side = resize_long_side
        if resize_long_side:
            self.choose_size = max
        else:
            self.choose_size = min

    def __call__(self, input_dict):
        img = input_dict['img']
        box = input_dict['box']
        
        size = random.choice(self.sizes)
        h, w = img.height, img.width
        ratio = float(size) / self.choose_size(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        resized_img = F.resize(img, (new_h, new_w))

        ratio_h, ratio_w = float(new_h) / h, float(new_w) / w
        resized_box = box * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])

        input_dict['img'] = resized_img
        input_dict['box'] = resized_box
        return input_dict


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int=20, check_method={}):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try  = max_try
        
        if check_method.get('func', 'area') == 'area':
            self.check = self.check_area
            self.area_thres = check_method.get('area_thres', 0)
        elif check_method.get('func') == 'iou':
            self.check = self.check_iou
            self.iou_thres = check_method.get('iou_thres', 0.5)
        else: raise NotImplementedError
    
    def check_iou(self, cropped_box, orig_box):
        iou = box_iou(cropped_box.view(-1, 4), orig_box.view(-1, 4))[0]
        return (iou >= self.iou_thres).all()

    def check_area(self, cropped_box, orig_box):
        cropped_box = cropped_box.reshape(-1, 2, 2)
        box_hw = cropped_box[:, 1, :] - cropped_box[:, 0, :]
        return (box_hw > 0).all() and (box_hw.prod(dim=1) > self.area_thres).all()
    
    def __call__(self, input_dict):
        img = input_dict['img']
        orig_box = input_dict['box']

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w]) # [i, j, target_w, target_h]
            i, j, th, tw = region
            cropped_box = orig_box - torch.as_tensor([j, i, j, i])
            cropped_box = torch.min(cropped_box.reshape(-1, 2, 2), torch.as_tensor([tw, th], dtype=torch.float32))
            cropped_box = cropped_box.clamp(min=0).reshape(-1) + torch.as_tensor([j, i, j, i])
            if self.check(cropped_box, orig_box):
                img, box = crop(img, orig_box, region)
                input_dict['img'] = img
                input_dict['box'] = box
                return input_dict

        return input_dict


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    
    def __call__(self, input_dict):
        text = input_dict['text']
        
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        for wd in dir_words:
            if wd in text:
                return self.transforms1(input_dict)

        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ToTensor(object):
    def __call__(self, input_dict):
        img = input_dict['img']
        # img = img.transpose((2,0,1))
        # img = torch.from_numpy(img).float()
        img = F.to_tensor(img)
        input_dict['img'] = img
        
        return input_dict


class NormalizeAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate
    
    def __call__(self, input_dict):
        img = input_dict['img']  # 3,800,800
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w  # -160
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)  # -80
            left = round(dw / 2.0 - 0.1)

        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_img[:, top:top+h, left:left+w] = img
        out_mask[top:top+h, left:left+w] = 0  # -80:640

        input_dict['img'] = out_img
        input_dict['mask']= out_mask

        if 'box' in input_dict.keys():
            box = input_dict['box']
            box[0], box[2] = box[0]+left, box[2]+left
            box[1], box[3] = box[1]+top, box[3]+top
            h, w = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([w, h, w, h], dtype=torch.float32)
            input_dict['box'] = box

        return input_dict
