# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import GroundingDataset


def make_transforms(args, split, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    if split == 'train':
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize-32*i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.
    
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4), 
            T.GaussianBlur(aug_blur=args.aug_blur),
            T.RandomHorizontalFlip(), 
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])

    if split in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {split}')


def make_rsvg_transforms(args, split): # keys: dataset, imsize, aug_scale, aug_crop, aug_translate
        
    imsize = args.imsize
        
    if split == 'train':
        scales = []
        if args.aug_scale:
            for i in range(8):
                scales.append(imsize-32*i)
        else:
            scales = [imsize]
    
        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.
        if imsize == 640:  # original
            crop_list = [400, 500, 600]
            cp1, cp2 = (384, 600)
        elif imsize == 800: 
            if args.dataset == 'DIOR_RSVG':
                crop_list = [imsize-50*i for i in range(1,7)]
                cp1, cp2 = (480, crop_list[0])
            elif args.dataset == 'OPT_RSVG':
                crop_list = [imsize-50*i for i in range(1,6)]
                cp1, cp2 = (512, crop_list[0])
        elif imsize == 512:
            crop_list = [500-50*i for i in range(3)]
            cp1, cp2 = (384, crop_list[0])
            
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([ 
                    T.RandomResize(crop_list, resize_long_side=False),
                    T.RandomSizeCrop(cp1,cp2,check_method=dict(func='iou', iou_thres=0.5)),
                    T.RandomResize(scales),
                ]), 
                p=crop_prob),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate),
        ])
        
    if split in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])
        
    raise ValueError(f'unknown {split}')


def build_dataset(split, args):
    if args.dataset in ['DIOR_RSVG', 'OPT_RSVG', 'VRSBench_Ref']:  # DIOR_RSVG,OPT_RSVG,VRSBench_Ref
        transform = make_rsvg_transforms(args, split)
    else:
        transform = make_transforms(args, split)
    if args.model_type == "ResNet":
        return GroundingDataset(
            image_root=args.image_root,
            split_root=args.split_root,
            dataset=args.dataset,
            transform=transform,
            split=split,
            max_query_len=args.max_query_len,
            bert_model=args.bert_model,
            im_size=args.imsize
        )
