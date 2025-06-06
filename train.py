import argparse
import datetime
import json
import random
import time
import math
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from utils.config import Config
from engine import train_one_epoch, validate
from datasets import build_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('RSVG1', add_help=False)
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='CSDNet',
                        help="Name of model to be exploited.")
    parser.add_argument('--model_type', type=str, default='ResNet', choices=('ResNet', ),
                        help="Name of model to be exploited.")
    # training
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)    
    parser.add_argument('--lr_scheduler', default='step', type=str, help ='step/multistep')
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # DETR 
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), 
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--pre_norm', action='store_true')

    # vl
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_dropout', default=0., type=float,
                        help="Dropout applied in the vision-language transformer")
    
    # Model architecture
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)
    
    # new
    parser.add_argument('--last_dwsample', default=False, type=bool, help='Downsample the features of the last layer output of resnet')
    parser.add_argument('--last_channels', default=256, type=int)
    parser.add_argument('--vl_enhancevit', default=False, type=bool)
    parser.add_argument('--vl_crosAttn', default=False, type=bool)
    parser.add_argument('--vl_enc_num', default=3, type=int)
    
    # dynamic new    
    parser.add_argument('--st_dec_dyn', default=False, type=bool)
    parser.add_argument('--vl_dec_num', default=3, type=int)
    parser.add_argument('--uniform_learnable', default=False, type=bool)
    parser.add_argument('--uniform_grid', default=False, type=bool)
    parser.add_argument('--in_points', default=32, type=int)
    
    # datasets
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/unc/unc+/gref/gref_umd/DIOR_RSVG/OPT_RSVG/VRSBench_Ref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--imsize', default=640, type=int, help='image size')
    # augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    # root
    parser.add_argument('--image_root', type=str, default='/datasets/rec_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='/datasets/rec_data/data/',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default=None, type=str, help='detr model')
    parser.add_argument('--bert_model', default='./checkpoints/bert-base-uncased', type=str, help='bert model')
    
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)  
    
    # distributed training parameters
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Configure file
    parser.add_argument('--config', type=str, help='Path to the configure file.')
    parser.add_argument('--model_config')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.model_type == "ResNet":
        visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
        visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("transformer" in n) \
            and ("extras" not in n) and p.requires_grad)]
        text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
        
        rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]
        rest_param.extend([p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("transformer" in n) \
            and ("extras" in n) and p.requires_grad)])
        rest_param.extend([p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) \
            and ("backbone" not in n) and ("transformer" not in n) and p.requires_grad)])
        
        param_list = [{"params": rest_param},
                       {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                       {"params": visu_tra_param, "lr": args.lr_visu_tra},
                       {"params": text_tra_param, "lr": args.lr_bert},
                       ]
    else:
        pass
    
    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')


    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)
    else:
        raise ValueError('Lr scheduler type not supportted.')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val   = build_dataset('val', args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    if args.model_type == "ResNet":
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    else:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn_clip, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn_clip, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    elif args.detr_model is not None:
        checkpoint = torch.load(args.detr_model, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys when loading detr model:')
        print(missing_keys)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
         
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu = val_stats['accu']

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)