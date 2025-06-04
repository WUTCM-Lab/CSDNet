# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, train_backbone, train_transformer):
        super().__init__()
        self.backbone = backbone
        
        # transformer encoder, multiscale fusion
        self.transformer = transformer
        if self.transformer is not None:
            hidden_dim = transformer.d_model
            
            self.mulEnh = None
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            hidden_dim = backbone.num_channels

        # train setting
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        
        if self.transformer is not None and not train_transformer:
            freeze_param = [self.transformer]
            freeze_param.append(self.input_proj)
            for m in freeze_param:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, samples, text_feat=None, text_mask=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)  # list, list
        feat, mask = features[-1].decompose()
        pos_embed = pos[-1]
        assert mask is not None
        
        if self.transformer is not None:
            if self.mulEnh is not None:
                pass
            else:
                feat = self.input_proj(feat)
                feat = feat.flatten(2).permute(0,2,1)  # [bs,n,dim]
            mask = mask.flatten(1)  # [bs,n]
            pos_embed = pos_embed.flatten(2).permute(0,2,1)  # [bs,n,dim]
            feat = self.transformer(feat, mask, pos_embed, text_feat, text_mask)  # [bs,n,dim]
            out = [feat.permute(1,0,2), mask, pos_embed.permute(1,0,2)]
        else:
            out = [feat, mask, pos_embed]
        
        return out  # [n,bs,dim]


def build_detr(args):
    backbone = build_backbone(args)
    if args.detr_enc_num > 0:
        transformer = build_transformer(args)
    else:
        transformer = None
    train_backbone = args.lr_visu_cnn > 0
    train_transformer = args.lr_visu_tra > 0
    model = DETR( backbone, transformer,
        train_backbone, train_transformer,
    )
    return model

