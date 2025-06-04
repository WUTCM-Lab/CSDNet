# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_transformer


class multiFusion(nn.Module):
    def __init__(self, d_model=256, dims=[512, 1024, 2048]):
        super().__init__()
        self.conv1 = nn.Conv2d(dims[0], d_model, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(d_model+dims[1], d_model, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(d_model+dims[2], d_model, kernel_size=1)
        
        self.cro_attn1 = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        self.cro_attn2 = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        
        self.t_proj1 = nn.Linear(768, d_model)
        self.t_proj2 = nn.Linear(768, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, features, text_feat, text_mask):
        src1, src2, src3, src4 = features
        feat1, _ = src1.decompose()  # 512, 1024, 2048
        feat2, _ = src2.decompose()
        feat3, _ = src3.decompose()
        feat4, _ = src4.decompose()
        # fusion
        feat1 = self.conv1(feat1)
        feat2 = self.conv2(torch.cat([feat1, feat2], dim=1))
        text_feat1 = self.t_proj1(text_feat)
        
        bs, _, h, w = feat2.shape
        feat2 = feat2.flatten(2).permute(0,2,1)  # torch.Size([60, 400, 256])
        feat2 = self.norm1(self.cro_attn1(query=feat2, key=text_feat1, value=text_feat1, key_padding_mask=text_mask)[0] + feat2)
        feat2 = feat2.permute(0,2,1).reshape(bs, _, h,w)
        
        feat3 = self.conv3(torch.cat([feat2, feat3], dim=1))
        text_feat2 = self.t_proj2(text_feat)
        feat3 = feat3.flatten(2).permute(0,2,1)  # bs,n,dim
        feat3 = self.norm2(self.cro_attn1(query=feat3, key=text_feat2, value=text_feat2, key_padding_mask=text_mask)[0] + feat3)
    
        return feat3  # [bs,n,dim]


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, train_backbone, train_transformer, vl_multiscale):
        super().__init__()
        self.backbone = backbone
        
        # transformer encoder, multiscale fusion
        self.transformer = transformer
        if self.transformer is not None:
            hidden_dim = transformer.d_model
            if vl_multiscale:  # 是否考虑多尺度特征
                self.mulEnh = multiFusion(d_model=hidden_dim)
            else:
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
                feat = self.mulEnh(features, text_feat, text_mask)  # [bs,n,dim]
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
        vl_multiscale=args.vl_multiscale,
    )
    return model

