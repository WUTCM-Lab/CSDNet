# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch.nn.functional as F
from torch import nn, Tensor


# 视觉特征进行self attn后再利用语言作为查询去解码


class VLTransformer(nn.Module):
    
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        # self.norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # fine-grained text with pos
    def forward(self, visu_feat, visu_mask, text_feat, visu_pos, text_mask, text_pos=None):
        # (bs,dim,in_points)
        visu_feat = visu_feat.permute(2,0,1)
        visu_pos = visu_pos.permute(2,0,1)
        
        attn = None
        visu_feat = self.norm1(visu_feat)
        text_feat = self.norm3(text_feat)
        output, attn = self.decoder(
            text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos)

        # if self.norm2 is not None:
        #     output = self.norm2(output)
           
        return output, attn  # last layer attention
    

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cros_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, text_feat, visu_feat,
                     text_mask: Optional[Tensor] = None, 
                     visu_mask: Optional[Tensor] = None,
                     text_pos: Optional[Tensor] = None,
                     visu_pos: Optional[Tensor] = None):
    
        memory = visu_feat
        # cros attn
        tgt2, attn = self.cros_attn(query=self.with_pos_embed(text_feat, text_pos), 
                                key=self.with_pos_embed(memory, visu_pos), value=memory, 
                                attn_mask=None, key_padding_mask=visu_mask)
        tgt = text_feat + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

    def forward(self, text_feat, visu_feat,
                text_mask: Optional[Tensor] = None, 
                visu_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visu_pos: Optional[Tensor] = None):
        
        return self.forward_post(text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos)


def build_vl_transformer1(args):
    return VLTransformer(
        d_model=args.vl_hidden_dim,
        nhead=args.vl_nheads,
        dim_feedforward=args.vl_dim_feedforward,
        dropout=args.vl_dropout,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

