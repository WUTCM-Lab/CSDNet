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
    
    def __init__(self, num_decoder_layers=3,
                 d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        decoder_item = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_item, num_decoder_layers, decoder_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # fine-grained text with pos
    def forward(self, text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos):
        # (bs,dim,in_points)        
        output, attn = self.decoder(
            text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos)
           
        return output, attn  # last layer attention


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, text_feat, visu_feat,
                text_mask: Optional[Tensor] = None, 
                visu_mask: Optional[Tensor] = None,
                visu_pos: Optional[Tensor] = None, 
                text_pos: Optional[Tensor] = None):
      
        attn = None

        for layer in self.layers:
            output, attn = layer(
                text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos)

        if self.norm is not None:
            output = self.norm(output)

        # return output.unsqueeze(0)
        return output, attn  # last layer attention
    

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
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
        # cros attn
        tgt2, attn = self.cros_attn(query=self.with_pos_embed(text_feat, text_pos), 
                                key=self.with_pos_embed(visu_feat, visu_pos), value=visu_feat, 
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
                visu_pos: Optional[Tensor] = None, 
                text_pos: Optional[Tensor] = None):
        
        return self.forward_post(text_feat, visu_feat, text_mask, visu_mask, text_pos, visu_pos)

 
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_vl_transformer2(args):
    return VLTransformer(
        num_decoder_layers=4,
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

