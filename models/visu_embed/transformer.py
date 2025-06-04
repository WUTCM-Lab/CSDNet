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

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class transEnh1(nn.Module):
    def __init__(self, d_model=256, t_model=768, keep_clas=True):
        super().__init__()
        self.t_proj1 = nn.Linear(t_model, d_model)
        self.v_proj1 = nn.Linear(d_model, d_model)
        
        self.t_proj2 = nn.Linear(t_model, d_model)
        if keep_clas:
            self.t_proj3 = nn.Linear(d_model*2, d_model)
        else: 
            self.t_proj3 = None
        
        self.hidden = d_model
        
    def forward(self, visu_feat, text_feat, text_mask):     
        text_mask_v = text_mask[:,1:]  # words mask
        attn_mask = torch.ones_like(text_mask_v, dtype=text_feat.dtype, device=text_mask.device)
        attn_mask.masked_fill_(text_mask_v, 0)
        
        # attn                    
        text_feat_p = self.t_proj1(text_feat)  # [bs,nt+1,256]
        text_feat_w = text_feat_p[:,1:]
        visu_feat_p = self.v_proj1(visu_feat)  # [bs,nv,256]
        
        attn = (text_feat_w @ visu_feat_p.transpose(-2,-1))  # [bs,nt,nv]
        # sum
        attn = torch.sum(attn, dim=2)  # bs,nt
        attn_new = attn*attn_mask.float()
        attn_l2_norm = attn_new.pow(2).sum(dim=1).sqrt().unsqueeze(1)  # bs,1
        attn_exp = torch.exp(attn_new/attn_l2_norm)  # bs,nt
        attn_fenmu = torch.sum(attn_exp, dim=1, keepdim=True)  # bs,1
        attn_chufa = attn_exp / attn_fenmu  # bs,nt
        attn_chufa = attn_chufa*attn_mask.float()
        ls = torch.sum(attn_chufa.unsqueeze(2)*text_feat[:,1:], dim=1, keepdim=True)  # bs,1,768
        
        # concat->proj
        if self.t_proj3 is not None:
            text_feat_c = text_feat_p[:,0:1]
            ls = torch.concat([self.t_proj2(ls), text_feat_c], dim=2)  # bs,1,256*2
            lv = torch.sigmoid(self.t_proj3(ls))  # bs,1,256
        else:
            lv = torch.sigmoid(self.t_proj2(ls))  # bs,1,256
        visu_feat1 = visu_feat + visu_feat*lv
                
        return visu_feat1


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, 
                 num_encoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu", 
                 normalize_before=False, 
                 vl_enhancevit=False):
        super().__init__()
        encoder_elem = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        
        if vl_enhancevit:
            enhance_elem = transEnh1()
        else:
            enhance_elem = None
        self.encoder = TransformerEncoder(encoder_elem, enhance_elem, num_encoder_layers, encoder_norm)
        
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, text_feat, text_mask):
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, 
                    text_feat=text_feat, text_mask=text_mask)

        return memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, enhance_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        if enhance_layer is not None:
            self.extras = _get_clones(enhance_layer, num_layers)
        else:
            self.extras = [None, None, None, None, None, None] 
        
        self.norm = norm
        
        self.num_layers = num_layers
        
    def forward(self, src, mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, 
                pos: Optional[Tensor] = None,
                text_feat=None,
                text_mask=None):
        output = src  # [bs,nv,256]
        
        for extra, layer in zip(self.extras, self.layers):
            if extra is not None: 
                output = extra(output, text_feat, text_mask)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.detr_enc_num,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,   
        normalize_before=args.pre_norm,
        vl_enhancevit=args.vl_enhancevit,
    )
    


    