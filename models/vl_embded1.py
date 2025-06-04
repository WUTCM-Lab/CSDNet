import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# Context-enhanced Interaction


class VisionLanguageEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", 
                num_visu_token=400):
        super().__init__()        
        self.blocks = crosAttn(
            d_model, nhead, dim_feedforward, dropout, activation)     
        
        self.num_visu_token = num_visu_token
        self.d_model = d_model
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vl_feat, vl_mask, vl_pos):
        visu_feat = vl_feat[:self.num_visu_token]  # [H*W,B,channel]
        text_feat = vl_feat[self.num_visu_token:]  # [max_len,B,channel]
        visu_mask = vl_mask[:,:self.num_visu_token]
        text_mask = vl_mask[:,self.num_visu_token:]
        visu_pos = vl_pos[:self.num_visu_token]  # [H*W,B,channel]
        text_pos = vl_pos[self.num_visu_token:]  # [max_len,B,channel]
        
        # cros attn
        v_feat1, t_feat1 = self.blocks(
            visu_feat, text_feat, visu_mask, text_mask, visu_pos, text_pos)  # v_feat_t, t_feat_v
      
        return torch.concat([v_feat1,t_feat1], dim=0)


class crosAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.visu2text = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)      
        
        self.text2visu = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.visu_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)        
        
        self.text_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.dropout8 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout5 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)
        self.dropout6 = nn.Dropout(dropout)
        self.norm6 = nn.LayerNorm(d_model)
        
        self.norm7 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, v_feat, t_feat, 
                    v_mask, t_mask, 
                    v_pos, t_pos):
        src2 = self.visu2text(
            self.with_pos_embed(v_feat, v_pos), 
            self.with_pos_embed(t_feat, t_pos), value=t_feat, 
            key_padding_mask=t_mask)[0]        
        v_feat1 = self.norm1(v_feat + src2)
        
        src2 = self.visu_attn(
            self.with_pos_embed(v_feat1, v_pos), 
            self.with_pos_embed(v_feat1, v_pos), value=v_feat, 
            key_padding_mask=v_mask)[0]        
        v_feat3 = self.norm3(src2)+self.norm5(v_feat)
        
        t_feat1 = self.norm7(t_feat)
        src2 = self.text2visu(
            self.with_pos_embed(t_feat1, t_pos), 
            self.with_pos_embed(v_feat3, v_pos), value=v_feat3, 
            key_padding_mask=v_mask)[0]        
        t_feat2 = self.norm2(t_feat1 + self.dropout2(src2))
        
        src2 = self.text_attn(
            self.with_pos_embed(t_feat2, t_pos), 
            self.with_pos_embed(t_feat2, t_pos), value=t_feat1, 
            key_padding_mask=t_mask)[0]        
        t_feat3 = self.norm4(t_feat1 + self.dropout8(src2))
        t_feat4 = self.linear4(self.dropout5(self.activation(self.linear3(t_feat3))))
        t_feat4 = self.norm6(t_feat3 + self.dropout6(t_feat4))
       
        return v_feat3, t_feat4
    

def build_vl_encoder1(args):
    if args.vl_crosAttn:
        divisor = 16 if args.dilation else 32
        num_visu_token = int((args.imsize / divisor) ** 2)
        return VisionLanguageEncoder(
            d_model=args.vl_hidden_dim, 
            nhead=args.vl_nheads,
            dim_feedforward=args.vl_dim_feedforward,
            dropout=args.vl_dropout,
            num_visu_token=num_visu_token
        )
    else:
        return None


def _get_activation_fn(activation):
    # Return an activation function given a string
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")