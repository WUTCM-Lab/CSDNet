import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from .visu_embed.detr import build_detr
from .text_embed.bert import build_bert
from .vl_transformer1 import build_vl_transformer1
# from .vl_transformer2 import build_vl_transformer2
from .vl_embded1 import build_vl_encoder1 as build_vl_encoder


class CSDNet(nn.Module):
    def __init__(self, args):
        super(CSDNet, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.text_pos = nn.Embedding(self.num_text_token, hidden_dim)
        if args.vl_crosAttn:
            self.vl_encoder = build_vl_encoder(args)
        else:
            self.vl_encoder = None
                
        # decoding        
        self.st_dec_dyn = args.st_dec_dyn  
        self.vl_dec_num = args.vl_dec_num    
        if self.st_dec_dyn:    
            self.uniform_learnable = args.uniform_learnable
            self.uniform_grid = args.uniform_grid
            # sampling relevant
            self.visual_feature_map_h = int(args.imsize / divisor)
            self.visual_feature_map_w = int(args.imsize / divisor)
            self.in_points = args.in_points
            self.offset_generators = nn.ModuleList([nn.Linear(hidden_dim, self.in_points*2) for _ in range(self.vl_dec_num)])
            self.update_sampling_queries = nn.ModuleList(
                [MLP(2 * hidden_dim, hidden_dim, hidden_dim, 2) for _ in range(self.vl_dec_num)])
            # init
            self.init_sampling_feature= nn.Embedding(1, hidden_dim)
            self.init_referent_points = nn.Embedding(1, 2)
            self.init_weights()
            # load text
            self.in_load = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
            self.vl_transformer = nn.ModuleList(
                [build_vl_transformer1(args) for _ in range(self.vl_dec_num)])  
            
            if self.uniform_grid:  # grid sampling
                h = int(math.sqrt(self.in_points))
                w = h
                step = 1 / h
                start = 1 / h / 2

                new_h = torch.tensor([start + i * step for i in range(h)]).view(-1, 1).repeat(1, w)
                new_w = torch.tensor([start + j * step for j in range(w)]).repeat(h, 1)
                grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
                grid = grid.view(-1, 2)  # (in_points,2)
                self.initial_sampled_points = torch.nn.Parameter(grid.unsqueeze(0))  # (1,in_points,2)
            
        else:
            # compare 
            # self.vl_transformer = build_vl_transformer2(args)  
            pass
        
        # detection head
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def init_weights(self):
        nn.init.constant_(self.init_referent_points.weight[:, 0], 0.5)
        nn.init.constant_(self.init_referent_points.weight[:, 1], 0.5)
        self.init_referent_points.weight.requires_grad=False

        for i in range(self.vl_dec_num):
            nn.init.zeros_(self.offset_generators[i].weight)
            nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
        if not self.uniform_learnable:
            self.offset_generators[0].weight.requires_grad = False
            self.offset_generators[0].bias.requires_grad = False

    def feautures_sampling(self, sampling_query, referent_point, feature_map, pos, stage):
        bs, channel = sampling_query.shape
        if self.uniform_grid:
            if stage != 0:
                xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
                sampled_points = (xy_offsets.permute(1, 0, 2) + referent_point).permute(1, 0, 2)  # (bs,in_points,2)
            else:
                sampled_points = self.initial_sampled_points.clone().repeat(bs, 1, 1)  # At the beginning, the grid sampling is done in the whole picture.
        else:
            xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
            sampled_points = (xy_offsets.permute(1, 0, 2) + referent_point).permute(1, 0, 2)  # (bs,in_points,2)
        feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs,dim,h,w)
        pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w)  # (bs,dim,h,w)

        # [0,1] to [-1,1]
        sampled_points = (2 * sampled_points) - 1

        sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
                                       align_corners=False).squeeze(-1)  # (bs,dim,in_points)
        pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs,dim,in_points)

        return sampled_features, pe

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        
        # language bert
        text_src = self.textmodel(text_data)
        text_feat, text_mask = text_src.decompose()
        assert text_mask is not None
        text_mask = text_mask.flatten(1)  # (B,L)
        
        # language-guided visual backbone
        visu_feat, visu_mask, v_pos = self.visumodel(img_data, text_feat, text_mask)  # out: [N,bs,dim]
        
        # visual and text
        ## feat_proj
        visu_feat = self.visu_proj(visu_feat)  # (N,bs,dim)
        text_feat = self.text_proj(text_feat).permute(1,0,2)  # (L,bs,dim)
        ## concat 
        vl_feat = torch.cat([visu_feat, text_feat], dim=0)  # [N+L,bs,dim]
        vl_mask = torch.cat([visu_mask, text_mask], dim=1)  # [bs,N+L]
        ## l_pos
        l_pos = self.text_pos.weight.unsqueeze(1).repeat(1,bs,1)
        ## interaction
        if self.vl_encoder is not None:
            vl_feat = self.vl_encoder(vl_feat, vl_mask, torch.cat([v_pos, l_pos], dim=0))  # (L+N)xBxC
        else:
            vl_feat = vl_feat
        
        # split
        visu_feat = vl_feat[:self.num_visu_token]  # (H*W,B,dim)
        text_feat = vl_feat[self.num_visu_token:]  # (max_len,B,dim)
                    
        # dynamic decoding
        pred_box = None
        if self.st_dec_dyn:
            sampling_init = self.init_sampling_feature.weight.repeat(bs,1)  # [bs,dim]
            # load textual information
            sampling_query = self.in_load(sampling_init.unsqueeze(0), text_feat, value=text_feat,
                                  key_padding_mask=text_mask)[0]
            sampling_query = sampling_query.squeeze(0)
            referent_point = self.init_referent_points.weight.repeat(bs,1)
            
            for i in range(0, self.vl_dec_num):
                # 2D adaptive sampling
                sampled_features, pe = self.feautures_sampling(
                    sampling_query, referent_point, visu_feat.permute(1,2,0), v_pos.permute(1,2,0), i)
                
                vg_hs = self.vl_transformer[i](
                    sampled_features, None, text_feat, pe, text_mask, l_pos)
                
                # prediction head
                text_feat, _ = vg_hs  # output, attn
                text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs,n,1)
                text_select_num = text_select.sum(dim=1)  # (bs,1)                
                # new language queries
                vg_hs = (text_select * text_feat.permute(1,0,2)).sum(dim=1) / text_select_num  # (bs,dim)
                
                pred_box = self.bbox_embed(vg_hs).sigmoid()
                # update reference point and sampling query
                referent_point = pred_box[:, :2]
                sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))            
        else:
            pass

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncodingSine(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 20):
        super(PositionalEncodingSine, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding

    def forward(self, token_embedding):
        return self.pos_embedding[:token_embedding.size(0), :]
