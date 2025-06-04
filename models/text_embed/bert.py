# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
from torch import nn

from utils.misc import NestedTensor

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel, BertConfig


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if 'bert-base-uncased' in name:
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num
        
        bert_config = BertConfig.from_pretrained(name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(name, config=bert_config)
        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)[2]
            xs = torch.stack(all_encoder_layers[-3:], 1).mean(1)
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert
