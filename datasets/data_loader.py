# -*- coding: utf-8 -*-

import re
import sys
import torch
import numpy as np
import os.path as osp
import torch.utils
import torch.utils.data as data
import torch.utils.data
sys.path.append('.')
from PIL import Image
from transformers import BertTokenizer
from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line 
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):  # A single set of features of data.
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class GroundingDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
        # remote sensing
        'DIOR_RSVG': {'splits': ('train', 'val', 'test')},
        'OPT_RSVG': {'splits': ('train', 'val', 'test')},
        'VRSBench_Ref': {'splits': ('train', 'val')},
    }

    def __init__(self, image_root, split_root='data', dataset='referit', transform=None, return_idx=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='./checkpoints/bert-base-uncased', 
                 im_size=640):
        self.images = []
        self.image_root = image_root  # File paths for images and annotations
        self.split_root = split_root  # File path for image partitions
        self.dataset = dataset
        
        self.transform = transform
        self.return_idx=return_idx
        self.split = split
        
        self.im_size = im_size
        self.txt_len = max_query_len  
        self.lstm = lstm
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        assert self.transform is not None
        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.image_root, 'referit')
            self.image_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.image_root, 'Flickr30k')
            self.image_dir = osp.join(self.dataset_root, 'flickr30k_images')
        elif self.dataset in ['unc', 'unc+', 'gref', 'gref_umd']:  ## refcoco, etc.
            self.dataset_root = osp.join(self.image_root, 'other')
            self.image_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
        elif self.dataset == 'DIOR_RSVG': 
            self.dataset_root = osp.join(self.image_root, 'DIOR_RSVG')
            self.image_dir = osp.join(self.dataset_root, 'JPEGImages')
        elif self.dataset == 'OPT_RSVG': 
            self.dataset_root = osp.join(self.image_root, 'OPT_RSVG')
            self.image_dir = osp.join(self.dataset_root, 'Image')
        elif self.dataset == 'VRSBench_Ref': 
            self.dataset_root = osp.join(self.image_root, 'VRSBench_Ref')
            if split == 'train':
                self.image_dir = osp.join(self.dataset_root, 'Images_train')
            else:
                self.image_dir = osp.join(self.dataset_root, 'Images_val')

        self.RESMOTE_DATASETS = ['DIOR_RSVG', 'OPT_RSVG', 'VRSBench_Ref']
        
        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']
        
        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if (self.dataset in self.RESMOTE_DATASETS) or (self.dataset in ['flickr']):
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        # image
        img_path = osp.join(self.image_dir, img_file)
        img_pilw = Image.open(img_path).convert('RGB')
        
        # box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=np.float32)
        if self.dataset in ['unc', 'unc+', 'gref', 'gref_umd']:
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        bbox = torch.tensor(bbox)
        
        # phrase
        phrase = phrase.lower()
        if phrase.endswith('.'):  
            phrase = phrase[:-1]
        
        return img_pilw, bbox, phrase

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.txt_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, bbox, phrase = self.pull_item(idx)
        input_dict = {
            'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        image = input_dict['img']
        imask = input_dict['mask']
        phrase= input_dict['text']
        bboxs = input_dict['box']
        
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.txt_len, tokenizer=self.tokenizer)
            word_id   = features[0].input_ids
            word_mask = features[0].input_mask
        
        return image, np.array(imask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bboxs, dtype=np.float32)
