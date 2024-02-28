from datasets import load_dataset

import configparser
import json
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import math
import json
with open('C:\\Users\\84898\\Desktop\\project\\WIP\\Machine Translation\\src\\config.json') as f:
    config = json.load(f)
def prepare_data():
    data = load_dataset(config['DATASET1'], config['DATASET2'])
    SRC_LANGUAGE = config['SRC_LANGUAGE']
    TGT_LANGUAGE = config['TGT_LANGUAGE']
    token_transform = {}
    vocab_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer(config['TOKENIZER'])
    token_transform[TGT_LANGUAGE] = get_tokenizer(config['TOKENIZER'])
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    def yield_tokens(data_iter, language):
        for data_sample in data_iter['translation']:
            yield token_transform[language](data_sample[language])
            
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator 
        vocab_transform[language] = build_vocab_from_iterator(yield_tokens(data['train'], language),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)
        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        vocab_transform[language].set_default_index(UNK_IDX)
    
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func
    
    def tensor_transform(token_ids: list):
        return torch.cat((torch.tensor([BOS_IDX]), 
                        torch.tensor(token_ids), 
                        torch.tensor([EOS_IDX])))
        
    text_transforms = {}
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transforms[language] = sequential_transforms(token_transform[language], #Tokenization
                                                        vocab_transform[language], #Numericalization
                                                        tensor_transform) # Add BOS/EOS and create tensor
    
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transforms[SRC_LANGUAGE](src_sample).to(dtype=torch.int64))
            tgt_batch.append(text_transforms[TGT_LANGUAGE](tgt_sample).to(dtype=torch.int64))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
    
    BATCH_SIZE = config['BATCH_SIZE']
    train_dataloader = DataLoader(data['train']['translation'], batch_size=BATCH_SIZE, 
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(data['validation']['translation'], batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(data['test']['translation'], batch_size=BATCH_SIZE,
                                 collate_fn=collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader, vocab_transform[SRC_LANGUAGE], vocab_transform[TGT_LANGUAGE], text_transforms[SRC_LANGUAGE], text_transforms[TGT_LANGUAGE]