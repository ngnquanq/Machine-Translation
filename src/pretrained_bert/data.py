import os
import numpy as np
import sacrebleu
import underthesea
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_metric
from transformers import *

# Create dataset
class NMTDataset(Dataset):
    def __init__(self, cfg, data_type = 'train'):
        super().__init__()
        self.cfg = cfg
        self.src_texts, self.tgt_text = self.read_data(data_type)
        self.src_input_ids, self.src_attention_mask = self.texts_to_sequence(self.src_texts)
        self.tgt_input_ids, self.tgt_attention_mask, self.labels = self.texts_to_sequence(self.tgt_text,
                                                                                          is_src = False)

    def read_data(self, data1 = "mt_eng_vietnamese",
                  data2 =  "iwslt2015-en-vi", data_type = 'train'):
        data = load_dataset(data1, data2, split = data_type)
        src_texts = [sample['translation'][self.cfg.src_lang] for sample in data]
        tgt_texts = [sample['translation'][self.cfg.tgt_lang] for sample in data]
        return src_texts, tgt_texts

    def texts_to_sequence(self, texts, is_src = True):
        if is_src:
            src_inputs = self.cfg.src_tokenizer(texts, max_length = self.cfg.src_max_len,
                                                padding = 'max_length', return_tensors = 'pt',
                                                truncation = True)
            return (
                src_inputs.input_ids,
                src_inputs.attention_mask
            )

        else:
            if self.cfg.add_special_tokens == True:
                texts = [
                    ' '.join(
                        [self.cfg.tgt_tokenizer.bos_token, underthesea.word_tokenize(text), self.cfg.tgt_tokenizer.eos_token]
                    ) for text in texts
                ]
            tgt_inputs = self.cfg.tgt_tokenizer(texts, padding = 'max_length', truncation = True,
                                                max_length = self.cfg.tgt_max_len, return_tensors = 'pt')
            labels  = tgt_inputs.input_ids.numpy().tolist()
            labels = [
                [
                    -100 if token_id == self.cfg.tgt_tokenizer.pad_token_id else token_id for token_id in label
                ]
                for label in labels
            ]
            labels = torch.LongTensor(labels)

            return (
                tgt_inputs.input_ids,
                tgt_inputs.attention_mask,
                labels
            )

    def __getitem__(self, index):
        return{
            "input_ids": self.src_input_ids[index],
            "attention_mask": self.src_attention_mask[index],
            "labels": self.labels[index],
            "decoder_input_ids": self.tgt_input_ids[index],
            "decoder_attention_mask": self.tgt_attention_mask[index]
        }

    def __len__(self):
        return np.shape(self.src_input_ids)[0]