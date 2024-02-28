import numpy as np
import json
import hyper as hp

from src.seq2seq.data import *
from src.seq2seq.model import *

import torch
import torch.nn as nn
import torchtext

# Load model:
def _load_model():
    train_dataloader, valid_dataloader, test_dataloader, source_vocab, target_vocab, source_text, target_text = prepare_data()
    transformer = TranslationModel(hp.NUM_ENCODER_LAYERS,
                                   hp.NUM_DECODER_LAYERS,
                                   hp.EMB_SIZE,
                                   hp.N_HEADS
                                   ,len(source_vocab),len(target_vocab),
                                   hp.FFN_HID_DIM,0.1)
    return transformer, source_vocab, target_vocab, source_text, target_text