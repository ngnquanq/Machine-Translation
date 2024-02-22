import warnings
warnings.filterwarnings("ignore")

from data import *
from model import *
from trainer import Trainer
import torch

import json

    
if __name__ == '__main__':
    with open('C:\\Users\\84898\\Desktop\\project\\WIP\\Machine Translation\\src\\config.json') as f:
        config = json.load(f)
    
    train_dataloader, valid_dataloader, test_dataloader, x, y = prepare_data()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = TranslationModel(encoder_layer=config['MODEL_CONFIG']['num_encoder_layers'],
                             decoder_layer=config['MODEL_CONFIG']['num_decoder_layers'],
                             emb_dim=config['MODEL_CONFIG']['d_model'],
                             n_head=config['MODEL_CONFIG']['nhead'],
                             src_vocab_size=len(x.vocab),
                             tgt_vocab_size=len(y.vocab),
                             d_ffn=config['MODEL_CONFIG']['dim_feedforward'],
                             dropout=config['MODEL_CONFIG']['dropout'])
    
    model.to(DEVICE)
    trainer =  Trainer(model, torch.optim.Adam(model.parameters(), lr=0.0001), 
                       nn.CrossEntropyLoss(), train_dataloader, valid_dataloader,
                       5, DEVICE)
    trainer.train()
