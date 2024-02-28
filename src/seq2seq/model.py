import configparser
import json
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math
import logging
import os


# Setup device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('C:\\Users\\84898\\Desktop\\project\\WIP\\Machine Translation\\src\\config.json') as f:
    config = json.load(f)

# Posiotnal Encoding
class PostionalEncoding(nn.Module):
    def __init__(self, emb_dim: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PostionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_dim, 2) * math.log(10000) / emb_dim)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
# Token embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim
        
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_dim)
    
# Translation model:
class TranslationModel(nn.Module):
    def __init__(self, encoder_layer: int, decoder_layer: int,
                 emb_dim: int, n_head: int, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,  
                 d_ffn: int, dropout: float
    ):
        super(TranslationModel, self).__init__()
        self.positional_encoding = PostionalEncoding(emb_dim, dropout)
        self.src_token_embedding = TokenEmbedding(src_vocab_size, emb_dim)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, emb_dim)
        self.Transformer = Transformer(d_model=emb_dim, nhead=n_head,
                                       num_encoder_layers=encoder_layer,
                                       num_decoder_layers=decoder_layer,
                                       dim_feedforward=d_ffn, dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_dim, tgt_vocab_size)
        
        logging.basicConfig(filename='model.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', 
                            datefmt='%d-%b-%y %H:%M:%S')
        logging.info(f"Number of layer: {encoder_layer} encoder, {decoder_layer} decoder")
        logging.info(f"Model dimension: {emb_dim}")
        logging.info(f"Feedforward dimension: {d_ffn}")
        logging.info(f"Vocab size: {src_vocab_size} source, {tgt_vocab_size} target")
        logging.info(f"Parameters: {self.count_param()}")
        logging.info("===============================================================")
        
    def count_param(self):
            return sum(p.numel() for p in self.Transformer.parameters())
    
    def encode(self, src: Tensor, src_mask: Tensor):
        return self.Transformer.encoder(self.positional_encoding
                                        (self.src_token_embedding(src)), src_mask
                                        )
    
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.Transformer.decoder(self.positional_encoding
                                        (self.tgt_token_embedding(tgt)), memory, tgt_mask
                                        )
    
    def forward(self, 
                src: Tensor, 
                tgt: Tensor,
                src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor) -> Tensor:
        
        src_emb = self.positional_encoding(self.src_token_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_token_embedding(tgt))
        
        outs = self.Transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        
        return self.generator(outs)
    
# Helper function:
def generate_square_subsequent_mask(sz):
    mask = torch.ones(sz, sz, device=DEVICE)
    mask = (torch.triu(mask) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[1] #0 is the batch size
    tgt_seq_len = tgt.shape[1]
    
    src_mask  = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len) # Do we need this to device?
    
    src_padding_mask = (src == config["PAD_IDX"])
    tgt_padding_mask = (tgt == config["PAD_IDX"])
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word[-1].item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word==config["EOS_IDX"]:
            break
    return ys

def translate(model, src_sentence):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=config["BOS_IDX"]).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")