from src.seq2seq.data import *
from src.seq2seq.model import *
import hyper
from hyper import *
import torch
import torch.nn as nn
import functools

# Load model
@functools.lru_cache(maxsize=None)
def _load_model():
    # Load model
    train_loader, valid_loader, test_loader, source_vocab, target_vocab, source_text, target_text = prepare_data()
    model = TranslationModel(encoder_layer=hyper.NUM_ENCODER_LAYERS,
                              decoder_layer=hyper.NUM_DECODER_LAYERS,
                              emb_dim=hyper.EMB_SIZE,
                                n_head=hyper.N_HEADS,
                                src_vocab_size=len(source_vocab),
                                tgt_vocab_size=len(target_vocab),
                                d_ffn=hyper.FFN_HID_DIM,
                                dropout=0.1)
    return model, source_vocab, target_vocab, source_text, target_text

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

def translate(model, src_sentence, source_text, target_vocab):
    model.eval()
    src = source_text(src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=config["BOS_IDX"]).flatten()
    return " ".join(target_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
