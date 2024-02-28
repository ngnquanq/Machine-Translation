# I dont know what to do now ._.
# Maybe flask, idk, hu ac hu ce
# Where should i host? heroku?
# Maybe torchserve is cool
import os
from flask import Flask, render_template
from flask import request, jsonify
from dotenv import load_dotenv
from src.seq2seq.model import *
from src.seq2seq.data import *

import logging
logging.basicConfig(level=logging.INFO)

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
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[-1].item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys

transformer = None

# actual function to translate input sentence into target language
def translate_function(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform_src(src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(
        tgt.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

app = Flask(__name__)

def load_model():
    global model
    with open('C:\\Users\\84898\\Desktop\\project\\WIP\\Machine Translation\\src\\config.json') as f:
        config = json.load(f)
    BOS_IDX = config['BOS_IDX']
    EOS_IDX = config['EOS_IDX']
    _,_,_,src,tgt,text_transform_src, text_transform_tgt = prepare_data()
    SRC_VOCAB_SIZE = len(src)
    TGT_VOCAB_SIZE = len(tgt)
    print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 256
    BATCH_SIZE = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    transformer = TranslationModel(encoder_layer=NUM_ENCODER_LAYERS,
                                    decoder_layer=NUM_DECODER_LAYERS,
                                    emb_dim=EMB_SIZE,
                                    n_head=NHEAD,
                                    src_vocab_size=SRC_VOCAB_SIZE,
                                    tgt_vocab_size=TGT_VOCAB_SIZE,
                                    d_ffn=FFN_HID_DIM,
                                    dropout=0.1)

@app.route('/')
def home():
    global model
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    global transformer
    if transformer is None:
        transformer = load_model()
        
    input_text = request.form['source']
    # Translate the input text using your model
    # Replace `translate_text` with the actual function to translate text
    output = translate_function(transformer, input_text)
    # Return the translated text
    return jsonify({'translation': output})

if __name__ == '__main__':

    #transformer.load_state_dict(torch.load('./models/transformer.pt'))
    logging.info("Starting app...")
    app.run(debug=True)
    logging.info("App started.")
