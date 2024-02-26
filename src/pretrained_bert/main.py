import os
import numpy as np

import sacrebleu
from tqdm import tqdm


import torch
from torch.utils.data import Dataset

from datasets import load_dataset, load_metric
from transformers import *

import warnings
warnings.filterwarnings("ignore")

# Prepare dataset
class NMTDataset(Dataset):
    def __init__(self, cfg, data_type="train"):
        super().__init__()
        self.cfg = cfg

        self.src_texts, self.tgt_texts = self.read_data(data_type)

        self.src_input_ids, self.src_attention_mask = self.texts_to_sequences(self.src_texts)
        self.tgt_input_ids, self.tgt_attention_mask, self.labels = self.texts_to_sequences(
            self.tgt_texts,
            is_src=False
        )

    def read_data(self, data_type):
        data = load_dataset(
            "mt_eng_vietnamese",
            "iwslt2015-en-vi",
            split=data_type
        )
        src_texts = [sample["translation"][self.cfg.src_lang] for sample in data]
        tgt_texts = [sample["translation"][self.cfg.tgt_lang] for sample in data]
        return src_texts, tgt_texts

    def texts_to_sequences(self, texts, is_src=True):
        if is_src:
            src_inputs = self.cfg.src_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.cfg.src_max_len,
                return_tensors='pt'
            )
            return (
                src_inputs.input_ids,
                src_inputs.attention_mask
            )

        else:
            if self.cfg.add_special_tokens:
                texts = [
                    ' '.join([
                        self.cfg.tgt_tokenizer.bos_token,
                        text,
                        self.cfg.tgt_tokenizer.eos_token
                        ])
                    for text in texts
                ]
            tgt_inputs = self.cfg.tgt_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.cfg.tgt_max_len,
                return_tensors='pt'
            )

            labels = tgt_inputs.input_ids.numpy().tolist()
            labels = [
                [
                    -100 if token_id == self.cfg.tgt_tokenizer.pad_token_id else token_id
                    for token_id in label
                ]
                for label in labels
            ]

            labels = torch.LongTensor(labels)

            return (
                tgt_inputs.input_ids,
                tgt_inputs.attention_mask,
                labels
            )

    def __getitem__(self, idx):
        return {
            "input_ids": self.src_input_ids[idx],
            "attention_mask": self.src_attention_mask[idx],
            "decoder_input_ids": self.tgt_input_ids[idx],
            "decoder_attention_mask": self.tgt_attention_mask[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return np.shape(self.src_input_ids)[0]
    
# Load tokenizer and model
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def load_tokenizer(model_name_or_path):
    if 'bert' in model_name_or_path.split('-'):
        return BertTokenizerFast.from_pretrained(model_name_or_path)
    elif 'gpt2' in model_name_or_path.split('-'):
        return GPT2TokenizerFast.from_pretrained(model_name_or_path)
    else:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    
class Manager():
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg

        print("Loading Tokenizer...")
        self.get_tokenizer()

        print("Loading Model...")
        self.get_model()

        print("Loading Metric...")
        self.bleu_metric = load_metric("sacrebleu")

        print("Check Save Model Path")
        if not os.path.exists(self.cfg.ckpt_dir):
            os.mkdir(self.cfg.ckpt_dir)

        if is_train:
            # Load dataloaders
            print("Loading Dataset...")
            self.train_dataset = NMTDataset(self.cfg, data_type="train")
            self.valid_dataset = NMTDataset(self.cfg, data_type="validation")

        print("Setting finished.")

    def get_tokenizer(self):
        if self.cfg.load_model_from_path:
            self.cfg.src_tokenizer = load_tokenizer(self.cfg.ckpt_dir)
            self.cfg.tgt_tokenizer = load_tokenizer(self.cfg.ckpt_dir)
        else:
            self.cfg.src_tokenizer = load_tokenizer(self.cfg.src_model_name)
            self.cfg.tgt_tokenizer = load_tokenizer(self.cfg.tgt_model_name)
            if "bert" in self.cfg.tgt_model_name.split('-'):
                self.cfg.add_special_tokens = False
                self.cfg.bos_token_id = self.cfg.tgt_tokenizer.cls_token_id
                self.cfg.eos_token_id = self.cfg.tgt_tokenizer.sep_token_id
                self.cfg.pad_token_id = self.cfg.tgt_tokenizer.pad_token_id
            else:
                self.cfg.add_special_tokens = True
                self.cfg.tgt_tokenizer.add_special_tokens(
                    {
                        "bos_token": "[BOS]",
                        "eos_token": "[EOS]",
                        "pad_token": "[PAD]"
                    }
                )
                self.cfg.bos_token_id = self.cfg.tgt_tokenizer.bos_token_id
                self.cfg.eos_token_id = self.cfg.tgt_tokenizer.eos_token_id
                self.cfg.pad_token_id = self.cfg.tgt_tokenizer.pad_token_id
        self.cfg.src_tokenizer.save_pretrained(
                os.path.join(self.cfg.ckpt_dir, f"{self.cfg.src_lang}_tokenizer_{cfg.src_model_name}")
            )

        self.cfg.tgt_tokenizer.save_pretrained(
                os.path.join(self.cfg.ckpt_dir, f"{self.cfg.tgt_lang}_tokenizer_{cfg.tgt_model_name}")
            )

    def get_model(self):
        if self.cfg.load_model_from_path:
            save_model_path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_name)
            self.model = EncoderDecoderModel.from_pretrained(save_model_path)
        else:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.cfg.src_model_name,
                self.cfg.tgt_model_name
            )
            self.model.decoder.resize_token_embeddings(len(self.cfg.tgt_tokenizer))
            self.model.config.decoder_start_token_id = self.cfg.bos_token_id
            self.model.config.eos_token_id = self.cfg.eos_token_id
            self.model.config.pad_token_id = self.cfg.pad_token_id
            self.model.config.vocab_size = len(self.cfg.tgt_tokenizer)
            self.model.config.max_length = self.cfg.max_length_decoder
            self.model.config.min_length = self.cfg.min_length_decoder
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.early_stopping = True
            self.model.config.length_penalty = 2.0
            self.model.config.num_beams = self.cfg.beam_size

    def train(self):
        print("Training...")
        if self.cfg.use_eval_steps:
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="steps",
                save_strategy='steps',
                save_steps=self.cfg.eval_steps,
                eval_steps=self.cfg.eval_steps,
                output_dir=self.cfg.ckpt_dir,
                per_device_train_batch_size=self.cfg.train_batch_size,
                per_device_eval_batch_size=self.cfg.eval_batch_size,
                learning_rate=self.cfg.learning_rate,
                weight_decay=0.005,
                num_train_epochs=self.cfg.num_train_epochs
            )
        else:
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                evaluation_strategy="epoch",
                save_strategy='epoch',
                output_dir=self.cfg.ckpt_dir,
                per_device_train_batch_size=self.cfg.train_batch_size,
                per_device_eval_batch_size=self.cfg.eval_batch_size,
                learning_rate=self.cfg.learning_rate,
                weight_decay=0.005,
                num_train_epochs=self.cfg.num_train_epochs
            )

        data_collator = DataCollatorForSeq2Seq(
            self.cfg.tgt_tokenizer,
            model=self.model
        )

        trainer = Seq2SeqTrainer(
            self.model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            data_collator=data_collator,
            tokenizer=self.cfg.tgt_tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.cfg.tgt_tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.cfg.tgt_tokenizer.pad_token_id)
        decoded_labels = self.cfg.tgt_tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        result = {"bleu_score": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.cfg.tgt_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
    
# Prepare config
class BaseConfig:
    """ base Encoder Decoder config """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class NMTConfig(BaseConfig):
    # Data
    src_lang = 'en'
    tgt_lang = 'vi'
    src_max_len = 75
    tgt_max_len = 75

    # Model
    src_model_name = "bert-base-multilingual-cased"
    tgt_model_name = "bert-base-multilingual-cased"

    # Training
    load_model_from_path = False
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learning_rate = 3e-5
    train_batch_size = 16
    eval_batch_size = 8
    num_train_epochs = 5
    ckpt_dir = src_model_name + '_to_' + tgt_model_name
    use_eval_steps = False
    eval_steps = 2000

    # Inference
    max_length_decoder = 75
    min_length_decoder = 25
    beam_size = 1

# Evaluate
def load_model(cfg, checkpoint_name):
    # Load Tokenizer
    src_tokenizer_save_path = f"{cfg.ckpt_dir}/{cfg.src_lang}_tokenizer_{cfg.src_model_name}"
    src_tokenizer = BertTokenizerFast.from_pretrained(src_tokenizer_save_path)

    tgt_tokenizer_save_path = f"{cfg.ckpt_dir}/{cfg.tgt_lang}_tokenizer_{cfg.tgt_model_name}"
    tgt_tokenizer = GPT2TokenizerFast.from_pretrained(tgt_tokenizer_save_path)

    # Load Model
    model_save_path = f"{cfg.ckpt_dir}/{checkpoint_name}"
    model = EncoderDecoderModel.from_pretrained(model_save_path)

    # Inference Param
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return src_tokenizer, tgt_tokenizer, model, device

# Inference
def inference(
    text,
    src_tokenizer,
    tgt_tokenizer,
    model,
    device="cpu",
    max_length=75,
    beam_size=5
    ):
    inputs = src_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
        )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    model.to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        early_stopping=True,
        num_beams=beam_size,
        length_penalty=2.0
    )

    output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

def inference_bath(
    texts,
    src_tokenizer,
    tgt_tokenizer,
    model,
    device="cpu",
    max_length=75,
    beam_size=5,
    batch_size=32
    ):

    pred_texts = []

    if len(texts) < batch_size:
        batch_size = len(texts)

    for x in tqdm(range(0, len(texts), batch_size)):
        text = texts[x:x+batch_size]

        inputs = src_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
            )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        model.to(device)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            early_stopping=True,
            num_beams=beam_size,
            length_penalty=2.0
        )

        output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_texts.extend(output_str)
        torch.cuda.empty_cache()

    return pred_texts

if __name__ == '__main__':
    cfg = NMTConfig()
    manager = Manager(cfg, is_train=True)
    manager.train()
    print("Training finished.")