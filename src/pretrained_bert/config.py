import torch

# Config
class BaseConfig:
    def __init__(self,**kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
# NMTConfig
class NMTConfig(BaseConfig):
    src_lang = 'en'
    tgt_lang = 'vi'
    src_max_len = 75
    tgt_max_len = 75

    #mdoel
    src_model_name = "bert-base-multilingual-cased"
    tgt_model_name = "bert-base-multilingual-cased"

    #Training
    load_model_from_path = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate = 3e-5
    train_batch_size = 16
    eval_batch_size = 16
    num_train_epochs = 10
    ckpt_dir = src_model_name + '_to_' + tgt_model_name
    use_eval_steps = False
    eval_steps = 400

    #Inference
    max_length_decoder = 75
    min_length_decoder = 25
    beam_size = 1