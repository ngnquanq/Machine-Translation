from data import *
from model import *
from trainer import Trainer

import json
with open('C:\\Users\\84898\\Desktop\\project\\WIP\\Machine Translation\\src\\config.json') as f:
    config = json.load(f)

train_dataloader, valid_dataloader, test_dataloader, x, y = prepare_data()