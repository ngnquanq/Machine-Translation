import time
import torch
import torch.nn as nn

from model import *

class Trainer:
    def __init__(self, model, optimizer, criterion, 
                 train_dataloader, test_dataloader,
                 epochs, device) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
    
    def _train_epoch(self, model, optimizer, criterion,
                     train_dataloader, device):
        model.train()
        losses = []
        
        for src_idx, tgt_idx in train_dataloader:
            src_idx, tgt_idx = src_idx.to(self.device), tgt_idx.to(self.device)
            tgt_input = tgt_idx[:, :-1]
            tgt_output = tgt_idx[:, 1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_idx, tgt_input)
            try:
                logits = self.model(src_idx, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            except Exception as e:
                print(f"src_idx.shape: {src_idx.shape}, tgt_idx.shape: {tgt_idx.shape}")
                
            optimizer.zero_grad()
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            
            optimizer.step()
            losses.append(loss.item())
        
        return sum(losses) / len(losses)
    
    def evaluate(self, model, test_dataloader, criterion):
        model.eval()
        losses = []
        with torch.no_grad():
            for src_idx, tgt_idx in test_dataloader:
                src_idx, tgt_idx = src_idx.to(self.device), tgt_idx.to(self.device)
                tgt_input = tgt_idx[:, :-1]
                tgt_output = tgt_idx[:, 1:]
                
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_idx, tgt_input)
                logits = model(src_idx, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
                losses.append(loss.item())
                
        return sum(losses) / len(losses)
    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss = self._train_epoch(self.model, self.optimizer, self.criterion, self.train_dataloader, self.device)
            end_time = time.time()
            print(f"Epoch: {epoch+1}, Train loss: {train_loss}, Epoch time = {end_time - start_time}s")
            
            test_loss = self.evaluate(self.model, self.test_dataloader, self.criterion, self.optimizer)
            print(f"Test loss: {test_loss}")
            
            print("="*50)