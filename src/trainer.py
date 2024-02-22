import time
import torch
import torch.nn as nn
import logging


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
        
        # Create logging
        logging.basicConfig(filename='training.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', 
                            datefmt='%d-%b-%y %H:%M:%S')
        logging.info(f'Device: {type(self.device).__name__}')
        logging.info(f'Model: {type(self.model).__name__}')
        logging.info(f'Optimizer: {type(self.optimizer).__name__}')
        logging.info(f'Criterion: {type(self.criterion).__name__}')

        
    
    def _train_epoch(self):
        self.model.train()
        losses = []
        
        for src_idx, tgt_idx in self.train_dataloader:
            src_idx, tgt_idx = src_idx.to(self.device), tgt_idx.to(self.device)
            tgt_input = tgt_idx[:, :-1]
            tgt_output = tgt_idx[:, 1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_idx, tgt_input)
            try:
                logits = self.model(src_idx, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            except Exception as e:
                print(f"src_idx.shape: {src_idx.shape}, tgt_idx.shape: {tgt_idx.shape}")
                
            self.optimizer.zero_grad()
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            
            self.optimizer.step()
            losses.append(loss.item())
        
        return sum(losses) / len(losses)
    
    def evaluate(self):
        logging.info("Evaluating model...")
        self.model.eval()
        losses = []
        with torch.no_grad():
            for src_idx, tgt_idx in self.test_dataloader:
                src_idx, tgt_idx = src_idx.to(self.device), tgt_idx.to(self.device)
                tgt_input = tgt_idx[:, :-1]
                tgt_output = tgt_idx[:, 1:]
                
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_idx, tgt_input)
                logits = self.model(src_idx, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
                losses.append(loss.item())
                
        return sum(losses) / len(losses)
    
    def train(self):
        logging.info("Training model...")
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss = self._train_epoch(self.model, self.optimizer, self.criterion, self.train_dataloader, self.device)
            test_loss = self.evaluate(self.model, self.criterion, self.test_dataloader, self.device)
            end_time = time.time() 
            print ((f" Epoch : { epoch }, Train loss : { train_loss :.3f}, Val loss : {test_loss:.3f}, "f" Epoch time = {( end_time - start_time ):.3f}s"))
            logging.info(f"Epoch : {epoch}, Train loss : {train_loss:.3f}, Val loss : {test_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")