import pandas as pd
import numpy as np
import torch
import json
import time
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model_prot2motif import (EncoderRNN, AttnDecoderRNN, Seq2Seq)
from typing import Tuple, Dict, List

class ValidationLossEarlyStopping:

    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop_check(self, validation_loss: float) -> bool:
        """
        Checks if early stopping criteria are met.
        
        Parameters
        ----------
        Args: validation_loss (float): Current validation loss.

        Returns
        -------
        bool: True if early stopping criteria are met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def generate_dataloader(df: pd.DataFrame, count_sample: int = None):
    """
    Generates a random sample from the DataFrame and returns the sample and its indices.

    Args:
        df (pd.DataFrame): Input DataFrame.
        count_sample (int, optional): Number of samples to select. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Index]: Randomly sampled DataFrame and its indices.
    """
    if count_sample is None or df.shape[0] < count_sample:
        count_sample = df.shape[0]

    random_rows = df.sample(n=count_sample)
    random_indexes = random_rows.index

    return random_rows, random_indexes

def pad_tensor_custom(tensor: torch.Tensor, target_height: int = 25, target_width: int = 5):
    """
    Pads a tensor to a specified height and width.

    Args:
        tensor (torch.Tensor): Input tensor.
        target_height (int): Target height.
        target_width (int): Target width.

    Returns:
        torch.Tensor: Padded tensor.
    """
    height, width = tensor.shape
    zeros_col = torch.zeros((height, 1), dtype=tensor.dtype)
    tensor_expanded = torch.cat((tensor, zeros_col), dim=1)

    rows_to_add = target_height - height
    if rows_to_add < 0:
        raise ValueError(f"Original height {height} exceeds target height {target_height}")

    added_rows = torch.zeros((rows_to_add, target_width), dtype=tensor.dtype)
    if rows_to_add > 0:
        added_rows[:, -1] = 1
    result = torch.cat((tensor_expanded, added_rows), dim=0)

    return result

def asMinutes(s: float):
    """Converts seconds to minutes and seconds format."""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since: float, percent: float):
    """Calculates time elapsed and estimated remaining time."""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def padding(tensor_aa: torch.Tensor):
    """Pads a tensor to a fixed size of (337, 1280)."""
    padded_tensor = torch.zeros((337, 1280), dtype=tensor_aa.dtype, device=tensor_aa.device)
    padded_tensor[:tensor_aa.size(0), :] = tensor_aa
    return padded_tensor

def process_batch(row: pd.DataFrame, boundary_domens: Dict, path_dir_embed: str, device: torch.device):
    """Processes a batch of data."""
    nucleotide_matrix = [torch.tensor(eval(row.Matrix[idx]), requires_grad=False) for idx in row.index.tolist()]
    nucleotide_tensor = list(map(lambda x: pad_tensor_custom(x, 25, 5), nucleotide_matrix))
    protein_tensor = [torch.load(path_dir_embed + path_embed +'.pt')['representations'][33][boundary_domens[path_embed][0]: boundary_domens[path_embed][1]] for path_embed in row.Proteins_samples]
    protein_tensor = [padding(a) for a in protein_tensor]
    data = list(zip(protein_tensor, nucleotide_tensor))
    input_tensors = [d[0].to(device) for d in data]
    target_tensors = [d[1].to(device) for d in data]
    return list(zip(input_tensors, target_tensors))

def train_epoch(train_dataloader: pd.DataFrame, boundary_domens: Dict, model: nn.Module, model_optimizer: optim.Optimizer, criterion: nn.Module, path_dir_embed: str, device: torch.device):

    model.train()
    total_loss = 0
    batch_size = 32
    
    index_list = train_dataloader.index.tolist()
    
    np.random.shuffle(index_list)
    
    for i in range(0, len(index_list), batch_size):
        batch_indices = index_list[i:i + batch_size]
        batch_rows = train_dataloader.loc[batch_indices]
        
        data = process_batch(batch_rows, boundary_domens, path_dir_embed, device)
        
        model_optimizer.zero_grad()
        
        batch_loss = 0
        for input_tensor, target_tensor in data:

            output = model(input_tensor)
            
            loss = criterion(output, target_tensor)
            batch_loss += loss
        
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optimizer.step()
        
        total_loss += batch_loss.item()
    
    avg_loss = total_loss / (len(train_dataloader) / batch_size)
    return avg_loss

def validate_epoch(validate_dataloader: pd.DataFrame, boundary_domens: Dict, model: nn.Module, criterion: nn.Module, path_dir_embed: str, device: torch.device):

    model.eval()
    total_loss = 0
    batch_size = 32
    
    index_list = validate_dataloader.index.tolist()
    
    np.random.shuffle(index_list)
    
    for i in range(0, len(index_list), batch_size):
        batch_indices = index_list[i:i + batch_size]
        batch_rows = validate_dataloader.loc[batch_indices]
        
        data = process_batch(batch_rows, boundary_domens, path_dir_embed, device)

        with torch.no_grad():
            batch_loss = 0
            for input_tensor, target_tensor in data:
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                batch_loss += loss
            total_loss += batch_loss.item()
    
    avg_loss = total_loss / (len(validate_dataloader) / batch_size)
    return avg_loss

def train(train_dataloader: pd.DataFrame, validate_dataloader: pd.DataFrame, model: nn.Module, n_epochs: int, learning_rate: float = 0.001, early_stop_patience: int = 4):

    start = time.time()
    losses_train: List[float] = []
    losses_valid: List[float] = []
    best_val_loss: float = float('inf') 
    
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    early_stopping = ValidationLossEarlyStopping(patience=early_stop_patience, min_delta=0.005)
    
    path_dir_embed = '/beegfs/scratch/ws/ws1/sheludyakov-predict_TFBS/TransFormerBS/seq2seqMatrix/embeddings_seq2seqMatrix/'
    with open('/beegfs/scratch/ws/ws1/sheludyakov-predict_TFBS/TransFormerBS/seq2seqMatrix/boundary_domens.json', 'r') as f:
        boundary_domens = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}')
        
        loss_train = train_epoch(train_dataloader, boundary_domens, model, model_optimizer, criterion, path_dir_embed, device)
        losses_train.append(loss_train)
        print(f'Train Loss: {loss_train}')
        
        loss_valid = validate_epoch(validate_dataloader, boundary_domens, model, criterion, path_dir_embed, device)
        losses_valid.append(loss_valid)
        print(f'Validation Loss: {loss_valid}')
        
        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optimizer.state_dict(),
                'loss': loss_train,
            }, f'/beegfs/scratch/ws/ws1/sheludyakov-predict_TFBS/TransFormerBS/seq2seqMatrix/checkpoint_best_model.pt')
            print("Model saved! (Best Validation Loss)")
        
        if early_stopping.early_stop_check(loss_valid):
            print('Early stopping triggered!')
            break
    
    with open('/beegfs/scratch/ws/ws1/sheludyakov-predict_TFBS/TransFormerBS/seq2seqMatrix/losses_models.txt', 'w+', encoding='utf-8') as file:
        file.write(f"Train_losses: {' '.join(map(str, losses_train))}\n")
        file.write(f"Validation_losses: {' '.join(map(str, losses_valid))}")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model_optimizer.state_dict(),
        'loss': loss_train,
    }, f'/beegfs/scratch/ws/ws1/sheludyakov-predict_TFBS/TransFormerBS/seq2seqMatrix/model_seq2seqMatrix.pt')
    
    return losses_train, losses_valid
