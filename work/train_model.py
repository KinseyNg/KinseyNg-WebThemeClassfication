import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import math
import re
from tqdm import tqdm
import torchtext
import copy

# Set seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Set memory usage optimizations
    torch.cuda.empty_cache()
else:
    print("GPU not available, using CPU")

# Global features
n_epochs = 200
MAX_DEPTHS = [5, 10]
MIN_FREQS = [3, 5]
embedding_size = 256  # Choose based on your needs
n_layers = 3
dropout_rate = 0.4
batch_size = 128
lr = 1e-3
tie_weights = True

year_token = 'YEAR'

def custom_tokenizer(path, MAX_DEPTH):
    path = path.lstrip('/')
    path_words = path.split('/')
    path_words = path_words[:MAX_DEPTH]
    for i, tok in enumerate(path_words):
        pattern = r'^\d{4}$'
        if re.match(pattern, tok):
            path_words[i] = year_token
    return path_words

def yield_tokens(data_iter, MAX_DEPTH):
    for path in data_iter:
        yield custom_tokenizer(path, MAX_DEPTH)

def create_vocabulary(MIN_FREQ, MAX_DEPTH, train_df):
    vocab = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens(train_df['Path'], MAX_DEPTH), 
        min_freq=MIN_FREQ
    )
    vocab.insert_token('<unk>', 0)
    vocab.insert_token('<eos>', 2)
    vocab.insert_token('<sos>', 1)
    vocab.insert_token('<pad>', 3)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def get_dataloader(tokens, vocab, batch_size, seq_len):
    data = []
    for token_list in tokens:
        token_list.append('<eos>')
        token_list = ['<sos>'] + token_list
        while len(token_list) < seq_len:
            token_list.append('<pad>')
        mapped_tokens = [vocab[token] for token in token_list]
        data.extend(mapped_tokens)
    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data.to(device)  # Move data to GPU

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate, tie_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError('When using tied weights, hidden_dim must be equal to embedding_dim')
            self.fc.weight = self.embedding.weight

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedded)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction

def train_model(model, train_data, valid_data, optimizer, criterion, n_epochs):
    best_valid_loss = float('inf')
    best_model = None
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for i in range(train_data.size(1) - 1):
            src = train_data[:, i].unsqueeze(1)
            trg = train_data[:, i + 1].unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.squeeze(1), trg.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for i in range(valid_data.size(1) - 1):
                src = valid_data[:, i].unsqueeze(1)
                trg = valid_data[:, i + 1].unsqueeze(1)
                
                output = model(src)
                loss = criterion(output.squeeze(1), trg.squeeze(1))
                valid_loss += loss.item()
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            
            # Save the best model
            torch.save(best_model.state_dict(), 'best_model.pt')
    
    return best_model

def main():
    # Load data
    data_folder = 'datasets/LM-training-datasets'
    train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(data_folder, 'validation.csv'))
    
    # Create vocabulary and prepare data
    MAX_DEPTH = MAX_DEPTHS[0]  # Choose the depth you want
    MIN_FREQ = MIN_FREQS[0]    # Choose the frequency you want
    seq_len = MAX_DEPTH + 2    # +2 for <sos> and <eos> tokens
    
    vocab = create_vocabulary(MIN_FREQ, MAX_DEPTH, train_df)
    vocab_size = len(vocab)
    
    train_tokens = [custom_tokenizer(url, MAX_DEPTH) for url in train_df['Path']]
    valid_tokens = [custom_tokenizer(url, MAX_DEPTH) for url in validation_df['Path']]
    
    train_data = get_dataloader(train_tokens, vocab, batch_size, seq_len)
    valid_data = get_dataloader(valid_tokens, vocab, batch_size, seq_len)
    
    # Initialize model
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_size,
        hidden_dim=embedding_size if tie_weights else embedding_size*2,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        tie_weights=tie_weights
    ).to(device)  # Move model to GPU
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # Train the model
    best_model = train_model(model, train_data, valid_data, optimizer, criterion, n_epochs)
    print("Training completed!")

if __name__ == "__main__":
    main()
