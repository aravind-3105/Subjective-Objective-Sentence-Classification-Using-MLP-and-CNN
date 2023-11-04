import torch
import torchtext
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import pandas as pd
import argparse

class baseline(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.word_embedding_size = vocab.vectors.shape[1]
        self.vocab_size = vocab.vectors.shape[0]
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.linear = nn.Linear(self.word_embedding_size, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(torch.mean(x, 0, True)).squeeze()
        return x

class CNN_module_unfreeze(nn.Module):
    def __init__(self, vocab, n1, k1, n2, k2):
        super().__init__()
        self.vocab = vocab
        self.n1 = n1
        self.k1 = k1
        self.n2 = n2
        self.k2 = k2
        self.embedding_dim = 100
        # Layer 1
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        # Layer 2
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.n1, (self.k1, self.embedding_dim),bias=False),
                                    nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1)))
        # Layer 3
        self.conv2 = nn.Sequential(nn.Conv2d(1, self.n2, (self.k2, self.embedding_dim),bias=False),
                                    nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1)))
        # FC Layer
        self.linear = nn.Linear(self.n1 + self.n2, 1)
        # Sigmoid function
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Layer 1
        x = self.embedding(x)
        x = torch.transpose(x, 0, 1).unsqueeze(1)
        # Layer 2
        x1 = self.conv1(x)
        # Layer 3
        x2 = self.conv2(x)
        # Combine both the conv layers
        x = torch.cat((x1, x2), dim=1)
        # Send to FC layer
        fc_out = self.linear(x.squeeze())
        # Sigmoid function
        sig_out = self.sig(fc_out)
        return sig_out.squeeze(), fc_out.squeeze()
    

class CNN_module(nn.Module):
    def __init__(self, vocab, n1, k1, n2, k2):
        super().__init__()
        self.vocab = vocab
        self.n1 = n1
        self.k1 = k1
        self.n2 = n2
        self.k2 = k2
        self.embedding_dim = 100
        # Layer 1
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        # Layer 2
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.n1, (self.k1, self.embedding_dim),bias=False),
                                    nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1)))
        # Layer 3
        self.conv2 = nn.Sequential(nn.Conv2d(1, self.n2, (self.k2, self.embedding_dim),bias=False),
                                    nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1)))
        # FC Layer
        self.linear = nn.Linear(self.n1 + self.n2, 1)
        # Sigmoid function
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Layer 1
        x = self.embedding(x)
        x = torch.transpose(x, 0, 1).unsqueeze(1)
        # Layer 2
        x1 = self.conv1(x)
        # Layer 3
        x2 = self.conv2(x)
        # Combine both the conv layers
        x = torch.cat((x1, x2), dim=1)
        # Send to FC layer
        fc_out = self.linear(x.squeeze())
        # Sigmoid function
        sig_out = self.sig(fc_out)
        return sig_out.squeeze(), fc_out.squeeze()
    