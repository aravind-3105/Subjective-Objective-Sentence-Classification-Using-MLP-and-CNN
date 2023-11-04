import torch
import torchtext
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import pandas as pd
import argparse
from models import CNN_module, CNN_module_unfreeze
from A2_Starter import TextDataset, my_collate_function
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


#   fix seed
torch.manual_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Using device:", device)
glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100


def print_closest_cosine_words(vec, n=5):
    dists = torch.cosine_similarity(glove.vectors, vec, dim=1)
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1], reverse=True)
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)


if __name__ == '__main__':

    network = CNN_module(glove, 40, 5, 40, 3)
    network = network.state_dict(torch.load("models/model_40_5_40_3_16_0.0001_30_0.9085.pt"))
    kernel_conv1 = network['conv1.0.weight'].squeeze().reshape(-1, 100)
    kernel_conv2 = network['conv2.0.weight'].squeeze().reshape(-1, 100)
    kernels = torch.cat((kernel_conv1, kernel_conv2), dim=0)
    print("Total kernels: ", kernels.shape[0])
    words = []
    for i in tqdm(range(len(kernels))):
        print("Kernel Number: ", i)
        words.append(print_closest_cosine_words(kernels[i], 5))
        print("\n")

    