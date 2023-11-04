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

def train_model(train_dl, val_dl, test_dl, n1, k1, n2, k2, batch_size, lr, epochs, intervals, freeze):
    if freeze == "yes":
        network = CNN_module(glove, n1, k1, n2, k2)
    else:
        network = CNN_module_unfreeze(glove, n1, k1, n2, k2)
    print ("Network: ", network)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    network.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    test_acc = 0
    print("Input Parameters: n1 = {}, k1 = {}, n2 = {}, k2 = {}, batch_size = {}, lr = {}, epochs = {}, intervals = {}".format(n1, k1, n2, k2, batch_size, lr, epochs, intervals))
    for e in tqdm(range(epochs)):
        cur_t_loss = 0
        cur_v_loss = 0
        cur_t_acc = 0
        cur_v_acc = 0
        for x, y in train_dl:
            # print("Value x, y are ", x, y)
            optimizer.zero_grad()
            y_pred, y_prob = network(x)
            loss = criterion(y_pred, y.float())
            loss.backward()
            optimizer.step()
            cur_t_loss += loss.item()
            cur_t_acc += torch.sum((y_prob > 0.5) == y).item()
        train_losses.append(cur_t_loss / len(train_dl))
        train_accs.append(cur_t_acc / len(train_dl.dataset))
        for x, y in val_dl:
            y_pred, y_prob = network(x)
            loss = criterion(y_pred, y.float())
            cur_v_loss += loss.item()
            cur_v_acc += torch.sum((y_prob > 0.5) == y).item()
        val_losses.append(cur_v_loss / len(val_dl))
        val_accs.append(cur_v_acc / len(val_dl.dataset))
        if e % intervals == 0:
            print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Acc: {}, Val Acc: {}".format(e, train_losses[-1], val_losses[-1], train_accs[-1], val_accs[-1]))
    for x, y in test_dl:
        y_pred, y_prob = network(x)
        test_acc += torch.sum((y_prob > 0.5) == y).item()
    test_acc /= len(test_dl.dataset)
    print("Test Accuracy: {}".format(test_acc))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Title of the plot
    fig.suptitle("n1 = {}, k1 = {}, n2 = {}, k2 = {}, batch_size = {}, lr = {}, epochs = {}, intervals = {}".format(n1, k1, n2, k2, batch_size, lr, epochs, intervals))
    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(val_losses, label="Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(train_accs, label="Train Accuracy")
    ax[1].plot(val_accs, label="Validation Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()
    return network, test_acc



def parameter_search():
    n1_options = [5, 20, 40]
    k1_options = [3, 4, 5]
    n2_options = [5, 20, 40]
    k2_options = [2, 3, 4]
    # lr_options = [0.0001, 0.001, 0.01]
    lr = 0.0001
    batch_size = 16
    epochs = 30
    intervals = 5
    train_dataset = TextDataset(glove, "train")
    val_dataset = TextDataset(glove, "valid")
    test_dataset = TextDataset(glove, "test")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))
    for n1 in n1_options:
        for k1 in k1_options:
            for n2 in n2_options:
                for k2 in k2_options:
                    _, test_acc = train_model(train_dataloader, validation_dataloader, test_dataloader,
                                n1, k1, n2, k2, batch_size, lr, 30, 5, "yes")
                    print("n1 = {}, k1 = {}, n2 = {}, k2 = {}, batch_size = {}, lr = {}, epochs = {}, intervals = {}, test_acc = {}".format(n1, k1, n2, k2, batch_size, lr, epochs, intervals, test_acc))
                    # Save above print statement to a file
                    with open("parameter_search.txt", "a") as f:
                        f.write("n1 = {}, k1 = {}, n2 = {}, k2 = {}, batch_size = {}, lr = {}, epochs = {}, intervals = {}, test_acc = {}\n".format(n1, k1, n2, k2, batch_size, lr, epochs, intervals, test_acc))
                        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_of_run", type=str, default="train")
    parser.add_argument("--n1", type=int, default=40)
    parser.add_argument("--k1", type=int, default=5)
    parser.add_argument("--n2", type=int, default=40)
    parser.add_argument("--k2", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--intervals", type=int, default=5)
    parser.add_argument("--freeze", type=str, default="yes")
    parser.add_argument("--save", type=str, default="no")
    args = parser.parse_args()
    torch.manual_seed(2)
    glove = torchtext.vocab.GloVe(name="6B",dim=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ("Using device:", device)
    if args.type_of_run == "train":
        train_dataset = TextDataset(glove, "train")
    else:
        train_dataset = TextDataset(glove, "overfit")
    if args.type_of_run == "param_search":
        parameter_search()
        print("Parameter search")
        # Exit the program
        exit()
    else:   
        val_dataset = TextDataset(glove, "valid")
        test_dataset = TextDataset(glove, "test")
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: my_collate_function(batch, device))

        validation_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: my_collate_function(batch, device))
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: my_collate_function(batch, device))
        
        if args.type_of_run == "train": 
            network, test_acc = train_model(train_dataloader, validation_dataloader, test_dataloader, 
                                args.n1, args.k1, args.n2, args.k2, args.batch_size, args.lr, args.epochs, args.intervals, args.freeze)
            if args.freeze == "yes" and args.save == "yes":
                torch.save(network.state_dict(),"model_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(args.n1, args.k1, args.n2, args.k2, args.batch_size, args.lr, args.epochs, args.intervals))
            if args.freeze == "no" and args.save == "yes":
                torch.save(network.state_dict(),"unf_model_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(args.n1, args.k1, args.n2, args.k2, args.batch_size, args.lr, args.epochs, args.intervals))
        else:
            # Overfit
            network, test_acc = train_model(train_dataloader, validation_dataloader, test_dataloader, 50, 2, 50, 4, 
                                16, 0.001, 50, 1, args.freeze)