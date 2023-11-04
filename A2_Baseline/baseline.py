import torch
import torchtext
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import pandas as pd
from A2_Starter import TextDataset, my_collate_function
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

def train_model(network, train_dl, valid_dl, 
                test_dl, epochs, lr, train_type="train"):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    train_acc = []
    val_acc = []
    test_acc =0
    train_loss = []
    val_loss = []
    test_loss = 0
    for epoch in tqdm(range(epochs)):
        loss_t = 0
        loss_v = 0
        acc_t = 0
        acc_v = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            # print(x.shape)
            # print(y.shape)
            optimizer.zero_grad()
            pred = network(x)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimizer.step()
            loss_t += loss.item()
            acc_t += torch.sum((pred > 0) == y.byte()).item()/len(y)
        train_loss.append(loss_t / len(train_dl))
        train_acc.append(acc_t / len(train_dl))
        if epoch  % 2 == 0:
            for x, y in valid_dl:
                x = x.to(device)
                y = y.to(device)
                pred = network(x)
                loss = loss_fn(pred, y.float())
                loss_v += loss.item()
                acc_v += torch.sum((pred > 0) == y.byte()).item()/len(y)
            val_loss.append(loss_v / len(valid_dl))
            val_acc.append(acc_v / len(valid_dl))
            print("Epoch: {} Train loss: {} Train acc: {} Val loss: {} Val acc: {}".format(epoch, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]))
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            pred = network(x)
            loss = loss_fn(pred, y.float())
            pred = torch.round(torch.sigmoid(pred))
            test_loss += loss.item()
            test_acc += torch.sum(pred == y.byte()).item()/len(y)
    test_loss = test_loss / len(test_dl)
    test_acc = test_acc / len(test_dl)
    print("Test acc: {} Test loss: {}".format(test_acc, test_loss))
    return train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, network



def overfit_model_run():
    train_dataset = TextDataset(glove, "overfit")
    val_dataset = TextDataset(glove, "valid")
    test_dataset = TextDataset(glove, "test")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))
    
    network = baseline(glove)
    network.to(device)
    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, network = train_model(network, train_dataloader, validation_dataloader, test_dataloader, 50, 0.1)
    eps = np.arange(0, 50, 2)
    plt.plot(eps,train_acc[::2], label="train accuracy")
    # plt.plot(eps,val_acc, label="validation accuracy")
    plt.legend()
    plt.show()
    plt.plot(eps,train_loss[::2], label="train loss")
    # plt.plot(eps,val_loss, label="validation loss")
    plt.legend()
    plt.show()

def test_batch_sizes():
    batch_sizes = [4, 8, 16, 32, 64, 128]
    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    for batch_size in batch_sizes:
        train_dataset = TextDataset(glove, "train")
        val_dataset = TextDataset(glove, "valid")
        test_dataset = TextDataset(glove, "test")
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: my_collate_function(batch, device))
        validation_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size= batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: my_collate_function(batch, device))
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size= batch_size,
            shuffle=False,
            collate_fn=lambda batch: my_collate_function(batch, device))
        
        network = baseline(glove)
        network.to(device)
        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, network = train_model(network, train_dataloader, validation_dataloader, test_dataloader, 50, 0.001)
        eps = np.arange(0, 50, 2)
        plt.plot(eps,train_acc[::2], label="train accuracy")
        plt.plot(eps,val_acc, label="validation accuracy")
        plt.legend()
        plt.show()
        plt.plot(eps,train_loss[::2], label="train loss")
        plt.plot(eps, val_loss, label="validation loss")
        plt.legend()
        plt.show()
        print("Batch size: {} \n\n".format(batch_size))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)


def final_model_run(batch_size):
    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    train_dataset = TextDataset(glove, "train")
    val_dataset = TextDataset(glove, "valid")
    test_dataset = TextDataset(glove, "test")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size= batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: my_collate_function(batch, device))
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size= batch_size,
        shuffle=False,
        collate_fn=lambda batch: my_collate_function(batch, device))
    
    network = baseline(glove)
    network.to(device)
    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, network = train_model(network, train_dataloader, validation_dataloader, test_dataloader, 50, 0.001)

    # Plot the accuraries and losses
    eps = np.arange(0, 50, 2)
    plt.plot(eps,train_acc[::2], label="train accuracy")
    plt.plot(eps,val_acc, label="validation accuracy")
    plt.legend()
    # plt.savefig("baseline_acc.png")
    plt.show()
    plt.plot(eps,train_loss[::2], label="train loss")
    plt.plot(eps, val_loss, label="validation loss")
    plt.legend()
    # plt.savefig("baseline_loss.png")
    plt.show()
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    train_losses.append(train_loss)

    # Save the model
    torch.save(network.state_dict(), "baseline_model.pt")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--type_of_run", type=str, default="final")
    args = parser.parse_args()
    batch_size = args.batch_size
    torch.manual_seed(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100
    if args.type_of_run == "overfit":
        print("Running overfit model")
        overfit_model_run()
    elif args.type_of_run == "batch_test":
        print("Running batch size test")
        test_batch_sizes()
    elif args.type_of_run == "final":
        print("Running final model")
        final_model_run(batch_size)
    else:
        print("Invalid type of run, select one from 'overfit', 'batch_test', 'final'.")