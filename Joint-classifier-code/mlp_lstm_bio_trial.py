#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Sample command to run:
python mlp_lstm_bio_trial -k 0 -c 0
'''

import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../new_iteration/")
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils import data as data2
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import os
import argparse
from sklearn.metrics import confusion_matrix
from datetime import datetime
from tas_utils_bs import get_trainValLoader, get_testLoader

np.random.seed(1)
torch.manual_seed(1)


# In[2]:


# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
args = parser.parse_args()


# In[3]:


# # For notebook args parser
# class Args:
#   kfold = 0
#   reduction = 1
#   cuda = '1'

# args=Args()


# In[4]:


data_dir = '../../new_data_folder/'
kfold_number = args.kfold

num_class = 20
learning_rate = 0.0001
num_epochs = 5000
hidden_size = 40
num_layers = 1
dropout = 0.2

logDir = 'models_and_stat/'
model_name = 'mlp18_lstm_bio_hs_{}_layer_{}_{}'.format(hidden_size, num_layers, str(kfold_number))
device = torch.device("cuda:{}".format(args.cuda))

train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False)




# ### set module

# In[9]:


class BIO_LSTM(nn.Module):
    def __init__(self, num_class, input_size=19, hidden_size=50, num_layers=2, dropout=0.5, latent_length=18, freeze_lstm=False):
        super(BIO_LSTM, self).__init__()
        lstm_input = input_size
        self.lstm = nn.LSTM(
            input_size=lstm_input, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
           dropout=dropout)

        self.linear = nn.Linear(hidden_size,num_class)

    def forward(self, x, device):
        batch_size, H, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out = x[...,t]
            cnn_embed_seq.append(cnn_out)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
#         print('cnn_embed_seq: ', cnn_embed_seq.shape)
        
        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(cnn_embed_seq)
#         print('lstm out: ', r_out.shape)
        
        # decision making layer
        r_out2 = self.linear(r_out[:, -1, :])
        output = F.log_softmax(r_out2, dim=1)
        return output

# In[19]:


class BIO_MLP_LSTM(nn.Module):
    # TODO: add linear layer from hidden_size to latent
    def __init__(self, num_class, input_size=19, lstm_input=18, hidden_size=50, num_layers=num_layers, dropout=dropout, latent_length=18, freeze_lstm=False):
        super(BIO_MLP_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=lstm_input, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
           dropout=dropout)

        self.mlp = nn.Linear(input_size,lstm_input)
        self.linear = nn.Linear(hidden_size,num_class)

    def forward(self, x, device):
        batch_size, H, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            x_t = x[...,t]
            cnn_out = self.mlp(x_t)
            cnn_embed_seq.append(cnn_out)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
#         print('cnn_embed_seq: ', cnn_embed_seq.shape)
        
        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(cnn_embed_seq)
#         print('lstm out: ', r_out.shape)
        
        # decision making layer
        r_out2 = self.linear(r_out[:, -1, :])
        output = F.log_softmax(r_out2, dim=1)
        return output


# In[10]:


model = BIO_MLP_LSTM(num_class, input_size=19, lstm_input=18, hidden_size=hidden_size, num_layers=2, dropout=0.2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[12]:


epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []
model.train()
max_val_acc = 0
for epoch in range(1, num_epochs + 1):
    
    # TRAIN
    model.train()
    correct = 0
    train_loss = 0
    for i, (_, tact_bio,  label) in enumerate(train_loader):
        tact_bio = tact_bio.to(device)
        label = label.to(device)
        label = label.long()
        optimizer.zero_grad()
        
        #print(tact_bio.shape)
        output = model(tact_bio, device)
        
        
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        
        # Obtain classification accuracy
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum().item()
        
        # loss
        train_loss += loss.item() #criterion(output, target).item()  # sum up batch loss
        
    # fill stats
    train_accuracy = correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    epoch_train_loss.append(train_loss)  # only save the last batch
    epoch_train_acc.append(train_accuracy) 

    # VALIDATION
    model.eval()
    correct = 0
    val_loss = 0
    for i, (_, tact_bio,  label) in enumerate(val_loader):
        tact_bio = tact_bio.to(device)
        label = label.to(device)
        label = label.long()

        output = model(tact_bio, device)
        loss = F.nll_loss(output, label)
        
        # Obtain classification accuracy
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum().item()
        
        # loss
        val_loss += loss.item() #criterion(output, target).item()  # sum up batch loss
        
    # fill stats
    val_accuracy = correct / len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)
    epoch_val_loss.append(val_loss)  # only save the last batch
    epoch_val_acc.append(val_accuracy)
    
    # choose model
    if max_val_acc <= val_accuracy:
        print('Saving model at ', epoch, ' epoch')
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), logDir + model_name + '.pt')
        
    if epoch < 20 or epoch % 200 == 0:
        print('Epoch: {} Loss: train {:.3f}, valid {:.3f}. Accuracy: train: {:.3f}, valid {:.3f}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))
        print('-'*20)


# In[13]:


# save stats
import pickle
all_stats = [
    epoch_train_loss,
    epoch_train_acc,
    epoch_val_loss,
    epoch_val_acc
]

pickle.dump(all_stats, open(logDir + model_name + '_stats.pkl', 'wb'))


# In[15]:


# testing set check
net_trained = BIO_MLP_LSTM(num_class, input_size=19, lstm_input=18, hidden_size=hidden_size, num_layers=2, dropout=0.2).to(device)
net_trained.load_state_dict(torch.load(logDir + model_name + '.pt'))
net_trained.eval()


# In[16]:


# VALIDATION
correct = 0
for i, (_, tact_bio,  label) in enumerate(test_loader):
    tact_bio = tact_bio.to(device)
    label = label.to(device)
    label = label.long()

    output = model(tact_bio, device)

    # Obtain classification accuracy
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(label.data.view_as(pred)).long().cpu().sum().item()

# fill stats
test_accuracy = correct / len(test_loader.dataset)


# In[17]:


print('Test accuracy for', str(kfold_number), ' fold : ', test_accuracy)


# In[ ]:




