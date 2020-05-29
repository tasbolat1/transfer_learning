#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Sample command to run:
python cnn_lstm_icub_trial.py -k 0 -c 0
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
model_name = 'cnn_lstm_icub_' + str(kfold_number)
device = torch.device("cuda:{}".format(args.cuda))

train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False)


# ### set module

# In[5]:


# define NN models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))

    def forward(self, x):
#         print('CNN input:', x.shape)
        x = self.conv1(x)
        # print('Conv', x.size())
        
        x = F.max_pool2d(x, 2)
        # print('Pool', x.size())
        x = x.view(-1, 3*2*3)
        return x
    

class CNN_LSTM(nn.Module):
     
    def __init__(self, num_class, lstm_input=18, hidden_size=50, num_layers=num_layers, dropout=dropout, latent_length=18, freeze_lstm=False):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=lstm_input, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
           dropout=dropout)

        self.linear = nn.Linear(hidden_size,num_class)

    def forward(self, x, device):
#         print('Model input ', x.size())
        x = x.unsqueeze(1)
        batch_size, C, H, W, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out = self.cnn(x[...,t])
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


# In[6]:


# # define NN models
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
#         self.conv1_drop = nn.Dropout2d(p=0.8)

#     def forward(self, x):
#         #print('Conv:', x.size())
#         x = self.conv1(x)
#         # print('Conv', x.size())
        
#         x = F.relu(F.max_pool2d(x, 2))
#         # print('Pool', x.size())
#         x = x.view(-1, 3*2*3)
#         return x
    

# class CNN_LSTM(nn.Module):
#     def __init__(self, num_class, load_lstm=False, freeze_lstm=False):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = CNN()
#         self.lstm = nn.LSTM(
#             input_size=3*2*3, 
#             hidden_size=50, 
#             num_layers=2,
#             batch_first=True,
#            dropout=0.8)
        
#         if load_lstm:
#             pretrained_dict = torch.load("BioTac_info_400/model_epoch_200_slide.ckpt")
#             lstm_dict = self.lstm.state_dict()

#             # 1. filter out unnecessary keys
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in lstm_dict}
#             # 2. overwrite entries in the existing state dict
#             lstm_dict.update(pretrained_dict)
#             # 3. load the new state dict
#             self.lstm.load_state_dict(lstm_dict)
#             # Freeze model weights
#             if freeze_lstm:
#                 for param in self.lstm.parameters():
#                     param.requires_grad = False

#         self.linear = nn.Linear(50,num_class)
#         self.hidden = []
        
        
#     def init_hidden(self, h, c):
#         self.hidden = (h, c)


#     def forward(self, x, device):
#         #print(x.size())
#         batch_size, H, W, sequence_size = x.size()
#         # init hidden states
#         (h, c) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
#         self.hidden = (h.contiguous(), c.contiguous())
#         #print(batch_size*timesteps,C, H, W, sequence_size)
#         c_in = x.view(batch_size *sequence_size, 1, H, W)
#         #print(c_in.size())
#         c_out = self.cnn(c_in)
#         #print(c_out.size())
        
#         r_in = c_out.view(batch_size,sequence_size,-1)
#         self.lstm.flatten_parameters()
#         r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
#         r_out2 = self.linear(r_out[:, -1, :])

#         output = F.log_softmax(r_out2, dim=1)
#         # # check num of GPU
#         # print("\tIn Model: input size", x.size(), "output size", output.size())
#         return output


# In[7]:


model = CNN_LSTM(num_class,hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[8]:


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
    for i, (tact_icub, _,  label) in enumerate(train_loader):
        tact_icub = tact_icub.to(device)
        label = label.to(device)
        label = label.long()
        optimizer.zero_grad()
        
        #print(tact_icub.shape)
        output = model(tact_icub, device)
        
        
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
    for i, (tact_icub, _,  label) in enumerate(val_loader):
        tact_icub = tact_icub.to(device)
        label = label.to(device)
        label = label.long()

        output = model(tact_icub, device)
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
        print('Epoch: {} Loss: train {}, valid {}. Accuracy: train: {}, valid {}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))
        print('-'*20)


# In[9]:


# save stats
import pickle
all_stats = [
    epoch_train_loss,
    epoch_train_acc,
    epoch_val_loss,
    epoch_val_acc
]

pickle.dump(all_stats, open(logDir + model_name + '_stats.pkl', 'wb'))


# In[10]:


# testing set check
net_trained = CNN_LSTM(num_class,hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
net_trained.load_state_dict(torch.load(logDir + model_name + '.pt'))
net_trained.eval()


# In[11]:


# VALIDATION
correct = 0
for i, (tact_icub, _,  label) in enumerate(test_loader):
    tact_icub = tact_icub.to(device)
    label = label.to(device)
    label = label.long()

    output = model(tact_icub, device)

    # Obtain classification accuracy
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(label.data.view_as(pred)).long().cpu().sum().item()

# fill stats
test_accuracy = correct / len(test_loader.dataset)


# In[12]:


print('Test accuracy for ', str(kfold_number), ' fold : ', test_accuracy)


# In[ ]:




