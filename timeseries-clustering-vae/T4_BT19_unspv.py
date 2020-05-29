#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
python T4_BT19_unspv.py -k 0 -c 0
'''

# Import

import os,sys
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../new_iteration/")
import pickle
import argparse
from sklearn.metrics import confusion_matrix
from datetime import datetime

from vrae.vrae import VRAEC
from vrae.utils import *
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data2
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tas_utils_bs import get_trainValLoader, get_testLoader
import plotly


# In[2]:


# # Parse argument
# parser = argparse.ArgumentParser()
# parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
# parser.add_argument("-r", "--reduction", type=int, default=1, help="data reduction ratio for partial training")
# parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
# args = parser.parse_args()


# In[3]:


# dummy class to replace argparser
class Args:
  kfold = 0
  reduction = 1
  cuda = '3'

args=Args()


# In[4]:


print("load {} kfold number, reduce data to {} folds, put to cuda:{}".format(args.kfold, args.reduction, args.cuda))

# Set hyper params
kfold_number = args.kfold
data_reduction_ratio = args.reduction
shuffle = False
num_class = 20
sequence_length = 400
number_of_features = 19

hidden_size = 90
hidden_layer_depth = 1
latent_length = 40
batch_size = 32
learning_rate = 0.0005
n_epochs = 2000

learning_rate_2 = 0.01
n_epochs_2 = 2000

dropout_rate = 0.2
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
header = None
w_r = 0.01
w_c = 1
np.random.seed(1)
torch.manual_seed(1)


# In[5]:


# Load data
data_dir = '../../new_data_folder/'
kfold_number = 0

logDir = 'models_and_stat/'
model_name = 'BT19_unspv_wrI_{}_wC_{}_{}'.format(w_r, w_c, str(kfold_number))
device = torch.device("cuda:{}".format(args.cuda))
print("Loading data...")
train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False, batch_size=batch_size, shuffle=shuffle)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False, batch_size=batch_size, shuffle=shuffle)


# In[6]:


# Create model
model = VRAEC(num_class=num_class,
            sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            dload = logDir,
            model_name=model_name,
            header=header,
            device = device)

model.to(device)


# In[7]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
cl_loss_fn = nn.NLLLoss()
recon_loss_fn = nn.MSELoss()


# In[8]:


# 1st stage training: with recon_loss
training_start=datetime.now()
#split fit
epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []
max_val_acc = 0

for epoch in range(n_epochs):
    
    # TRAIN
    model.train()
    correct = 0
    train_loss = 0
    train_num = 0
    for i, (XI, XB,  y) in enumerate(train_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        train_num += x.size(0)
        optimizer.zero_grad()
  
        x_decoded, latent, output = model(x)

        # construct loss function
        loss = recon_loss_fn(x_decoded, x)
        loss.backward()
        optimizer.step()
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

        # accumulator
        train_loss += loss.item()

    # fill stats
    if epoch < 20 or epoch%200 == 0:
        print("train last batch: recon_loss {}".format(loss))
    train_accuracy = correct / train_num 
    train_loss /= train_num
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_accuracy) 
    
    # VALIDATION
    model.eval()
    correct = 0
    val_loss = 0
    val_num = 0
    for i, (XI, XB,  y) in enumerate(val_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        val_num += x.size(0)
        x_decoded, latent, output = model(x)

        # construct loss function
        loss = recon_loss_fn(x_decoded, x)
        
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        
        # accumulator
        val_loss += loss.item()
    
#     print("test last batch: recon_loss {}, kl_loss {}, cl_loss {}".format(recon_loss, kl_loss, cl_loss))
    # fill stats
    val_accuracy = correct / val_num# / len(val_loader.dataset)
    val_loss /= val_num #len(val_loader.dataset)

    epoch_val_loss.append(val_loss)  # only save the last batch
    epoch_val_acc.append(val_accuracy)
    
    if epoch < 20 or epoch%200 == 0:
        print('Epoch: {} Loss: train {}, valid {}. Accuracy: train: {}, valid {}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))
    
    # choose model
    if max_val_acc <= val_accuracy:
        model_dir = logDir + model_name + '.pt'
        print('Saving model at {} epoch to{}'.format(epoch, model_dir))
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), model_dir)
        
training_end =  datetime.now()
training_time = training_end -training_start 
print("training takes time {}".format(training_time))


# In[9]:


# 2nd stage training: train classifier using classification loss
epoch_cl_train_loss = []
epoch_cl_train_acc = []
epoch_cl_val_loss = []
epoch_cl_val_acc = []
max_val_acc = 0

optimizer = optim.Adam(model.parameters(), lr=learning_rate_2)

for epoch in range(n_epochs_2):
    
    # TRAIN
    model.train()
    correct = 0
    train2_loss = 0
    train_num = 0
    
    # freeze params except for the classifier
    trained_names = ['classifier.0.bias', 'classifier.0.weight']
    for name, param in model.named_parameters():
        if name in trained_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for i, (XI, XB,  y) in enumerate(train_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        train_num += x.size(0)
        
        optimizer.zero_grad()
        x_decoded, latent, output = model(x)

        # construct loss function
        loss = cl_loss_fn(output, y)
        loss.backward()
        optimizer.step()
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

        # accumulator
        train2_loss += loss.item()

    # fill stats
    if epoch < 20 or epoch%200 == 0:
        print("train last batch:  cl_loss {}".format(loss))
    train2_accuracy = correct / train_num # len(train_loader.dataset)
    train2_loss /= train_num #len(train_loader.dataset)
    epoch_cl_train_loss.append(train2_loss)
    epoch_cl_train_acc.append(train2_accuracy) 
    
    # VALIDATION
    model.eval()
    correct = 0
    val2_loss = 0
    val_num = 0
    for i, (XI, XB,  y) in enumerate(val_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break
        val_num += x.size(0)
        x_decoded, latent, output = model(x)

        # construct loss function
        loss = cl_loss_fn(output, y)
    
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        
        # accumulator
        val2_loss += loss.item()

    # fill stats
    val2_accuracy = correct / val_num# / len(val_loader.dataset)
    val2_loss /= val_num #len(val_loader.dataset)

    epoch_cl_val_loss.append(val2_loss)  # only save the last batch
    epoch_cl_val_acc.append(val2_accuracy)
    if epoch < 20 or epoch%200 == 0:
        print('Epoch: {} Loss: train {}, valid {}. Accuracy: train: {}, valid {}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))
    
    # choose model
    if max_val_acc <= val_accuracy:
        model_dir = logDir + model_name + '.pt'
        print('Saving model at {} epoch to{}'.format(epoch, model_dir))
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), model_dir)


# In[10]:


model.is_fitted = True
model.eval()


# In[11]:


# testing set check
net_trained = VRAEC(num_class=num_class,
            sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            dload = logDir,
            model_name=model_name,
            header=header,
            device = device)

net_trained.load_state_dict(torch.load(logDir + model_name + '.pt'))
net_trained.eval()


# In[12]:


correct = 0
test_num = 0
for i, (XI, XB,  y) in enumerate(test_loader):
    if model.header == 'CNN':
        x = XI
    else:
        x = XB
    x, y = x.to(device), y.long().to(device)
    
    if x.size(0) != batch_size:
        print(" test batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
        break
    test_num += x.size(0)
    x_decoded, latent, output = model(x)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
    
test_acc = correct / test_num #len(test_loader.dataset)

print('Test accuracy for', str(kfold_number), ' fold : ', test_acc)


# In[13]:


# save stats
all_stats = {"epoch_train_loss": epoch_train_loss,
             "epoch_train_acc": epoch_train_acc,
             "epoch_val_loss": epoch_val_loss,
             "epoch_val_acc": epoch_val_acc,
             "epoch_cl_train_loss":epoch_cl_train_loss,
             "epoch_cl_train_acc": epoch_cl_train_acc,
             "epoch_cl_val_loss": epoch_cl_val_loss,
             "epoch_cl_val_acc": epoch_cl_val_acc,
             "test_acc": test_acc}

dict_name = logDir + model_name + '_stats.pkl'
pickle.dump(all_stats, open(dict_name, 'wb'))


# In[14]:


assert n_epochs == len(epoch_train_acc), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc, label="train acc")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name +"_train1_acc.png"
plt.savefig(figname)
plt.show()


# In[15]:


# print(epoch_cl_train_acc)


# In[16]:


assert n_epochs_2 == len(epoch_cl_train_acc), "different epoch length {} {}".format(n_epochs_2, len(epoch_cl_train_acc))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs_2), epoch_cl_train_acc, label="train acc")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name +"_train2_acc.png"
plt.savefig(figname)
plt.show()


# In[ ]:




