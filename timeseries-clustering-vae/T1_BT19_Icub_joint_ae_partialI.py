#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
python T1_BT19_Icub_joint_ae_partialI.py -k 0 -c 0 -r 1 -rr 3

Note: still use one-stage training with both recon loss and classification loss
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


# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
parser.add_argument("-r", "--reduction", type=int, default=1, help="data reduction ratio for partial training")
parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
parser.add_argument("-rr", "--removal", type=int, default=0, help="number of batches removed from training")
args = parser.parse_args()


# In[3]:


# # dummy class to replace argparser
# class Args:
#   kfold = 0
#   reduction = 1
#   cuda = '2'
#   removal = 2

# args=Args()


# In[4]:


if args.reduction != 1:
    print("load {} kfold number, reduce data to {} folds, put to cuda:{}".format(args.kfold, args.reduction, args.cuda))
    assert args.removal == 0, "removal must be 0 for kfold reduction"
elif args.removal != 0:
    print("load {} kfold number, remove {} batch of training data, put to cuda:{}".format(args.kfold, args.removal, args.cuda))
else:
    print("load {} kfold number, put to cuda:{}, train with full data".format(args.kfold, args.cuda))

# Set hyper params
kfold_number = args.kfold
data_reduction_ratio = args.reduction
removal = args.removal
shuffle = False
num_class = 20
sequence_length_B = 400
sequence_length_I = 75
number_of_features_B = 19
number_of_features_I = 60

hidden_size = 90
hidden_layer_depth = 1
latent_length = 40
batch_size = 32
learning_rate = 0.0005
n_epochs = 2000

dropout_rate = 0.2
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
header_B = None
header_I = "CNN"

w_mse = 1 # mse between latent vectors
w_rB = 0.01 # recon for B
w_rI = 0.01 # recon for I
w_cB = 1 # classify for B
w_cI = 1 # classify for I


np.random.seed(1)
torch.manual_seed(1)


# ### Load data and preprocess

# In[5]:


# Load data
data_dir = '../../new_data_folder/'

logDir = 'models_and_stat/'
# new model
model_name_B = 'BT19_joint_ae_wrB_{}_wcB_{}_wrI_{}_wcI_{}_wC_{}_reductI_{}_rm_{}_{}'.format(w_rB,w_cB, w_rI, w_cI, w_mse, data_reduction_ratio, removal, str(kfold_number))
model_name_I = 'IcubCNN_joint_ae_wrB_{}_wcB_{}_wrI_{}_wcI_{}_wC_{}_reductI_{}_rm_{}_{}'.format(w_rB,w_cB, w_rI, w_cI, w_mse, data_reduction_ratio, removal, str(kfold_number))

device = torch.device("cuda:{}".format(args.cuda))
print("Loading data...")

train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False, batch_size=batch_size, shuffle=shuffle)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False, batch_size=batch_size, shuffle=shuffle)


# In[6]:


model_B = VRAEC(num_class=num_class,
            sequence_length=sequence_length_B,
            number_of_features = number_of_features_B,
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
            model_name=model_name_B,
            header=header_B,
            device = device)
model_B.to(device)

model_I = VRAEC(num_class=num_class,
            sequence_length=sequence_length_I,
            number_of_features = number_of_features_I,
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
            model_name=model_name_I,
            header=header_I,
            device = device)
model_I.to(device)


# In[7]:


criterion = nn.CrossEntropyLoss()
optimB = optim.Adam(model_B.parameters(), lr=learning_rate)
optimI = optim.Adam(model_I.parameters(), lr=learning_rate)
cl_loss_fn = nn.NLLLoss()
recon_loss_fn = nn.MSELoss()


# In[8]:


# one stage training: with recon_loss and mse_loss
training_start=datetime.now()

epoch_train_loss_B = []
epoch_train_acc_B = []
epoch_val_loss_B = []
epoch_val_acc_B = []
max_val_acc_B = 0

epoch_train_loss_I = []
epoch_train_acc_I = []
epoch_val_loss_I = []
epoch_val_acc_I = []

epoch_train_tot_loss = []
epoch_val_tot_loss = []
max_val_acc_I = 0

for epoch in range(n_epochs):

    # TRAIN
    model_B.train()
    model_I.train()

    correct_B = 0
    train_loss_B = 0
    correct_I = 0
    train_loss_I = 0
    train_loss_tot = 0
    train_num_B = 0
    train_num_I = 0
            
    for i, (XI, XB,  y) in enumerate(train_loader):
        
        if i >= len(train_loader)-removal:
            break
        XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)
        
        if XI.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break

        train_num_B += XB.size(0)
        
        # train model_B
        optimB.zero_grad()  
        x_decoded_B, latent_B, output = model_B(XB)
        
        # construct loss function
        recon_loss_B = recon_loss_fn(x_decoded_B, XB)
        cl_loss_B = cl_loss_fn(output, y)
        loss_B = w_rB*recon_loss_B + w_cB*cl_loss_B
        
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        train_loss_B += loss_B.item()
        
        # train partially on I 
        if i % data_reduction_ratio == 0:
            train_num_I += XI.size(0)
            # train modelI
            optimI.zero_grad()  
            x_decoded_I, latent_I, output = model_I(XI)

            # construct loss function
            recon_loss_I = recon_loss_fn(x_decoded_I, XI)
            cl_loss_I = cl_loss_fn(output, y)
            loss_I = w_rI*recon_loss_I + w_cI*cl_loss_I

            # compute classification acc
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
            # accumulator
            train_loss_I += loss_I.item()
        
            loss_C = F.mse_loss(latent_B, latent_I)
            loss = loss_B + loss_I + w_mse*loss_C

            if epoch < 20:
                loss_B.backward()
                loss_I.backward()
            else:
                loss.backward()

            optimB.step() 
            optimI.step() 
        
        else:
            # only train model B
            loss = loss_B
            loss.backward()
            optimB.step()
            
        train_loss_tot += loss.item()
        
        
    if epoch < 20 or epoch%200 == 0:
        print("last batch training: LB: {:.2f}, LI: {:.2f}, LC: {:.2f} \n recon_B {:.2f}, cl_B {:.2f}, recon_I {:.2f}, cl_I {:.2f}"              .format(loss_B, loss_I, loss_C, recon_loss_B, cl_loss_B, recon_loss_I, cl_loss_I))
    
    
    # fill stats
    train_accuracy_B = correct_B / train_num_B # len(train_loader.dataset)
    train_loss_B /= train_num_B #len(train_loader.dataset)
    epoch_train_loss_B.append(train_loss_B)
    epoch_train_acc_B.append(train_accuracy_B) 
    
    train_accuracy_I = correct_I / train_num_I # len(train_loader.dataset)
    train_loss_I /= train_num_I #len(train_loader.dataset)
    epoch_train_loss_I.append(train_loss_I)
    epoch_train_acc_I.append(train_accuracy_I) 
    

    # VALIDATION
    model_B.eval()
    model_I.eval()

    correct_B = 0
    val_loss_B = 0
    correct_I = 0
    val_loss_I = 0
    val_loss_tot = 0
    val_num = 0

    for i, (XI, XB,  y) in enumerate(val_loader):
        XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)
        
        if XI.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
            break

        val_num += XI.size(0)
        
        # eval model_B
        x_decoded_B, latent_B, output = model_B(XB)
        # construct loss function
        recon_loss_B = recon_loss_fn(x_decoded_B, XB)
        cl_loss_B = cl_loss_fn(output, y)
        loss_B = w_rB*recon_loss_B + w_cB*cl_loss_B
       
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        val_loss_B += loss_B.item()
        
        
        # eval modelI 
        x_decoded_I, latent_I, output = model_I(XI)
        # construct loss function
        recon_loss_I = recon_loss_fn(x_decoded_I, XI)
        cl_loss_I = cl_loss_fn(output, y)
        loss_I = w_rI*recon_loss_I + w_cI*cl_loss_I

        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        val_loss_I += loss_I.item()
        
        loss_C = F.mse_loss(latent_B, latent_I)
        loss = loss_B + loss_I + w_mse*loss_C
        val_loss_tot += loss.item()

    # fill stats
    val_accuracy_B = correct_B / val_num # len(train_loader.dataset)
    val_loss_B /= val_num #len(train_loader.dataset)
    epoch_val_loss_B.append(val_loss_B)
    epoch_val_acc_B.append(val_accuracy_B) 
    
    val_accuracy_I = correct_I / val_num # len(train_loader.dataset)
    val_loss_I /= val_num #len(train_loader.dataset)
    epoch_val_loss_I.append(val_loss_I)
    epoch_val_acc_I.append(val_accuracy_I) 
    
    if epoch < 20 or epoch%200 == 0:
#         print("train_num {}, val_num {}".format(train_num, val_num))
        print("Epoch {}: Loss: lc {:.3f},  train_B {:.3f}, val_B {:.3f}, train_I{:.3f}, val_I{:.3f}, \n\t\t Acc: train_B {:.3f}, val_B {:.3f}, train_I {:.3f}, val_I {:.3f}"              .format(epoch, loss_C, train_loss_B, val_loss_B, train_loss_I, val_loss_I, train_accuracy_B, val_accuracy_B, train_accuracy_I, val_accuracy_I))
        print("-"*20)
    # choose model
    # TODO: not save at the same time, may have bad common representation
    if max_val_acc_B <= val_accuracy_B:
        model_dir = logDir + model_name_B + '.pt'
        print("Saving model at {} epoch to {}".format(epoch, model_dir))
        max_val_acc_B = val_accuracy_B
        torch.save(model_B.state_dict(), model_dir)

    if max_val_acc_I <= val_accuracy_I:
        model_dir = logDir + model_name_I + '.pt'
        print("Saving model at {} epoch to {}".format(epoch, model_dir))
        max_val_acc_I = val_accuracy_I
        torch.save(model_I.state_dict(), model_dir)
    

training_end =  datetime.now()
training_time = training_end -training_start 
print("RAE training takes time {}".format(training_time)) 


# In[9]:


model_B.is_fitted = True
model_I.is_fitted = True

model_B.eval()
model_I.eval()


# model_B_trained = VRAEC(num_class=num_class,
#             sequence_length=sequence_length_B,
#             number_of_features = number_of_features_B,
#             hidden_size = hidden_size, 
#             hidden_layer_depth = hidden_layer_depth,
#             latent_length = latent_length,
#             batch_size = batch_size,
#             learning_rate = learning_rate,
#             n_epochs = n_epochs,
#             dropout_rate = dropout_rate, 
#             cuda = cuda,
#             print_every=print_every, 
#             clip=clip, 
#             max_grad_norm=max_grad_norm,
#             dload = logDir,
#             model_name=model_name_B,
#             header=header_B,
#             w_r = w_r, 
#             w_k = w_k, 
#             w_c = w_c,
#             device = device)
# model_B_trained.load_state_dict(torch.load(logDir + model_name_B + '.pt'))
# model_B_trained.to(device)
# model_B_trained.eval()
# 
# model_I_trained = VRAEC(num_class=num_class,
#             sequence_length=sequence_length_I,
#             number_of_features = number_of_features_I,
#             hidden_size = hidden_size, 
#             hidden_layer_depth = hidden_layer_depth,
#             latent_length = latent_length,
#             batch_size = batch_size,
#             learning_rate = learning_rate,
#             n_epochs = n_epochs,
#             dropout_rate = dropout_rate, 
#             cuda = cuda,
#             print_every=print_every, 
#             clip=clip, 
#             max_grad_norm=max_grad_norm,
#             dload = logDir,
#             model_name=model_name_I,
#             header=header_I,
#             w_r = w_r, 
#             w_k = w_k, 
#             w_c = w_c,
#             device = device)
# model_I_trained.load_state_dict(torch.load(logDir + model_name_I + '.pt'))
# model_I_trained.to(device)
# model_I_trained.eval()

# In[10]:


# TEST
correct_B = 0
correct_I = 0
test_num = 0

for i, (XI, XB,  y) in enumerate(test_loader):
    XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)

    if XI.size()[0] != batch_size:
#             print("batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
        break

    test_num += XI.size(0)

    # test model_B
    x_decoded_B, latent_B, output = model_B(XB)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

    
    # test modelI 
    x_decoded_I, latent_I, output = model_I(XI)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

test_acc_B = correct_B/test_num
test_acc_I = correct_I/test_num
print('Test accuracy for {} fold {} samples: B {}, I {}'.format(str(kfold_number),test_num, test_acc_B, test_acc_I))


# In[11]:


# plot_clustering(z_runs[0], y_val[0], engine='matplotlib', download = True, folder_name='figures', filefix='_BT19_joint_{}'.format(n_epochs))


# In[12]:


# plot_clustering(z_runs[1], y_val[1], engine='matplotlib', download = True, folder_name='figures', filefix='_Icub_joint_{}'.format(n_epochs))


# In[13]:


# save stats
results_dict = {"epoch_train_loss_B": epoch_train_loss_B,
                "epoch_train_loss_I": epoch_train_loss_I,
                "epoch_val_loss_B": epoch_val_loss_B,
                "epoch_val_loss_I": epoch_val_loss_I,
                "epoch_train_acc_B": epoch_train_acc_B,
                "epoch_train_acc_I": epoch_train_acc_I,
                "epoch_val_acc_B": epoch_val_acc_B,
                "epoch_val_acc_I": epoch_val_acc_I,
                "test_acc": [test_acc_B, test_acc_I]}
dict_name = "BT19Icub_joint_ae_reductI_{}_rm_{}_wrB_{}_wcB_{}_wrI_{}_wcI_{}_wC_{}_{}.pkl".format(data_reduction_ratio, removal, w_rB, w_cB, w_rI, w_cI, w_mse, str(kfold_number))
pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
print("dump results dict to {}".format(dict_name))


# ### plot the train acc

# In[14]:


assert n_epochs == len(epoch_train_acc_B), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc_B))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc_B, label="train acc B")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name_B+"_train_acc.png"
plt.savefig(figname)
plt.show()


# In[15]:


assert n_epochs == len(epoch_train_acc_I), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc_I))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc_I, label="train acc I")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name_I + "_train_acc.png"
plt.savefig(figname)
plt.show()


# In[ ]:




