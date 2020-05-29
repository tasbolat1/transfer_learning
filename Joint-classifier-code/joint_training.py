import os, sys
import math
import numpy as np
import pickle

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as data2
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import confusion_matrix


#run forward pass of the model
''' 
Design a two-stage training
firstly train two "connector" networks, secondly train the LSTM classifier
default naming: slide 1, BioTac 2
'''

class LSTM(nn.Module):
    def __init__(self, num_class, batch_size, sequence_size, upsampling_factor=None):
        super(LSTM, self).__init__()
        self.num_class = num_class
        self.batch_size = batch_size
        self.sequence_size = sequence_size

        self.lstm = nn.LSTM(
            input_size=3*2*3, 
            hidden_size=50, 
            num_layers=2,
            batch_first=True,
           dropout=0.8)
        
        self.linear = nn.Linear(50,num_class)
        self.hidden = []
        
    def init_hidden(self, h, c):
        self.hidden = (h, c)

    def forward(self, x):
        # TODO: check the input size
        r_in = x.view(self.batch_size,self.sequence_size,-1)
        print("LSTM input", x.size(), r_in.size())
        if upsampling_factor is not None:
            upsampling_factor = 6
            upsampling_idx = torch.LongTensor([i for i in range(sequence_size) if (i % upsampling_factor == 0)])
            upsampling_idx = upsampling_idx.to(device)
            r_in = r_in.index_select(1, upsampling_idx)
            print("after upsampling", r_in.size())
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)


def two_stage_framework(datafile, num_class):

    # create log dir if not existing
    logDir = 'two_stage'
    try:
        os.mkdir(logDir)
    except FileExistsError:
        print('Overwrite the touch')
    logDir = logDir + '/'

    # set parameters
    batch_size  = 21 # TODO: check batch_size, slide-23, Bio-21
    num_epochs = 200
    num_train_inner_feat = 10
    log_interval = 10
    save_interval  = 10
    seq_len = 75
    
    # configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load datasets
    print("loading data...")
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open(datafile, 'rb'))
    
    # load slide dataset
    DATA_DIR1 = 'slide_6_10/'
    training_dataset1 = Dataset(train_ids, train_labels, DATA_DIR1)
    test_dataset1 = Dataset(test_ids, test_labels, DATA_DIR1)
    train_loader1 = data2.DataLoader(training_dataset1, batch_size=batch_size)
    test_loader1 = data2.DataLoader(test_dataset1, batch_size=batch_size)


    # load BioTac dataset
    DATA_DIR2 = 'tactile_img_data_smooth/'
    training_dataset2 = Dataset(train_ids, train_labels, DATA_DIR2)

    test_dataset2 = Dataset(test_ids, test_labels, DATA_DIR2)

    train_loader2 = data2.DataLoader(training_dataset2, batch_size=batch_size)
    test_loader2 = data2.DataLoader(test_dataset2, batch_size=batch_size)

    # set model
    model1 = CNN().to(device)
    model2 = BioCNN().to(device)
    # TODO: do upsampling before the dataset, cuz they share the same init of model3
    model3 = LSTM(num_class, batch_size, seq_len, upsampling_factor=None).to(device)
    print(model1)
    print(model2)
    print(model3)
    

    # setup optimizer. TODO: add scheduler
    learning_rate = 0.0001 # 0.001
    optim1 = optim.Adam(model1.parameters(), lr=learning_rate)
    optim2 = optim.Adam(model2.parameters(), lr=learning_rate)
    optim3 = optim.Adam(model3.parameters(), lr=learning_rate)

    # set initial hidden state
    (h_ini, c_ini) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    len_train = min(len(train_loader1), len(train_loader2)) # take min first, try side by side later
    # (i.e. when one loader is used up, set its loss to 0 and continue to train the other)

    for epoch in range(1, num_epochs + 1):
        # TODO: continue from here
        iter_count = 0
        while iter_count < len_train:
            # first stage, train the two "connector" network
            inner_iter_count = 0
            while inner_iter_count < num_train_inner_feat:
                for batch_idx in range(1, len_train+1):
                    data1, target1 = train_loader1_common[batch_idx]
                    data1 = torch.FloatTensor(np.expand_dims(data1, axis=1))
                    if target1.size()[0] != batch_size:
                        # print("epoch {}, batch {} size {} does not match {}, skip".format(epoch, batch_idx, target.size()[0], batch_size))
                        continue
                    data1, target1 = data1.to(device), target1.to(device)
                    feat1 = model1(data)

                    data2, target2 = train_loader2_common[batch_idx]
                    data2 = torch.FloatTensor(np.expand_dims(data2, axis=1))
                    if target2.size()[0] != batch_size:
                        # print("epoch {}, batch {} size {} does not match {}, skip".format(epoch, batch_idx, target.size()[0], batch_size))
                        continue
                    data2, target2 = data2.to(device), target2.to(device)
                    feat2 = model2(data)

                    optim1.zero_grad()
                    optim2.zero_grad()
                    # compute the difference between two features as the loss_s1
                    loss_s1 = F.mse_loss(feat1, feat2)
                    loss_s1.backward()
                    optim1.step() # update theta1
                    optim2.step() # update theta2
                    inner_iter_count += 1



            # second stage, train the LSTM


            # fix parameters of two connector networks
            for p in model1.parameters():
                p.requires_grad = False  # to avoid computation
            for p in model2.parameters():
                p.requires_grad = True  # to avoid computation
            netF.zero_grad()


            model3.init_hidden(h_ini, c_ini)
            
            feat1 = model1(data)
            output1 = model3(feat1)
            loss1 = F.nll_loss(output1, target1)
            
            feat2 = model2(data)
            output2 = model3(feat2)
            loss2 = F.nll_loss(output2, target2)

            loss1 = F.nll_loss(output1, target1)
            loss2 = F.nll_loss(output2, target2)
            loss = loss1 + loss2

            optim3.zero_grad()
            loss.backward()
            optim3.step() #update theta3

            iter_count += 1





# main function
datafile = 'BioTac.pkl'
num_class = 21
print("num_class", num_class)
two_stage_framework(datafile, num_class)
