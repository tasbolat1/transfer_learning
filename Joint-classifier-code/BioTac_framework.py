import os, sys
import math
import numpy as np
import pickle
import argparse

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

class Dataset(data2.Dataset):
    def __init__(self, data_dir, list_IDs, labels, cut_idx=None):
        # initialize
        self.data_dir = data_dir
        self.labels = labels
        self.list_IDs = list_IDs
        self.cut_idx = cut_idx
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load(self.data_dir + ID + '.pt')
        X.unsqueeze_(0)
        if self.cut_idx is None:
            # use original len
            self.cut_idx = X.size()[1]
        # print("cut_idx ", self.cut_idx)
        X = X[:,:self.cut_idx,:,:]
        # print("in getitem", X.size()) # (1, 400, 8, 5)
        data_nan = torch.isnan(X)==1
        if True in data_nan:
            print("nan data", self.data_dir + ID)
            result = np.where(data_nan==True)
            print(result)
        y = self.labels[ID]

        return X, y
    
    def get_X(self):
        X = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            x = torch.load(self.data_dir + ID + '.pt')
            # convert to np array
            X.append(x.numpy())
        return np.array(X)
    
    def get_y(self):
        y = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            y.append(self.labels[ID])
        return np.array(y)


# define NN models
class CNN(nn.Module):
    def __init__(self, seq_len):
        super(CNN, self).__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
        self.conv1_drop = nn.Dropout2d(p=0.8)

    def forward(self, x):
        # print("begin cnn")
        # print('Conv:', x.size())
        x = self.conv1(x)
        # print('Conv', x.size())
        x = F.relu(F.max_pool2d(x, 2))
        # print('Pool', x.size())
        x = x.view(-1, 3*2*3)
        return x
    

class CNN_LSTM(nn.Module):
    def __init__(self, seq_len, num_class):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(seq_len)
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
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x):
        
        # print(x.size())
        batch_size, timesteps, C, H, W, sequence_size = x.size()
        #print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        #print(c_in.size())
        
        c_out = self.cnn(c_in)
        #print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)

# define NN models
class BioCNN(nn.Module):
    def __init__(self):
        super(BioCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv1_drop = nn.Dropout2d(p=0.8)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.hidden1 = nn.Linear(64*4, 18)
        # self.hidden2 = nn.Linear(1024, 18)

    def forward(self, x):
        # print("begin BioCNN")
        # print('before Conv:', x.size()) # [12600, 1, 8, 5]

        x = self.act(self.conv1(x)) # ([12600, 32, 6, 3])
        # print(x.size())
        x = self.act(self.conv2(x)) # ([12600, 64, 4, 1])
        # print(x.size())
        x = x.view(x.size(0), -1) # flatten ([12600, 256])
        # print(x.size())
        x = self.act(self.hidden1(x)) # [batch_size*seq, 1024] ([12600, 1024])
        # x = self.act(self.hidden2(x))

        # print(x.size())

        return x


class BioCNN_LSTM(nn.Module):
    def __init__(self, num_class, load_lstm=False, freeze_lstm=False):
        super(BioCNN_LSTM, self).__init__()
        self.cnn = BioCNN()
        self.lstm = nn.LSTM(
            input_size=3*2*3, 
            hidden_size=50, 
            num_layers=2,
            batch_first=True,
           dropout=0.8)

        if load_lstm:
            pretrained_dict = torch.load("BioTac_info_400/model_epoch_200_slide.ckpt")
            lstm_dict = self.lstm.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in lstm_dict}
            # 2. overwrite entries in the existing state dict
            lstm_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.lstm.load_state_dict(lstm_dict)
            # Freeze model weights
            if freeze_lstm:
                for param in self.lstm.parameters():
                    param.requires_grad = False
        
        self.linear = nn.Linear(50,num_class)
        self.hidden = []

        
    def init_hidden(self, h, c):
        self.hidden = (h, c)
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x, h_ini, c_ini):
        # print("enter BioCNN_LSTM")
        batch_size, timesteps, C, sequence_size, H, W = x.size()
        # init hidden states
        # h = h_ini.transpose(0, 1)
        # c = c_ini.transpose(0, 1)
        (h, c) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
        self.hidden = (h.contiguous(), c.contiguous())    
        # print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        # print("c_in", c_in.size())
        c_out = self.cnn(c_in)
        # print("c_out", c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        # print("r_in", r_in.size())
        # # upsampling the BioImages
        # upsampling_factor = 6
        # upsampling_idx = torch.LongTensor([i for i in range(sequence_size) if (i % upsampling_factor == 0)])
        # # print("upsampling")
        # upsampling_idx = upsampling_idx.to(device)
        # r_in = r_in.index_select(1, upsampling_idx)
        # # print(r_in.size())

        # # (2.4) cut the leading and tailing signal since they introduce unrelated acceleration and deceleration info
        #  r_cut = r_in[:,150:-150,:]
        
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        output = F.log_softmax(r_out2, dim=1)
        # check num of GPU
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, 100),
            nn.ReLU(),
            nn.Linear(100, dim_out)
        )
        
    def forward(self, x):
        print("in cnn", x.size())
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.layers(x)
        print(x.size())
        return x


class MLP_LSTM(nn.Module):
    def __init__(self, L, num_class, load_lstm=False, freeze_lstm=False):
        super(MLP_LSTM, self).__init__()
        LSTM_in = 18 # 3*2*3, same as CNN-LSTM
        self.mlp = MLP(L, LSTM_in) # process one timestamp data
        self.lstm = nn.LSTM(
            input_size=LSTM_in, 
            hidden_size=50, 
            num_layers=2,
            batch_first=True,
           dropout=0.8)

        if load_lstm:
            pretrained_dict = torch.load("BioTac_info_400/model_epoch_200_slide.ckpt")
            print("pretrained_dict")
            print(pretrained_dict)
            lstm_dict = self.lstm.state_dict()
            print("lstm_dict")
            print(lstm_dict)

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in lstm_dict}
            # 2. overwrite entries in the existing state dict
            lstm_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.lstm.load_state_dict(lstm_dict)
            # # Freeze model weights
            # if freeze_lstm:
            #     for param in self.lstm.parameters():
            #         param.requires_grad = False
            

        
        self.linear = nn.Linear(50,num_class)
        self.hidden = []
        
        
    def init_hidden(self, h, c):
        self.hidden = (h, c)
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x):
        
        print(x.size()) # torch.Size([21, 1, 1, 600, 44]) H,W => length
        batch_size, timesteps, C, sequence_size, L = x.size()
        c_in = x.view(batch_size * timesteps*sequence_size, C, L)
        #print(c_in.size())
        
        c_out = self.mlp(c_in)
        #print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BioTac_learn(data_dir, datafile, num_class, args, iteration=400):
    
    dirName = 'BioTac_info_' + str(iteration)
    try:
        # Create target Directory
        os.mkdir(dirName)
    except FileExistsError:
        print('Overwrite the touch')
    dirName = dirName + '/'
    
    # set parameters
    batch_size= 32
    num_epochs = 2000
    learning_rate = 0.0001 # 0.001
    log_interval = 10
    save_interval  = 100
    seq_len = 75
    n_epochs_stop = 4
    cut_idx = args.cutidx
    load_lstm = args.lstm
    print("cut the data by first {} timesteps".format(cut_idx))
    print("load lstm? ", load_lstm)

    # load data
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open(datafile, 'rb'))
    training_dataset = Dataset(data_dir, train_ids, train_labels, cut_idx=cut_idx)
    test_dataset = Dataset(data_dir, test_ids, test_labels, cut_idx=cut_idx)
    train_loader = data2.DataLoader(training_dataset, batch_size=batch_size)
    test_loader = data2.DataLoader(test_dataset, batch_size=batch_size)

    model = BioCNN_LSTM(num_class, load_lstm=load_lstm)
    # enable multiple GPU (DataParallel)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # STEP 7: INSTANTIATE STEP LEARNING SCHEDULER CLASS
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=4, verbose=True)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.96)
    
    # print("init hidden state", h_ini.size())  #torch.Size([2, 21, 50])
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_loss = []
    epoch_test_acc = []
    model.train()

    is_nan = False
    min_val_loss = np.Inf

    num_gpu = torch.cuda.device_count()
    # set initial hidden state
    # (h_ini, c_ini) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
    (h_ini, c_ini) = (torch.zeros(batch_size, 2, 50).to(device) , torch.zeros(batch_size, 2, 50).to(device))

    for epoch in range(1, num_epochs + 1):
        model.train()
        # scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = np.expand_dims(data, axis=1)
            data = torch.FloatTensor(data) 
            # print("train epoch %d, batch %d " %(epoch, batch_idx), target.size())
            # if target.size()[0] != batch_size:
            #     print("epoch {}, batch {} size {} does not match {}, skip".format(epoch, batch_idx, target.size()[0], batch_size))
            #     continue
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            
            output = model(data, h_ini, c_ini)

            if 1 in torch.isnan(output):
                is_nan = True
                # print("nan values in output")
                data_nan = torch.isnan(data)==1
                # print(data_nan)
   
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # if batch_idx % 10 == 0:
            #     # Calculate Accuracy for step scheduler       
            #     correct = 0
            #     total = 0
            #     test_loss = 0
            #     # Iterate through test dataset
            #     for data, target in test_loader:
            #         model.eval()
            #         with torch.no_grad():
            #             # Load data to a Torch Variable
            #             data = np.expand_dims(data, axis=1)
            #             data = torch.FloatTensor(data)        
            #             data, target = data.to(device), target.to(device)

            #             # Forward pass only to get logits/output
            #             output = model(data, h_ini, c_ini)

            #             # Get predictions from the maximum value
            #             _, predicted = torch.max(output.data, 1)

            #             # Total number of labels
            #             total += target.size(0)

            #             test_loss += criterion(
            #                 output, target).item()  # sum up batch loss
            #             pred = output.data.max(
            #                 1, keepdim=True)[1]  # get the index of the max log-probability
            #             correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            #     accuracy = 100 * correct / total

        # # add early stopping
        # if loss < min_val_loss:
        #      epochs_no_improve = 0
        #      min_val_loss = loss
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve == n_epochs_stop:
        #     print('Early stopping!' )
        #     break
        
        # Decay Learning Rate, pass validation accuracy for tracking at every epoch
        # print('Epoch {} completed, with lr {}'.format(epoch, scheduler.get_lr())) 
        # scheduler.step(accuracy)

        # (2.2) validation for each epoch
        model.eval()

        # Calculate Train Accuracy         
        correct = 0
        total = 0.
        train_loss = 0
        train_nll_loss = []
        # Iterate through test dataset
        for data, target in train_loader:
            if target.size()[0] != batch_size:
                print("batch size {} does not match, skip".format(target.size()[0]))
                continue
            with torch.no_grad():
                # Load data to a Torch Variable
                data = np.expand_dims(data, axis=1)
                data = torch.FloatTensor(data)        
                data, target = data.to(device), target.to(device)
                output = model(data, h_ini, c_ini)

                # print("Outside: input size ", data.size(), "output size", output.size())

                # Obtain nll_loss, keep consistent criteria with training
                train_nll_loss.append(F.nll_loss(output, target).item()) 

                # Obtain classification accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0) # Total number of labels

                train_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(
                    1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        train_accuracy = 100 * correct / total
        # test_loss /= len(test_loader.dataset)
        train_loss = np.mean(np.array(train_nll_loss))
        epoch_train_loss.append(loss.item())  # only save the last batch
        epoch_train_acc.append(train_accuracy)


        # Calculate Test Accuracy         
        correct = 0
        total = 0.
        test_loss = 0
        test_nll_loss = []
        # Iterate through test dataset
        for data, target in test_loader:
            if target.size()[0] != batch_size:
                print("batch size {} does not match, skip".format(target.size()[0]))
                continue
            with torch.no_grad():
                # Load data to a Torch Variable
                data = np.expand_dims(data, axis=1)
                data = torch.FloatTensor(data)        
                data, target = data.to(device), target.to(device)
                output = model(data, h_ini, c_ini)

                # Obtain nll_loss, keep consistent criteria with training
                test_nll_loss.append(F.nll_loss(output, target).item()) 

                # Obtain classification accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0) # Total number of labels

                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(
                    1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        test_accuracy = 100 * correct / total
        # test_loss /= len(test_loader.dataset)
        test_loss = np.mean(np.array(test_nll_loss))
        epoch_test_loss.append(test_loss)
        epoch_test_acc.append(test_accuracy)

        print('Epoch: {} Loss: train {}, test {}. Accuracy: train: {}, test {}'.format(epoch, loss.item(), test_loss, train_accuracy, test_accuracy))
        print('-'*20)

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), dirName + 'model_raw_epoch_'.format(load_lstm ,cut_idx) + str(epoch) +'.ckpt')


    # after all epochs
    ## check for accuracy
    print("check for accuracy")        
    results = []
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    for data, target in train_loader:
        # print("train loader", counter, target.size())
        if target.size()[0] != batch_size:
            # print("batch size {} does not match, skip".format(target.size()[0]))
            continue
        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)        
        data, target = data.to(device), target.to(device)
        
        output = model(data, h_ini, c_ini)
        test_loss += criterion(
            output, target).item()  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        counter += 1

    test_loss /= len(train_loader.dataset)
    results.append( 100.0 * correct.item() / len(train_loader.dataset) )
    print("train loader acc: ", results)

    test_loss = 0
    correct = 0
    counter = 0
    # implement confusion matrix
    tot_pred = []
    tot_target = []
    for data, target in test_loader:
        # print("test loader", counter, target.size())

        if target.size()[0] != batch_size:
            # print("batch size {} does not match, skip".format(target.size()[0]))
            continue
        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)        
        data, target = data.to(device), target.to(device)

        output = model(data, h_ini, c_ini)
        test_loss += criterion(
            output, target).item()  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        target_ = target.data.view_as(pred)
        # print("sample pred size", pred.size(), "sample target_", target_.size()) # [21,1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        # prepare confusion matrix
        pred_ = pred.cpu().numpy()
        target_ = target_.cpu().numpy()
        # print(pred, target_)
        tot_pred.append(pred_)
        tot_target.append(target_)
        counter += 1


    test_loss /= len(test_loader.dataset)
    print("test_loss", test_loss, "correct", correct)

    results.append( 100.0 * correct.item() / len(test_loader.dataset) )

    # pickle.dump(results, open(dirName + 'results.pkl', 'wb'))

    print("results for {}".format(cut_idx))
    print(results)
    
    # construct confusion matrix
    #tot_pred = np.array(tot_pred).reshape((-1, 1))
    #tot_target = np.array(tot_target).reshape((-1, 1))
    #print("tot_pred", tot_pred.shape, "tot_target", tot_target.shape)
    #print(tot_pred)
    #print(tot_target)
    #print(type(tot_pred), type(tot_target))
    tot_pred = torch.tensor(tot_pred, dtype=torch.long)
    tot_target = torch.tensor(tot_target, dtype=torch.long)
    # print(tot_pred)
    # print(tot_target)
    conf = confusion_matrix(tot_pred.view(-1), tot_target.view(-1))
    # print("confusion matrix")
    # print(conf)
    results_dict = {"results": results,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "test_loss": epoch_test_loss,
                    "test_acc": epoch_test_acc,
                    "conf_mat": conf}
    dict_name = 'results_dict_comb_loadLSTM_{}_cut_{}_'.format(load_lstm, cut_idx) + str(num_epochs) + '.pkl'
    pickle.dump(results_dict, open(dirName + dict_name, 'wb'))

def main():

    # create the parser
    parser = argparse.ArgumentParser(description="Train a BioTac classifier")

    # add the arguments
    parser.add_argument("--cutidx", type=int, default=400, help="length for cutting the time window of sliding data")
    parser.add_argument("--lstm", type=bool, default=False, help="load pretrained lstm model from slide network")
    args = parser.parse_args()

    # # for CNN/MLP
    # data_diir = 'material_data/'

    # for BioCNN
    data_dir = 'data/BioTac/'
    
    print("Load data from {}".format(data_diir))

    datafile = 'BioTac_20_50.pkl'
    num_class = 20
    print("num_class", num_class)
    BioTac_learn(data_dir, datafile, num_class, args)

if __name__ == "__main__":
    main()
