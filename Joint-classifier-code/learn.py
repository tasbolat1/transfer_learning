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
        y = self.labels[ID]

        return X, y
    
    def get_X(self):
        X = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            x = torch.load(self.data_dir + ID + '.pt')
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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
        self.conv1_drop = nn.Dropout2d(p=0.8)

    def forward(self, x):
        #print('Conv:', x.size())
        x = self.conv1(x)
        # print('Conv', x.size())
        
        x = F.relu(F.max_pool2d(x, 2))
        # print('Pool', x.size())
        x = x.view(-1, 3*2*3)
        return x
    

class CNN_LSTM(nn.Module):
    def __init__(self, num_class, load_lstm=False, freeze_lstm=False):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
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


    def forward(self, x, device):
        #print(x.size())
        batch_size, timesteps, C, H, W, sequence_size = x.size()
        # init hidden states
        (h, c) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
        self.hidden = (h.contiguous(), c.contiguous())
        #print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        #print(c_in.size())
        c_out = self.cnn(c_in)
        #print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        output = F.log_softmax(r_out2, dim=1)
        # # check num of GPU
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output


class BioCNN(nn.Module):
    def __init__(self, c1=32, c2=64, h=18):
        super(BioCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=(3,3))
        self.conv1_drop = nn.Dropout2d(p=0.8)
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.hidden1 = nn.Linear(c2*4, h)
        # self.hidden2 = nn.Linear(1024, 18)

    def forward(self, x):
        # print("begin BioCNN")
        # print('before Conv:', x.size()) # [12600, 1, 8, 5] 12600 = 21*600

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
    def __init__(self, num_class, load_lstm=False, freeze_lstm=False, c1=32, c2=64, h=18):
        super(BioCNN_LSTM, self).__init__()
        self.cnn = BioCNN(c1, c2, h)
        self.lstm = nn.LSTM(
            input_size=h, 
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

    def forward(self, x, device):
        # print("enter BioCNN_LSTM")
        batch_size, timesteps, C, sequence_size, H, W = x.size()
        # init hidden states
        (h, c) = (torch.zeros(2, batch_size, 50).to(device), torch.zeros(2, batch_size, 50).to(device))
        self.hidden = (h.contiguous(), c.contiguous())    
        # print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        # print("c_in", c_in.size())
        c_out = self.cnn(c_in)
        # print("c_out", c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        self.lstm.flatten_parameters()
        # TODO: lstm output update (Tas)
        r_out, (h_n, h_c) = self.lstm(r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        output = F.log_softmax(r_out2, dim=1)
        # check num of GPU
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output


def learn(sensor, data_dir, datafile, num_class, args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logDir = sensor + '_info'
    try:
        # Create target Directory
        os.mkdir(logDir)
    except FileExistsError:
        print('Overwrite the {}'.format(logDir))
    logDir = logDir + '/'
    print("Load data from {}, log data to {}, datafile: {}, num_class: {}".format(data_dir, logDir, datafile, num_class))
    
    # set parameters
    batch_size= args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    save_interval  = args.save_interval
    cut_idx = args.cut_idx
    load_lstm = args.lstm
    c1B = args.c1B
    c2B = args.c2B
    hB = args.hB

    print("cut the data by first {} timesteps".format(cut_idx))
    print("load lstm? ", load_lstm)

    # load data
    [train_ids, train_labels,  valid_ids, valid_labels, test_ids, test_labels] = pickle.load(open(datafile, 'rb'))
    train_dataset = Dataset(data_dir, train_ids, train_labels, cut_idx=cut_idx)
    valid_dataset = Dataset(data_dir, valid_ids, valid_labels, cut_idx=cut_idx)
    test_dataset = Dataset(data_dir, test_ids, test_labels, cut_idx=cut_idx)
    
    train_loader = data2.DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = data2.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = data2.DataLoader(test_dataset, batch_size=batch_size)

    # create model
    if sensor == 'BioTac':
        model = BioCNN_LSTM(num_class, load_lstm=load_lstm, c1=c1B, c2=c2B, h=hB)
    elif sensor == 'Icub':
        model = CNN_LSTM(num_class, load_lstm=load_lstm)
    else:
        raise ValueError("Invalid sensor type {}".format(sensor))

    # enable multiple GPU (DataParallel)
    num_gpus = torch.cuda.device_count()
    print("Let's use {} GPUs!".format(num_gpus))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # calculate execution time 
    start=datetime.now()

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_valid_loss = []
    epoch_valid_acc = []
    model.train()

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = torch.FloatTensor(np.expand_dims(data, axis=1)) 
            data, target = Variable(data.to(device)), Variable(target.to(device))

            optimizer.zero_grad()
            output = model(data, device)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # validation for each epoch
        model.eval()

        # Calculate Train Accuracy         
        correct = 0
        total = 0.
        train_loss = 0
        train_nll_loss = []
        for data, target in train_loader:
            with torch.no_grad():
                # Load data to a Torch Variable
                data = torch.FloatTensor(np.expand_dims(data, axis=1))        
                data, target = Variable(data.to(device)), Variable(target.to(device))
                output = model(data, device)

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
        train_loss = np.mean(np.array(train_nll_loss))
        epoch_train_loss.append(train_loss)  # only save the last batch
        epoch_train_acc.append(train_accuracy)


        # Calculate Validation Accuracy         
        correct = 0
        total = 0.
        valid_loss = 0
        valid_nll_loss = []
        for data, target in valid_loader:
            with torch.no_grad():
                # Load data to a Torch Variable
                data = torch.FloatTensor(np.expand_dims(data, axis=1))        
                data, target = Variable(data.to(device)), Variable(target.to(device))
                output = model(data, device)

                # Obtain nll_loss, keep consistent criteria with training
                valid_nll_loss.append(F.nll_loss(output, target).item()) 

                # Obtain classification accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0) # Total number of labels

                valid_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(
                    1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

        valid_accuracy = 100 * correct / total
        valid_loss = np.mean(np.array(valid_nll_loss))
        epoch_valid_loss.append(valid_loss)
        epoch_valid_acc.append(valid_accuracy)

        print('Epoch: {} Loss: train {}, valid {}. Accuracy: train: {}, valid {}'.format(epoch, train_loss, valid_loss, train_accuracy, valid_accuracy))
        print('-'*20)

        if epoch % save_interval == 0:
            if sensor == "BioTac":
                model_name = 'model_{}_epoch_{}_lr_{}_c1B_{}_c2B_{}_hB_{}_at_{}.pkl'.format(sensor, str(num_epochs), learning_rate, c1B, c2B, hB,  epoch)
            else:
                model_name = 'model_{}_epoch_{}_at_{}.pkl'.format(sensor, str(num_epochs), epoch)
            print("model name", model_name)
            torch.save(model.state_dict(), logDir + model_name)


    # after all epochs
    print("check for final accuracy")        
    model.eval()
    results = []

    correct = 0
    total = 0.
    for data, target in train_loader:
        # Load data to a Torch Variable
        data = torch.FloatTensor(np.expand_dims(data, axis=1))        
        data, target = Variable(data.to(device)), Variable(target.to(device))
        output = model(data, device)
        total += target.size(0) # Total number of labels
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    train_accuracy = 100 * correct / total

    correct = 0
    total = 0.
    for data, target in valid_loader:
        # Load data to a Torch Variable
        data = torch.FloatTensor(np.expand_dims(data, axis=1))        
        data, target = Variable(data.to(device)), Variable(target.to(device))
        output = model(data, device)
        total += target.size(0) # Total number of labels
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    valid_accuracy = 100 * correct / total

    correct = 0
    total = 0.
    # implement confusion matrix
    tot_pred = []
    tot_target = []
    for data, target in test_loader:
        # Load data to a Torch Variable
        data = torch.FloatTensor(np.expand_dims(data, axis=1))        
        data, target = Variable(data.to(device)), Variable(target.to(device))
        output = model(data, device)
        total += target.size(0) # Total number of labels
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        target_ = target.data.view_as(pred)
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
        pred_ = pred.cpu().numpy()
        target_ = target_.cpu().numpy()
        tot_pred.extend(list(pred_))
        tot_target.extend(list(target_))
    test_accuracy = 100 * correct / total

    results = [train_accuracy, valid_accuracy, test_accuracy]
    
    time_elapsed = datetime.now()-start 

    print("results {}, time: {}".format(results, time_elapsed))
    
    # construct confusion matrix
    tot_pred, tot_target = np.squeeze(np.array(tot_pred)), np.squeeze(np.array(tot_target))
    tot_pred = torch.tensor(tot_pred, dtype=torch.long)
    tot_target = torch.tensor(tot_target, dtype=torch.long)
    conf = confusion_matrix(tot_pred.view(-1), tot_target.view(-1))

    results_dict = {"time": time_elapsed,
                    "results": results,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "valid_loss": epoch_valid_loss,
                    "valid_acc": epoch_valid_acc,
                    "conf_mat": conf}
    if model_name:
        print("save dict name")
        dict_name = model_name.replace("model", "dict").split("_at")[0] + ".pkl"
    else:
        print("create new name")
        dict_name = 'dict_{}_epoch_{}.pkl'.format(sensor, str(num_epochs))
    pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
    print("dump results dict to {}".format(dict_name))

def main():
    # create the parser
    parser = argparse.ArgumentParser(description="Train a BioTac classifier")

    # add the arguments
    parser.add_argument("--sensor", type=str, help="sensor type, either BioTac or Icub")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=2000, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--save_interval", type=int, default=100, help="interval to save the intermediate ckpt model")
    parser.add_argument("--cut_idx", type=int, default=None, help="length for cutting the time window of sliding data")
    parser.add_argument("--lstm", type=bool, default=False, help="whether load pretrained lstm model from slide network")
    parser.add_argument("--c1B", type=int, default=32, help="conv1 channel for BioCNN")
    parser.add_argument("--c2B", type=int, default=64, help="conv2 channel for BioCNN")
    parser.add_argument("--hB", type=int, default=18, help="hidden layer size for BioCNN_LSTM")

    args = parser.parse_args()

    num_class = 20    
    post_fix = '_20_50.pkl'

    sensor = args.sensor
    data_dir = 'data/' + sensor + "/"
    datafile = sensor + post_fix
    learn(sensor, data_dir, datafile, num_class, args)

if __name__ == "__main__":
    main()


'''
Note:
For BioTac
---test---
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1 --num_epochs 1 --lr 0.0001 --c1B 32 --c2B 64 --hB 18

python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.1 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.01 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.001 --c1B 32 --c2B 64 --hB 18
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.0001 --c1B 32 --c2B 64 --hB 18


python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.0001 --c1B 16 --c2B 32 --hB 32
python learn.py --sensor BioTac --cut_idx 400 --save_interval 1000 --num_epochs 5000 --lr 0.0001 --c1B 8 --c2B 16 --hB 32

For Icub
python learn.py --sensor Icub --lr 0.001



'''












