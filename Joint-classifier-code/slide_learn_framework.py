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

class Dataset(data2.Dataset):
    def __init__(self, data_dir, list_IDs, labels):
        # initialize
        self.data_dir = data_dir
        self.labels = labels
        self.list_IDs = list_IDs
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
        y = self.labels[ID]

        return X, y
    
    def get_X(self):
        X = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            # x = torch.load('slide_6_10/' + ID + '.pt').unsqueeze_(0)
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
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x, h_ini, c_ini):
        #print(x.size())
        batch_size, timesteps, C, H, W, sequence_size = x.size()
        # init hidden states
        # h = h_ini.transpose(0, 1)
        # c = c_ini.transpose(0, 1)
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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slide_learn(data_dir, datafile, num_class, args, iteration=400):
    
    dirName = 'slide_info_' + str(iteration)
    try:
        # Create target Directory
        os.mkdir(dirName)
    except FileExistsError:
        print('Re-writing the touch')
    dirName = dirName + '/'

    # set parameters
    batch_size=32
    num_epochs = 2000
    learning_rate = 0.001
    log_interval = 10
    save_interval  = 200
    seq_len = 75
    load_lstm = args.lstm
    print("load lstm? ", load_lstm)
    
    # load data
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open(datafile, 'rb'))
    training_dataset = Dataset(data_dir, train_ids, train_labels)
    test_dataset = Dataset(data_dir, test_ids, test_labels)

    train_dataset, val_dataset = random_split(training_dataset,[train_dataset_len//2, train_dataset_len-dataset_len//2] )

    train_loader = data2.DataLoader(training_dataset, batch_size=batch_size)
    test_loader = data2.DataLoader(test_dataset, batch_size=batch_size)
    
    # create model and enable multiple GPU (DataParallel)
    model = CNN_LSTM(num_class, load_lstm=load_lstm)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # set initial hidden state
    # (h_ini, c_ini) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
    (h_ini, c_ini) = (torch.zeros(batch_size, 2, 50).to(device) , torch.zeros(batch_size, 2, 50).to(device))
    # print("init hidden state", h_ini.size()) 

    epoch_lists = []
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_loss = []
    epoch_test_acc = []
    model.train()

    for epoch in range(1, num_epochs + 1):
        model.train()
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

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            #if batch_idx % log_interval == 0:
        epoch_lists.append(loss.item())
        print("Epoch {}, loss: {}".format(epoch, loss.item()))
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), dirName + 'model_epoch_' + str(epoch) +'.ckpt')
         
    # save epochs
    pickle.dump(epoch_lists, open(dirName + 'loss_20_50.pkl', 'wb'))
    
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
        data, target = Variable(data), Variable(target)
        
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
    for data, target in test_loader:
        # print("test loader", counter, target.size())

        if target.size()[0] != batch_size:
            # print("batch size {} does not match, skip".format(target.size()[0]))
            continue
        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)        
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        
        output = model(data, h_ini, c_ini)
        test_loss += criterion(
            output, target).item()  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        counter += 1

    test_loss /= len(test_loader.dataset)
    print("test_loss", test_loss, "correct", correct)

    results.append( 100.0 * correct.item() / len(test_loader.dataset) )
    
    pickle.dump(results, open(dirName + 'results.pkl', 'wb'))

    print("results")
    print(results)

def main():
   # create the parser
    parser = argparse.ArgumentParser(description="Train a BioTac classifier")

    # add the arguments
    parser.add_argument("--cutidx", type=int, default=None, help="length for cutting the time window of sliding data")
    parser.add_argument("--lstm", type=bool, default=False, help="load pretrained lstm model from slide network")
    args = parser.parse_args()
    data_dir = 'data/Icub/'
    datafile = 'Icub_20_50.pkl'
    num_class = 20
    # print("num_class", num_class)
    slide_learn(data_dir, datafile, num_class, args)

if __name__ == "__main__":
    main()