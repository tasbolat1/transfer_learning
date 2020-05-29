import os, sys
print(sys.path)

import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

import numpy as np  
import matplotlib.pyplot as plt  
import argparse
import pickle


class ComboDataset(Dataset):
    def __init__(self, data_dirs, list_IDs, labels, cut_idx, transform=False):
        self.data_dirs = data_dirs
        self.labels = labels
        self.list_IDs = list_IDs
        self.cut_idx = cut_idx
        self.transform = transform
        
    def __len__(self):
        assert len(self.list_IDs[0]) == len(self.list_IDs[1]), "Different lens for datasets. {} VS {}".format(len(self.list_IDs[0]), len(self.list_IDs[1]))
        return len(self.list_IDs[0])
    
    def __getitem__(self, index):
        # IDs = []
        Xs = []
        ys = []
        for i in range(len(self.data_dirs)):
            ID = self.list_IDs[i][index]
            X = torch.load(self.data_dirs[i] + ID + '.pt')
            if i == 1:
                # Icub data
                X = X.transpose(0, 2)
            if self.cut_idx[i] is None:
                self.cut_idx[i] = X.size()[0]
            # print("cut_idx ", self.cut_idx[i])
            X = X[:self.cut_idx[i],:,:]

            # Normalize your data here
            # print("In Combodataset {}".format(i))
            # print(X.size()) # torch.Size([400, 8, 5]) ([75, 10, 6])
            if self.transform:
                mean = np.repeat(0.5, X.size(0))
                std = np.repeat(0.5, X.size(0))
                transform_= transforms.Compose([
                            transforms.Normalize(mean=mean,std=std)])

                X = transform_(X)
            
            X.unsqueeze_(0)
            y = self.labels[i][ID]

            Xs.append(X)
            ys.append(y)

        return Xs, ys
    

class BioTac_AE(nn.Module):
    def __init__(self):
        super(BioTac_AE, self).__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            nn.ReLU()
        )
        self.act = nn.ReLU()
        self.enc_h1 = nn.Linear(64*4, 18)

        self.dec_h1 = nn.Linear(18, 64*4) 
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3,3)), 
            nn.ReLU()
        )

        # self.lstm = nn.LSTM
        
    
    def forward(self, x):
        batch_size, timesteps, C, sequence_size, H, W = x.size() # [4, 1, 1, 400, 8, 5]
        x = x.view(batch_size * timesteps*sequence_size, C, H, W)
        # print("going to enc_conv", x.size()) # [1600, 1, 8, 5]
        x = self.enc_conv(x) 
        # print("after enc_conv", x.size()) # [1600, 64, 4, 1]
        bs = x.size(0)
        x = x.view(bs, -1)
        enc_out = self.act(self.enc_h1(x))
        # print("enc_out", enc_out.size()) # [1600, 18])

        dec = self.dec_h1(enc_out) # [1600, 256]
        dec = dec.view(bs, 64, -1, 1) # [1600, 64, 4, 1]
        dec = self.dec(dec)
        # print("dec_out", dec.size()) # [1600, 1, 8, 5]
        dec = dec.view(batch_size, timesteps, C, sequence_size, H, W)

        return dec, enc_out.view(batch_size, -1, sequence_size)
        # r_in = x.view(-1, 3*2*3)
        # rout = self.lstm(r_in)
        # return rout, dec


class Icub_AE(nn.Module):
    def __init__(self, shared_lstm):
        super(Icub_AE, self).__init__()
        # self.enc_con1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
        # self.act = nn.ReLU()
        # self.enc_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5,3)),
            nn.ReLU(),
            nn.MaxPool2d(2, return_indices=True), # TODO: check
        )

        self.unpool = nn.MaxUnpool2d(2)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=(5,3)),# TODO: check
            nn.ReLU()
        )
        # self.lstm = shared_lstm

    def forward(self, x):
        batch_size, timesteps, C, sequence_size, H, W = x.size()  #[4, 1, 1, 75, 10, 6]
        x = x.view(batch_size * timesteps*sequence_size, C, H, W) # [300, 1, 10, 6]
        # print("Icub going to enc", x.size()) # [300, 1, 10, 6]
        enc_out, indices = self.enc(x) # [300, 3, 2, 3]
        out = self.unpool(enc_out, indices) # , output_size=size
        # print("out", out.size()) # [300, 3, 6, 4]
        # flatten enc_out
        enc_out = enc_out.view(enc_out.size(0), -1)
        # print("enc_out", enc_out.size()) # [300, 18]
        dec = self.dec(out)
        # print("dec", dec.size()) # [300, 1, 10, 6]
        dec = dec.view(batch_size, timesteps, C, sequence_size, H, W)

        return dec, enc_out.view(batch_size, -1, sequence_size)
        # r_in = enc_out.view(-1, 3*2*3)
        # rout = self.lstm(r_in)
        # return rout, dec


def joint_train_AE(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logDir =  'joint_info'
    try:
        # Create target Directory
        os.mkdir(logDir)
    except FileExistsError:
        print('Overwrite the {}'.format(logDir))
    logDir = logDir + '/'
 
    # set parameters
    num_class = args.num_class
    batch_size= args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    save_interval  = args.save_interval

    # load data
    sensors = ['BioTac', 'Icub']
    post_fix = '_20_50.pkl'
    data_dirs = []
    datafiles = []
    train_ids = []
    train_labels = []
    valid_ids = []
    valid_labels = []
    test_ids = []
    test_labels = []

    for i in range(len(sensors)):
        sensor = sensors[i]
        data_dirs.append('data/' + sensor + "/")
        datafile =sensor + post_fix
        datafiles.append(datafile)
        id_label = pickle.load(open(datafile, 'rb'))
        train_ids.append(id_label[0])
        train_labels.append(id_label[1])
        valid_ids.append(id_label[2])
        valid_labels.append(id_label[3])
        test_ids.append(id_label[4])
        test_labels.append(id_label[5])

    cut_idx = [400, None]

    norm = True
    train_dataset = ComboDataset(data_dirs, train_ids, train_labels, cut_idx=cut_idx, transform=norm)
    valid_dataset = ComboDataset(data_dirs, valid_ids, valid_labels, cut_idx=cut_idx, transform=norm)
    test_dataset = ComboDataset(data_dirs, test_ids, test_labels, cut_idx=cut_idx, transform=norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Load data from {}, log data to {}, datafile: {}, num_class: {}".format(data_dirs, logDir, datafiles, num_class))
    
    # create model
    net_B = BioTac_AE() 
    net_I = Icub_AE(shared_lstm=None)
    # net_I = Icub_AE(shared_lstm=net_B.lstm)

    # # enable multiple GPU (DataParallel)
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     print("Let's use {} GPUs!".format(num_gpus))
    #     model = nn.DataParallel(model)
    net_B.to(device)
    net_I.to(device)

    optim_B = optim.Adam(net_B.parameters(), lr=learning_rate)
    optim_I = optim.Adam(net_I.parameters(), lr=learning_rate)

    net_B.train()
    net_I.train()

    LB = []
    LI = []
    LC = []

    # loss weights
    alpha, beta, gamma = 1, 1, 1

    for epoch in range(1, num_epochs + 1):
        net_B.train()
        net_I.train()

        for batch_idx, (datas,targets) in enumerate(train_loader):
            data_B = datas[0]
            target_B = targets[0]
            data_B = torch.FloatTensor(np.expand_dims(data_B, axis=1)) 
            data_B, target_B = Variable(data_B.to(device)), Variable(target_B.to(device))
            output_B, enc_out_B = net_B(data_B)
            print("B ", output_B.size(), enc_out_B.size()) 
            # B  torch.Size([4, 1, 1, 400, 8, 5]) torch.Size([4, 18, 400])

            data_I = datas[1]
            target_I = targets[1]
            data_I = torch.FloatTensor(np.expand_dims(data_I, axis=1)) 
            data_I, target_I = Variable(data_I.to(device)), Variable(target_I.to(device))
            output_I, enc_out_I = net_I(data_I)
            print("I ", output_I.size(), enc_out_I.size())
            # I  torch.Size([4, 1, 1, 10, 6, 75]) torch.Size([4, 18, 75])
            
            # print("data B")
            # print(data_B.size(), output_B.size()) # torch.Size([4, 1, 1, 400, 8, 5])
            # print(torch.squeeze(data_B[0,:,:,0, :, :]))

            # print("data I")
            # print(data_I.size(),output_I.size()) # torch.Size([4, 1, 1, 75, 10, 6])
            # print(torch.squeeze(data_I[0, :, :, :, :, 0]))

            print("enc_out B")
            print(enc_out_B[0, :, 1])

            assert torch.all(torch.eq(target_B, target_I)), "Different targets B: {}, I: {}".format(target_B, target_I)
            
            # reconstruction loss
            loss_B = F.mse_loss(output_B, data_B)
            loss_I = F.mse_loss(output_I, data_I)
            
            # common feature loss
            # `mini-batch x channels x [optional depth] x [optional height] x width`
            # batch_size x channel (18) x seq_size 
            enc_out_B_ip = F.interpolate(enc_out_B, size=enc_out_I.size(-1), mode='linear')
            print("enc_out B ip")
            print(enc_out_B_ip[0, :, 1])
            loss_C = F.mse_loss(enc_out_B_ip, enc_out_I)
            print("enc_out I")
            print(enc_out_I[0, :, 1])
            

            optim_B.zero_grad()
            optim_I.zero_grad()
            # loss_B.backward()
            # loss_I.backward()
            # loss_C.backward()
            loss = alpha*loss_B + beta*loss_I + gamma*loss_C
            loss.backward()

            optim_B.step()
            optim_I.step()

            # compare the loss scales
            print("loss: B {}, I {}, C {}".format(loss_B, loss_I, loss_C))

            sys.exit()

        LB.append(loss_B.item())
        LI.append(loss_I.item())
        LC.append(loss_C.item())

        print('Epoch: {} Loss: B {}, I {}, C {}.'.format(epoch, 100*loss_B.item(), loss_I.item(), loss_C.item()))
        print('-'*20)
        if epoch % save_interval == 0:
            net_B_name = 'net_B_epoch_{}_at_{}.pkl'.format(str(num_epochs), epoch)
            net_I_name = 'net_I_epoch_{}_at_{}.pkl'.format(str(num_epochs), epoch)
            torch.save(net_B.state_dict(), logDir + net_B_name)
            torch.save(net_I.state_dict(), logDir + net_I_name)

    results_dict = {"LB": LB,
                    "LI": LI,
                    "LC": LC}
    dict_name = "BI_joint_epoch_{}.pkl".format(num_epochs)
    pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
    print("dump results dict to {}".format(dict_name))



def main():
    # create the parser
    parser = argparse.ArgumentParser(description="Train a BioTac classifier")

    # add the arguments
    parser.add_argument("--num_class", type=int, default=20, help="number of classes")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=5000, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--save_interval", type=int, default=100, help="interval to save the intermediate ckpt model")
    
    args = parser.parse_args()
    joint_train_AE(args)


if __name__ == "__main__":
    main()




