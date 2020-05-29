import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os, sys
import pickle


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout

    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        
        self.lstm = nn.LSTM(input_size=self.number_of_features, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.hidden_layer_depth, 
                            batch_first=True,dropout=dropout)

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        lstm: h_n of shape (num_layers * num_directions, batch, hidden_size)
        gru: h_n of shape (num_layers * num_directions, batch, hidden_size):
        """

        batch_size, H, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out = x[...,t]
            cnn_embed_seq.append(cnn_out)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        
        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(cnn_embed_seq)
        
        h_end = h_n[-1, :, :] # TODO check

        return h_end


class CNN(nn.Module):
    def __init__(self, C=1, H=6, W=10, cnn_number_of_features=18):
        super(CNN, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.cnn_number_of_features = cnn_number_of_features

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 5)),
            nn.MaxPool2d(2, return_indices=True),
        )

    def forward(self, x):
        # print("in cnn: ", self.cnn_number_of_features) # 18
        cnn_out, mp_indices = self.seq(x)
        cnn_out = cnn_out.view(-1, self.cnn_number_of_features)
        return cnn_out, mp_indices


class CnnEncoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout


    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, cnn_number_of_features=None):

        super(CnnEncoder, self).__init__()
        if cnn_number_of_features is not None:
            self.number_of_features = cnn_number_of_features
        else:
            self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.cnn = CNN(cnn_number_of_features=cnn_number_of_features)
        self.lstm = nn.LSTM(input_size=self.number_of_features, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,dropout=dropout)


    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        lstm: h_n of shape (num_layers * num_directions, batch, hidden_size)
        gru: h_n of shape (num_layers * num_directions, batch, hidden_size):
        """
        
        x = x.unsqueeze(1)
        batch_size, C, H, W, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out, mp_indices = self.cnn(x[...,t])
            cnn_embed_seq.append(cnn_out)
        
#         print("in cnn-encoder", cnn_out.size(), len(cnn_embed_seq)) # [32, 18], 75
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
#         print('cnn_embed_seq: ', cnn_embed_seq.shape)

        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(cnn_embed_seq)
#         print('lstm out: ', r_out.shape)

        h_end = h_n[-1, :, :]
        return h_end, mp_indices


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length, var=False):
        super(Lambda, self).__init__()
        
        self.var = var
        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """
        
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.var:
            if self.training:
                std = torch.exp(0.5 * self.latent_logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(self.latent_mean)
            else:
                return self.latent_mean
        else:
            return self.latent_mean
            

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.device = device

        self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
#         # batch, seq, feature
#         self.lstm = nn.LSTM(
#             input_size=3*2*3, 
#             hidden_size=50, 
#             num_layers=2,
#             batch_first=True,
#            dropout=0.8)

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).to(self.device)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """

        h_state = self.latent_to_hidden(latent)
        c_0 = torch.zeros(self.hidden_layer_depth, h_state.size(0), self.hidden_size, requires_grad=True).to(self.device)
#         print("latent {}, h_state {}, c_0 {}".format(latent.size(), h_state.size(), c_0.size()))

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)]).to(self.device)
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, c_0))
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        out = out.permute(1, 2, 0)
        return out


class CnnDecoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device, cnn_number_of_features=None):

        super(CnnDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        # manually specify the C, H, W 
        # TODO: make consistent with CnnEncoder
        self.C = 1
        self.H = 6
        self.W = 10

        if cnn_number_of_features is None:
            self.output_size = output_size
        else: 
            self.output_size = cnn_number_of_features
        self.device = device

        # for mirroring CNN
        self.unpool = nn.MaxUnpool2d(2)
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=(3, 5)),# TODO: check
            nn.ReLU()
        )

        self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).to(self.device)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, mp_indices):
        """Converts latent to hidden to output

        :param latent: latent vector, mp_indices to reverse maxpooling correctly
        :return: outputs consisting of mean and std dev of vector
        """
        
        h_state = self.latent_to_hidden(latent)
        c_0 = torch.zeros(self.hidden_layer_depth, h_state.size(0), self.hidden_size, requires_grad=True).to(self.device)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, c_0))
        else:
            raise NotImplementedError
        
        # RH add mirror cnn
        out = self.hidden_to_output(decoder_output)
        # print("in cnn decoder: ", out.size()) # [75, 32, 18]
        sequence_size, batch_size, number_of_features = out.size()
        # print("unpooling", out.size()) #torch.Size([75, 32, 18]) 
        out = out.permute(1, 2, 0)
        # mirror CNN embedding
        dcnn_seq = []
        # print("in cnn encoder: ", x.size()) # [32, 1, 6, 10, 75]
        for t in range(sequence_size):
            
            x = out[..., t].view(batch_size,3,2,3)
            x = self.unpool(x, mp_indices) 
            x = self.dcnn(x)
            dcnn_seq.append(x)
            
        dcnn_seq = torch.stack(dcnn_seq, dim=0).transpose_(0, 1)
    
        dcnn_seq = dcnn_seq.reshape(batch_size, 6, 10, sequence_size)

        return dcnn_seq

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class VRAEC(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder with classifier.
    This module is used for dimensionality reduction of timeseries and perform classification using hidden representation.

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    :param model_name name of state dict to be stored under dload directory
    :param header: "CNN", "GCN", "MLP" or "None", hearder implemented before encoder and after decoder
    """
    def __init__(self, num_class, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005,
                 n_epochs=5, dropout_rate=0., loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.', model_name='model.pth', header=None, device='cpu'):

        super(VRAEC, self).__init__()

        self.dtype = torch.FloatTensor
        self.ydtype = torch.LongTensor
        self.use_cuda = cuda
        self.header = header
        self.device = device
        self.epoch_train_acc = []
        
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            self.ydtype = torch.cuda.LongTensor

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate)

            self.decoder = Decoder(sequence_length=sequence_length,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device)
        
        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18)
            
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18)
        
        else:
            raise NotImplementedError
            

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.classifier = nn.Sequential(
            nn.Linear(latent_length, num_class),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload
        self.model_name = model_name

        if self.use_cuda:
            self.cuda()

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output = self.encoder(x)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices)
        else:
            raise NotImplementedError
        output = self.classifier(latent)

        return x_decoded, latent, output

    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function
        Add [0] so that it is compatible with CnnEncoder that returns two args, enc_output and mp_indices

        :param x: input batch tensor
        :return: intermediate latent vector


        """
        if self.header is None:
            enc = self.encoder(
                            Variable(x.type(self.dtype), requires_grad = False)
                        )
        elif self.header == "CNN":
            enc, _ = self.encoder(
                            Variable(x.type(self.dtype), requires_grad = False)
                        )
        else:
            raise NotImplementedError
        # print("enc to labd", enc.size())
        return self.lmbd(enc).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for i, (XI, XB,  y) in enumerate(test_loader):
                    if self.header == 'CNN':
                        X = XI
                    else:
                        X = XB

                    x_decoded_each = self._batch_reconstruct(X)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')

    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []
                correct = 0
                total = 0.
                for i, (XI, XB,  y) in enumerate(test_loader):
                    if self.header == 'CNN':
                        X = XI
                    else:
                        X = XB
                    # obtain the latent representation
                    z_run_each = self._batch_transform(X)
                    z_run.append(z_run_each)
                    # perform classification on latent representation
                    z_cl = Variable(torch.from_numpy(z_run_each).type(self.dtype), requires_grad = False)
                    output = self._classify(z_cl)
                    
                    # compute acc
                    # Obtain classification accuracy
                    y = Variable(y.type(self.ydtype), requires_grad=False)
                    total += y.size(0) # Total number of labels
                    pred = output.data.max(
                        1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

                acc = 100 * correct / total
                print("test acc ", acc)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run, acc

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))