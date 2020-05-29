#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joint_tsne
import pickle
import seaborn as sns
import pandas as pd
t.manual_seed(1)


# In[3]:


'''Load data'''
num_epochs = 2000
# all epochs
df_latent_all_epochs_B = pickle.load(open("df_latent_all_epochs_{}_B.pkl".format(num_epochs), "rb"))
df_latent_all_epochs_I = pickle.load(open("df_latent_all_epochs_{}_I.pkl".format(num_epochs), "rb"))
epochs = df_latent_all_epochs_B.shape[0]//192
# print(df_latent_all_epochs.shape, epochs)
# print(df_latent_all_epochs.head())
# create column names
latent_length = 40
latent_B_cols = ['latentB_'+str(i) for i in range(latent_length)]
latent_I_cols = ['latentI_'+str(i) for i in range(latent_length)]
mat_list = ["CP", "CT", "BT", "LF", "PP", "FL", "S2", "P2", "PS", "PT", "S1", "CK", "EV", "P1", "FB", "WH", "SF", "SS", "FM", "MT"]


# In[4]:


# save latent for B
tsne_columns = ['latentB_tsne-2d-one', 'latentB_tsne-2d-two']
df_all_epochs = pd.DataFrame(columns=tsne_columns+['y', 'mat', 'epoch'])
for epoch in range(num_epochs):
    print("process tsne for epoch {}".format(epoch))
    df_latent_B = df_latent_all_epochs_B.loc[df_latent_all_epochs_B['epoch'] == epoch]
    # convert panda dataframe to numpy array
    data_subset_B = df_latent_B[latent_B_cols].values
    y_array = df_latent_B['y'].values

    # convert numpy array to torch tensor
    # device = torch.device("cuda:{}".format(args.cuda))
    XB = torch.from_numpy(data_subset_B).float() #.to(device)
    C = torch.from_numpy(y_array).float() #.to(device)

    # initialize and train pytorch TSNE
    TB = joint_tsne.myTSNE(XB)
    resB = TB.train(epoch=400, lr=10, weight_decay=0, momentum=1, show=False).numpy()

    # prepare dataframe for sns
    df = pd.DataFrame(resB, columns=tsne_columns)
    df['y'] = C.numpy()
    # df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
    df['mat'] = df.apply(lambda row: mat_list[int(row.y)], axis=1)
    df['epoch'] = np.repeat(epoch, 192)#[:,None]
    df_all_epochs = df_all_epochs.append(df)


# save df to csv
df_all_epochs.to_csv('B_tsne_{}_epochs.csv'.format(num_epochs), index = False)
print("finish processing B", df_all_epochs.shape)


# In[5]:


# save latent for I
tsne_columns = ['latentI_tsne-2d-one', 'latentI_tsne-2d-two']
df_all_epochs = pd.DataFrame(columns=tsne_columns+['y', 'mat', 'epoch'])
for epoch in range(num_epochs):
    print("process tsne for epoch {}".format(epoch))
    df_latent_I = df_latent_all_epochs_I.loc[df_latent_all_epochs_I['epoch'] == epoch]
    # convert panda dataframe to numpy array
    data_subset_I = df_latent_I[latent_I_cols].values
    y_array = df_latent_I['y'].values

    # convert numpy array to torch tensor
    # device = torch.device("cuda:{}".format(args.cuda))
    XI = torch.from_numpy(data_subset_I).float() #.to(device)
    C = torch.from_numpy(y_array).float() #.to(device)

    # initialize and train pytorch TSNE
    TI = joint_tsne.myTSNE(XI)
    resI = TI.train(epoch=400, lr=10, weight_decay=0, momentum=1, show=False).numpy()

    # prepare dataframe for sns
    df = pd.DataFrame(resI, columns=tsne_columns)
    df['y'] = C.numpy()
    # df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
    df['mat'] = df.apply(lambda row: mat_list[int(row.y)], axis=1)
    df['epoch'] = np.repeat(epoch, 192)#[:,None]
    df_all_epochs = df_all_epochs.append(df)


# save df to csv
df_all_epochs.to_csv('I_tsne_{}_epochs.csv'.format(num_epochs), index = False)
print("finish processing I", df_all_epochs.shape)


# In[6]:



''' visualization offline'''
#     # visualize with sns
#     # extract columns from dataframe
#     df_B = df[["latentB_tsne-2d-one", "latentB_tsne-2d-two", "y", "mat"]]
#     df_I = df[["latentI_tsne-2d-one", "latentI_tsne-2d-two", "y", "mat"]]
#     subset_B = df_B.rename(columns={"latentB_tsne-2d-one": "latent_tsne-2d-one", "latentB_tsne-2d-two": "latent_tsne-2d-two", "y":"y", "mat":"mat"})
#     subset_I = df_I.rename(columns={"latentI_tsne-2d-one": "latent_tsne-2d-one", "latentI_tsne-2d-two": "latent_tsne-2d-two", "y":"y", "mat":"mat"})

#     concatenated = pd.concat([subset_B.assign(dataset='setB'), subset_I.assign(dataset='setI')])

#     img = sns.scatterplot(
#     x="latent_tsne-2d-one", y="latent_tsne-2d-two",
#     hue="mat",
#     style="dataset",
#     palette=sns.color_palette("hls", 20, desat=1),
#     data=concatenated,
#     legend="full",
#     alpha=0.99)

#     #set legend position
#     # img.set_position([0,0, 1, 0.8])
#     lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
#     # set title
#     # img.set(title = 'Perplexity: {}'.format(perplexity))
#     # plt.savefig("PT_tsne_BI_joint_sns_epoch_{}.png".format(str(epoch+1)), dpi=1200)
#     # save legend outside

#     plt.savefig("PT_tsne_BI_joint_sns_epoch_{}.png".format(str(epoch+1)), dpi=1200, bbox_extra_artists=[lgd], bbox_inches='tight')
#     plt.show()

