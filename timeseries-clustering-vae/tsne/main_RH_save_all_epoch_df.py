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
df_latent_all_epochs = pickle.load(open("df_latent_all_epochs_{}_BI.pkl".format(num_epochs), "rb"))
epochs = df_latent_all_epochs.shape[0]//192
# print(df_latent_all_epochs.shape, epochs)
# print(df_latent_all_epochs.head())
# create column names
latent_length = 40
latent_B_cols = ['latentB_'+str(i) for i in range(latent_length)]
latent_I_cols = ['latentI_'+str(i) for i in range(latent_length)]
mat_list = ["CP", "CT", "BT", "LF", "PP", "FL", "S2", "P2", "PS", "PT", "S1", "CK", "EV", "P1", "FB", "WH", "SF", "SS", "FM", "MT"]


# In[4]:


tsne_columns = ['latentB_tsne-2d-one', 'latentB_tsne-2d-two', 'latentI_tsne-2d-one', 'latentI_tsne-2d-two']
df_all_epochs = pd.DataFrame(columns=tsne_columns+['y', 'mat', 'epoch'])
for epoch in range(num_epochs):
    print("process tsne for epoch {}".format(epoch))
    df_latent = df_latent_all_epochs.loc[df_latent_all_epochs['epoch'] == epoch]
    # convert panda dataframe to numpy array
    data_subset_B = df_latent[latent_B_cols].values
    data_subset_I = df_latent[latent_I_cols].values
    y_array = df_latent['y'].values

    # convert numpy array to torch tensor
    # device = torch.device("cuda:{}".format(args.cuda))
    XB = torch.from_numpy(data_subset_B).float() #.to(device)
    XI = torch.from_numpy(data_subset_I).float() #.to(device)
    C = torch.from_numpy(y_array).float() #.to(device)

    # initialize and train pytorch joint-TSNE
    TJ = joint_tsne.myJointTSNE(XB, XI)
    resB, resI = TJ.train(epoch=400, lr=10, weight_decay=0, momentum=1, show=False)
    resB, resI = resB.numpy(), resI.numpy()

    # prepare dataframe for sns
    data = np.concatenate((resB, resI), axis=1)
    df = pd.DataFrame(data, columns=tsne_columns)
    df['y'] = C.numpy()
    # df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
    df['mat'] = df.apply(lambda row: mat_list[int(row.y)], axis=1)
    df['epoch'] = np.repeat(epoch, 192)#[:,None]
    df_all_epochs = df_all_epochs.append(df)
#     print(df.shape)
#     print(df.head())

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

# save df to csv
df_all_epochs.to_csv('Joint_tsne_{}_epochs.csv'.format(num_epochs), index = False)


# In[ ]:




