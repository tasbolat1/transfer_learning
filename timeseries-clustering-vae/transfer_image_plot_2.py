# -*- coding: utf-8 -*-
import os, sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
from celluloid import Camera

plt.rcParams['animation.ffmpeg_path'] = '/home/ruihan/anaconda3/envs/opencv/bin/ffmpeg' # Add the path of ffmpeg here!!

# initiate a writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

# create a figure
fig = plt.figure()
camera = Camera(fig)
# fig = plt.figure(figsize=(16,12))
# plt.xlim(1999, 2016)
# plt.ylim(np.min(overdose)[0], np.max(overdose)[0])
# plt.xlabel('latentB_tsne-2d-one',fontsize=20)
# plt.ylabel('latentB_tsne-2d-two',fontsize=20)
# plt.title('tsne of latentB representation',fontsize=20)



pickle_dir = "/home/ruihan/Desktop/MLDA/df_latent_all_epochs_2000_BI.pkl"
# df_latent_all_epochs = pickle.load(open(pickle_dir, "rb"))
df_latent_all_epochs = pd.read_csv("/home/ruihan/Desktop/MLDA/Joint_tsne_2000_epochs.csv")
epochs = df_latent_all_epochs.shape[0]//192

# add 'mat' column
if "mat" not in df_latent_all_epochs.columns:
    print("add mat column")
    mat_list = ["CP", "CT", "BT", "LF", "PP", "FL", "S2", "P2", "PS", "PT", "S1", "CK", "EV", "P1", "FB", "WH", "SF", "SS", "FM", "MT"]
    df_latent_all_epochs['mat'] = df_latent_all_epochs.apply(lambda row: mat_list[int(row.y)], axis=1)
    df_latent_all_epochs.to_pickle(pickle_dir)

# print(df_latent_all_epochs.shape, epochs) # (384000, 87) 2000 86 = 40*2 + latent (4) + 'epoch' + 'y' + 'mat'
# print(df_latent_all_epochs.head())

# extract columns from dataframe
df_all_epochs_B = df_latent_all_epochs[["epoch", "latentB_tsne-2d-one", "latentB_tsne-2d-two", "y", "mat"]]
df_all_epochs_I = df_latent_all_epochs[["epoch", "latentI_tsne-2d-one", "latentI_tsne-2d-two", "y", "mat"]]
df_all_epochs_B = df_all_epochs_B.rename(columns={"epoch":"epoch","latentB_tsne-2d-one": "latent_tsne-2d-one", "latentB_tsne-2d-two": "latent_tsne-2d-two", "y":"y", "mat":"mat"})
df_all_epochs_I = df_all_epochs_I.rename(columns={"epoch":"epoch","latentI_tsne-2d-one": "latent_tsne-2d-one", "latentI_tsne-2d-two": "latent_tsne-2d-two", "y":"y", "mat":"mat"})


images = []

for epoch in range(1999, 2000):
    print(epoch)
    subset_B = df_all_epochs_B.loc[df_all_epochs_B['epoch'] == epoch]
    subset_I = df_all_epochs_I.loc[df_all_epochs_I['epoch'] == epoch]
    concatenated = pd.concat([subset_B.assign(dataset='setB'), subset_I.assign(dataset='setI')])

    img = sns.scatterplot(
    x="latent_tsne-2d-one", y="latent_tsne-2d-two",
    hue="mat",
    style="dataset",
    palette=sns.color_palette("hls", 20, desat=1),
    data=concatenated,
    legend="full",
    alpha=0.99)

    img.set(title = 'TSNE visualization of joint latent space at epoch: {}'.format(epoch)) # set title for static image
    # img.axes.text(-8, 8,'Epoch: {}'.format(epoch), fontsize=16) # add floating for video

    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig("/home/ruihan/Desktop/MLDA/figures/temp_joint_latent_data_{}.png".format(epoch), dpi=1200,bbox_extra_artists=(lgd,),
            bbox_inches='tight')
    plt.show()
#     camera.snap()

# anim = camera.animate(interval=500, repeat_delay=3000, blit=True)
# anim.save("/home/ruihan/Desktop/MLDA/tsne_I_start.mp4", writer=writer)
# plt.show()