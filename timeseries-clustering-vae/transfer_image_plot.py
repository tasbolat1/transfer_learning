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

df_latent_all_epochs = pickle.load(open("/home/ruihan/Desktop/MLDA/df_latent_all_epochs_2000_BI.pkl", "rb"))
epochs = df_latent_all_epochs.shape[0]//192
print(df_latent_all_epochs.shape, epochs)
images = []
print(df_latent_all_epochs.head())

for epoch in range(1990, 2000):
    # print(epoch)
    subset = df_latent_all_epochs.loc[df_latent_all_epochs['epoch'] == epoch]

    # img = plt.scatter(subset['latentB_tsne-2d-one'], subset['latentB_tsne-2d-two'], c=subset['y'],
    #               marker='o', s=10, edgecolor='')
    fig, ax = plt.subplots()
    img = sns.scatterplot(
    x="latentB_tsne-2d-one", y="latentB_tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 20, desat=1),
    data=subset,
    legend="full",
    alpha=0.3, ax=ax)

    imgI = sns.scatterplot(
    x="latentI_tsne-2d-one", y="latentI_tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 20, desat=1),
    data=subset,
    legend="full",
    marker="+",
    alpha=0.3, ax=ax)

    img.axes.text(-8, 8,'Epoch: {}'.format(epoch), fontsize=16) #add text

    camera.snap()
    # img = latent_B_tsne_plot.get_figure()    
    # img.savefig("latent_B_tsne_plot.png")
#     images.append([img])
# print(type(img))

anim = camera.animate(interval=500, repeat_delay=3000, blit=True)
anim.save("/home/ruihan/Desktop/MLDA/tsne-sns_latent_end.mp4", writer=writer)
plt.show()

# ani = ArtistAnimation(fig, images, interval=500, repeat_delay=3000, blit=True)
# ani.save("/home/ruihan/Desktop/MLDA/tsne-sns.mp4", writer=writer)
# # ani.save("/home/ruihan/Desktop/MLDA/mlp_process.mp4",  extra_args=['-vcodec', 'libx264'])
# plt.show()


'''
tsne visualization
'''
# # visualize training process
# data_subset_B = df_latent[latent_B_cols].values
# data_subset_I = df_latent[latent_I_cols].values
# tsneB = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
# tsne_results_B = tsneB.fit_transform(data_subset_B)
# tsneI = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
# tsne_results_I = tsneI.fit_transform(data_subset_I)
# df_latent['latentB_tsne-2d-one'] = tsne_results_B[:,0]
# df_latent['latentB_tsne-2d-two'] = tsne_results_B[:,1]
# df_latent['latentI_tsne-2d-one'] = tsne_results_I[:,0]
# df_latent['latentI_tsne-2d-two'] = tsne_results_I[:,1]