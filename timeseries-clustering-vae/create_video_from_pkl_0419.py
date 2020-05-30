import pickle
from PIL import Image
import numpy as np 
from matplotlib import cm
import matplotlib.pyplot as plt 
import cv2
from pathlib import Path
import matplotlib.image as mpimg



def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


T = 75
data_dir = "/home/ruihan/Desktop/MLDA/"


'''
display the whole picture by putting timeframe as horizontal axis
'''

def plot_the_whole(pkl_name):
    myarray = pickle.load(open(pkl_name, 'rb'))
    width, length, timestep = myarray.shape
    myarray = myarray.reshape((-1, timestep))
    # print(myarray.shape)
    plt.imshow(myarray)
    # num_row = 6
    # fig, axes = plt.subplots(nrows=num_row, ncols=1) 
    # fig.suptitle(pkl_name.split("/")[-1][:-4])
    # for row_idx in range(num_row):
    #     axes[row_idx].imshow(myarray[row_idx, :, :])
    # save fig
    figname = pkl_name[:-4]+"_whole.png"
    plt.savefig(figname)

    
''' plot whole for all samples of one class '''
available_class = [2, 3, 9, 14, 17, 18]
class_number_list = [2, 14, 17]
num_samples = 6


# fig, axes = plt.subplots(nrows=num_samples, ncols=len(class_number_list)) 

# for ax, col in zip(axes[0], class_number_list):
#     ax.set_title("Class: " + str(col))

# for ax, row in zip(axes[:,0], rows):
#     ax.set_ylabel(row, rotation=0, size='large')


# for col_idx in range(len(class_number_list)):
#     for row_idx in range(0, num_samples):
#         class_number = class_number_list[col_idx]
#         pkl_name = data_dir+"Dec_imgs/Dec_C{}_S{}.pkl".format(class_number, row_idx)
#         # plot_the_whole(pkl_name)
#         myarray = pickle.load(open(pkl_name, 'rb'))
#         width, length, timestep = myarray.shape
#         myarray = myarray.reshape((-1, timestep))
#         axes[row_idx, col_idx].imshow(myarray)
# figname = data_dir+"figures/combine_{}_class_{}__samples.png".format(len(class_number_list), num_samples)
# plt.savefig(figname, dpi=1200)


col_idx = 1
row_idx = 3
class_number = class_number_list[col_idx]
pkl_name = data_dir+"Dec_imgs/Dec_C{}_S{}.pkl".format(class_number, row_idx)
# plot_the_whole(pkl_name)
myarray = pickle.load(open(pkl_name, 'rb'))
width, length, timestep = myarray.shape
for ts in range(10):
    plt.figure(ts)
    plt.imshow(myarray[:, :, ts])
    figname = data_dir+"figures/sample_I_data/{}_{}_{}.png".format(class_number, row_idx, ts)
    plt.savefig(figname, dpi=1200)

# plt.show()
plt.close('all')



'''
Save each frame as .png image
'''
def save_all_frames(pkl_name):
    # create dir if not existing
    dirname = pkl_name[:-4]+"/"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    # load pkl data
    myarray = pickle.load(open(pkl_name, 'rb'))
    for t in range(T):  
        data = myarray[:,:, t]
        plt.imshow(data)
        # save the image for this particular timeframe
        fname = dirname + "Dec_C{}_S{}_T{}.png".format(class_number, n, t)
        plt.savefig(fname)
        plt.close()

'''
after visual inspection for representative samples,
save png for one specific sample of one class
'''
# class_number = 9
# n = 8
# pkl_name = "Dec_imgs/Dec_C{}_S{}.pkl".format(class_number, n)
# save_all_frames(pkl_name)


# '''
# Plot the class with maximum number of samples to visualize similarity
# '''
# # Create subplot matrix

# num_col = 1
# # row for frames
# num_row = 25

# # create labels
# col_labels = list(np.repeat(class_number, num_col))
# cols = [str(ele) for ele in col_labels]

# rows = ['CH {} '.format(row+1) for row in range(num_row)]
# print("cols", cols, "rows", rows)

# fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(20,10)) 

# # set labels
# # for ax, col in zip(axes[0], cols):
# #     ax.set_title(col)

# # for ax, row in zip(axes[:,0], rows):
# #     ax.set_ylabel(row, rotation=0, size='large')

# # plot the figures
# # for col_idx in range(num_col):
# col_idx = 0
# for row_idx in range(num_row):
#     # axes[row_idx, col_idx].imshow(myarray[:, :,row_idx])
#     axes[row_idx].imshow(myarray[:, :,row_idx])

# fig.tight_layout()

# # plt.savefig("Dec_imgs/C{}_NS{}_NT{}.png".format(class_number, num_col, num_row))
# plt.show()

    