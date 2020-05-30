import pickle
from PIL import Image
import numpy as np 
from matplotlib import cm
import matplotlib.pyplot as plt 
import cv2
from pathlib import Path
import matplotlib.image as mpimg

T = 75
num_class = 20
data_dir = "/home/ruihan/Desktop/MLDA/"

decoded_dict = pickle.load(open(data_dir+"decoded_dict_I_0420.pkl", "rb"))
print(decoded_dict.keys())
        
# map the class_number to number of samples
def count_dict_class(key):
    return [key, len(decoded_dict[key])]

pairs = map(count_dict_class, decoded_dict.keys())
pairs = list(pairs)
# pairs.sort(key=lambda x:x[0]) # in-place sorting
pairs = np.array(sorted(pairs, key=lambda x:x[0])) # non in-lace sorting

# find the class with minimum number of test samples
_, min_idx = np.argmin(pairs, axis=0)

# class_number = pairs[min_idx][0]
num_samples = pairs[min_idx][1]
print("at least have {} samples available ".format(num_samples))

''' how to obtain a single sample of a single class '''
for class_idx in range(num_class):
    for n in range(num_samples):
        myarray = decoded_dict[class_idx][n][:, :, :]    
        

''' select a few samples from a few classes for demo '''
class_number_list = [10, 8, 2, 9]
num_samples = 6
fig, axes = plt.subplots(nrows=num_samples, ncols=len(class_number_list)) 

for ax, col in zip(axes[0], class_number_list):
    ax.set_title("Class: " + str(col))

# for ax, row in zip(axes[:,0], rows):
#     ax.set_ylabel(row, rotation=0, size='large')

for col_idx in range(len(class_number_list)):
    for row_idx in range(0, num_samples):
        class_number = class_number_list[col_idx]
        myarray = decoded_dict[class_number][row_idx][:, :, :]    
        width, length, timestep = myarray.shape
        myarray = myarray.reshape((-1, timestep))
        axes[row_idx, col_idx].imshow(myarray)
figname = data_dir+"figures/decoded_I_{}_class_{}_samples.png".format(len(class_number_list), num_samples)
plt.savefig(figname, dpi=1200)

plt.show()
# plt.close('all')
        