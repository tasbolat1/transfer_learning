import numpy as np 
import torch 
import os, sys
import glob

data_dir = "/home/ruihan/Joint-classifier/data/BT_19"
newdata_dir = "/home/ruihan/Joint-classifier/data/BT19_400"
# count = 0
# os.chdir(data_dir)
# for file in list(glob.glob('Bio*.pt')):   
#     count += 1             
#     sample = torch.load(file)
#     print(file)
#     print(sample.shape)
#     sys.exit()
#     new_sample = sample[200:600,:]
#     print(file)
#     print(sample.shape, new_sample.shape)
#     torch.save(new_sample, os.path.join(newdata_dir, file))
# print("process {} files".format(count))

# os.chdir(newdata_dir)
# data_all = []
# name_all = []
# for file in sorted(list(glob.glob("*.pt"))):
#     sample = torch.load(file)
#     sample = sample.transpose(0,1)
#     data_all.append(sample)
#     name_all.append(file)
# print(type(data_all), len(data_all))
# data_all = torch.stack(data_all)
# print(data_all.shape)
# torch.save(data_all, os.path.join(newdata_dir, "BioTac_all.pt"))
# with open("/home/ruihan/Joint-classifier/data/labels_B.txt", 'w') as output:
#     for row in name_all:
#         output.write(str(row) + '\n')

file_dir = "/home/ruihan/Joint-classifier/data/"
labels = []
with open(file_dir+"labels_B.txt", "r") as fin:
    for filename in fin.readlines():
        label = filename.split("_")[1]
        label = int(label[3:])
        labels.append(label)
labels = np.array(labels)
np.save(file_dir+"all_labels.npy", labels)
read_label = np.load(file_dir+"all_labels.npy")
print(read_label)