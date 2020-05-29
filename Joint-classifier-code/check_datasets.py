import os, sys
import numpy as np 
import pickle
import glob
import torch

'''
check original data
'''
# dict1 = pickle.load(open("slide_6_10.pkl", 'rb'))
# dict2 = pickle.load(open("BioTac.pkl", 'rb'))

# print(dict1)
# print(dict2)

# rename the slide dataset
rename_dict = {
    'bathTowel': 3,
    'carpetNet': 7,
    'cork': 12,
    'cotton': 2,
    'cuisionFoam': 19,
    'eva': 13,
    'fakeLeather': 9,
    'felt': 6,
    'fiberBoard': 15,
    'metal': 21,
    'spongeWhiteSmall': 18,
    'styrofoam': 17,
    'thinPolypropylene': 10,
    'woodHard': 16,

    'cardBoard': 22,
    'denim': 23,
    'paper1': 24,
    'paper2': 25,
    'polypropileno': 26,
    'polypropilenoSmooth': 27,
    'softMaterial1': 28,
    'softMaterial2': 29,
    'yogaMat': 30
}

''' 
rename the pt files for slide_6_10, note BioTac idx starts from 1, while org slide_6_10 starts from 0
'''
# # use torch instead of pickle
# directory = "/home/ruihan/Joint-classifier/slide_6_10"
# new_dir = "/home/ruihan/Joint-classifier/slide_6_10_rename"

# for filename in os.listdir(directory):
#     mat, idx = filename.split('_')
#     idx = idx.split('.')[0]
#     idx = int(idx)+1
#     # print(mat, idx) # polypropileno 16.pt

#     if mat in rename_dict:
#         new_filename = 'mat' + str(rename_dict[mat]) + '_' + str(idx) + '.pt'
#         X = torch.load(os.path.join(directory,filename))
#         torch.save(X, os.path.join(new_dir,new_filename))
#     else:
#         raise ValueError("not existing ", mat)


'''
prepare the renamed .pkl file for slide_6_10 following the rename_dict labels
'''

# test_inds = [1, 9, 12]

# num_sample_tot = 62
# num_sample_test = len(test_inds)
# num_sample_train = num_sample_tot - num_sample_test

# # name of the object
# num_class = 30
# object_names = []
# for i in range(1,num_class+1):
#     object_names.append('mat'+ str(i))

# data_dir = 'slide_6_10_rename/'
# fout = 'slide_6_10_rename.pkl'
# slide_obj_names = set()
# for filename in os.listdir(data_dir):
#     mat = filename.split('_')[0]
#     slide_obj_names.add(mat)
# print(slide_obj_names)
# slide_obj_names = list(slide_obj_names)

# # initialize empty arrays to store ids per class
# train_labels = {}
# test_labels = {}

# train_count= 0
# test_count = 0

# train_ids = []
# test_ids = []
    
# for obj_name in slide_obj_names:
#     # read each slide
#     for i in range(1, num_sample_tot+1):
#         name_convention = obj_name + '_' + str(i)
#         if i in test_inds:
#             test_ids.append(name_convention)
#             test_labels[name_convention] = int(obj_name[3:])-1
#             # print(name_convention, test_labels[name_convention])
#             test_count += 1
#         else:
#             train_ids.append(name_convention)
#             train_labels[name_convention] = int(obj_name[3:])-1
#             train_count += 1
        
# pickle.dump([train_ids, train_labels, test_ids, test_labels], open(fout,'wb'))

# # check renamed pkl file
# [train_ids, train_labels, test_ids, test_labels] = pickle.load(open(fout, 'rb'))
