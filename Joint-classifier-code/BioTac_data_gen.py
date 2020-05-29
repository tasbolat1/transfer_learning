import numpy as np
import os, sys
import glob
import pandas as pd
from scipy.signal import savgol_filter
import math

import torch
import torch.nn as nn

import pickle
from random import shuffle
import copy
# interpret tactile sensor data on iCub.
# ref: http://wiki.icub.org/wiki/Tactile_sensors_(aka_Skin)

# macros for creating BioTac imgs
TACTILE_IMAGE_ROWS = 8
TACTILE_IMAGE_COLS = 5
ELECTRODES_INDEX_ROWS = np.array([1, 3, 4, 5, 6, 7, 0, 1, 1, 2, 1, 3, 4, 5, 6, 7, 3, 6, 7])
ELECTRODES_INDEX_COLS = np.array([0, 1, 1, 0, 1, 0, 2, 1, 3, 2, 4, 3, 3, 4, 3, 4, 2, 2, 2])

NORM = 'stdnorm' # 'stdnorm', 'featurescaling'
FILL_STRATEGY = 'cero2mean' # 'cero2lesscontact' 'cero2mean'

num_sample = 15
num_class = 21
# name of the object
object_names = []
for i in range(1,num_class+1):
    object_names.append('mat'+ str(i))
# print(object_names)

# # selected window size
# DEPTH_SIZE = 75

def filter_picks(ll):
    # choose picks
    picks = []
    for i in ll.index:
        add_it = True
        for j in picks:
            if i-j <= 40:
                add_it = False
                break
        if(add_it):
            picks.append(i)
    return picks

def generate_BioTac_data(names, test_inds, fout):
    train_labels = {}
    test_labels = {}

    train_count= 0
    test_count = 0

    data_dir = 'material_data/'
    train_ids = []
    test_ids = []
    
    print("generate BioTac data for: ")
    print(names)
    for obj_name in names:
        # read object data file
        for sample_num in range(1, num_sample+1):
            parent_name = data_dir + obj_name + '/' + obj_name + '_' + str(sample_num)
            csv_file = parent_name +'_bio.csv'

            # read csv and filter by time window
            df = pd.read_csv(csv_file, header=0, index_col=False)
            df = df.drop(labels=['timestamp'], axis=1)
            
            name_convention = obj_name + '_' + str(sample_num) # save .pt files together outside the subfolders of each material
            print(name_convention, df.shape[0])
            if df.shape[0] == 599:
                last_row = df.iloc[-1:]
                df = df.append(last_row)
            df_arr = df.values.astype(int)
            torch.save(torch.from_numpy(df_arr), data_dir + name_convention + '.pt')
            print(name_convention, df.shape)

            if sample_num in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1

    pickle.dump([train_ids, train_labels, test_ids, test_labels], open(fout,'wb'))

def generate_datafile_from_pt(data_dir, fout, num_class=21, test_ratio=0.2, seed=1):
    files = glob.glob(data_dir+"*.pt")
    num_sample = len(files)//num_class
    print("In {} we have {} samples for {} classes".format(data_dir, num_sample, num_class))

    names = []
    for i in range(1,num_class+1):
        names.append('mat'+ str(i))

    print("generate BioTac data for: ")
    print(names)

    if seed:
        np.random.seed(seed)
    test_inds = np.random.choice([i for i in range(1, num_sample+1)], int(num_sample*test_ratio), replace=False)
    print("test_inds are: ", test_inds)
    
    train_labels = {}
    test_labels = {}

    train_count= 0
    test_count = 0

    train_ids = []
    test_ids = []
    
    for obj_name in names:
        for sample_num in range(1, num_sample+1):            
            name_convention = obj_name + '_' + str(sample_num) # save .pt files together outside the subfolders of each material
            # print(name_convention)

            if sample_num in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1
    pickle.dump([train_ids, train_labels, test_ids, test_labels], open(fout,'wb'))
    print("dump train & test ids to {}".format(fout))

def generate_classwise_slide_data(names, test_inds, fout, num_class=23, C=1, W=6, H=10):
    num_sample_tot = 62
    num_sample_test = len(test_inds)
    num_sample_train = num_sample_tot - num_sample_test
    # images_train = np.zeros((num_class, num_sample_train, C, W, H), dtype=np.uint8)
    # labels_train = np.zeros((num_class, num_sample_train), dtype=int)
    # images_val = 0
    # labels_val = 0
    # images_test = np.zeros((num_class, num_sample_test, C, W, H), dtype=np.uint8)
    # labels_test = np.zeros((num_class, num_sample_test), dtype=int)


    train_labels = {}
    test_labels = {}

    train_count= 0
    test_count = 0

    data_dir = 'slide_6_10/'
    train_classes_ids = []
    test_classes_ids = []
    train_classes_labels = []
    test_classes_labels = []

    print("generate classwise slide data for: ")
    print(names)
    
    for obj_name in names:
        # initialize empty arrays to store ids per class
        train_ids = []
        test_ids = []
        train_labels = {}
        test_labels = {}
        # read data
        df_tactile = read_tactile(obj_name)
        df_encoder, peaks = read_encoder(obj_name)

        # read each slide
        for i in range(num_sample_tot):
            df_chunk = df_tactile.loc[peaks[2*i] : peaks[2*i] + DEPTH_SIZE -1, :].values.reshape(6,DEPTH_SIZE,10)
            df_chunk = np.swapaxes(df_chunk, 1,2).astype('float64')

            name_convention = obj_name + '_' + str(i)
            torch.save(torch.from_numpy(df_chunk), data_dir + name_convention + '.pt')

            if i in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1
            
            
        # list of list
        test_classes_ids.append(test_ids) 
        train_classes_ids.append(train_ids)
        # list of dict
        test_classes_labels.append(test_labels)
        train_classes_labels.append(train_labels) 
            
    pickle.dump([train_classes_ids, train_classes_labels, test_classes_ids, test_classes_labels], open(fout,'wb'))

# ref: https://github.com/yayaneath/biotac-sp-images/blob/master/code/Dataset%20Processing.ipynb
def get_neighbours(tactile_image, cell_x, cell_y):
    pad = 2
    padded_x = cell_x + pad
    padded_y = cell_y + pad
    
    padded = np.pad(tactile_image, ((pad, pad), (pad, pad)), 'constant') #0s
    
    neighbours_xs = [padded_x - 1, padded_x - 1, padded_x - 1, 
                     padded_x, padded_x, 
                     padded_x + 1, padded_x + 1, padded_x + 1]
    neighbours_ys = [padded_y - 1, padded_y, padded_y + 1,
                     padded_y - 1, padded_y + 1,
                     padded_y - 1, padded_y, padded_y + 1]
    num_neighbours = len(neighbours_xs)
    neighbours = []
    
    for i in range(num_neighbours):
        some_x = neighbours_xs[i]
        some_y = neighbours_ys[i]
        neighbours.append(padded[some_x, some_y])

    return neighbours

def ceros_2_mean(tactile_image):
    prev_tactile_image = np.copy(tactile_image)
    cero_xs, cero_ys = np.where(tactile_image == 0)

    for i in range(len(cero_xs)):
        cell_x = cero_xs[i]
        cell_y = cero_ys[i]
        cell_neighs = get_neighbours(prev_tactile_image, cell_x, cell_y)
        cell_neighs = [value for value in cell_neighs if value > 0.0]

        if len(cell_neighs) > 0:
            tactile_image[cell_x, cell_y] = np.mean(cell_neighs)
            
    return tactile_image    


def create_finger_tactile_image(finger_biotac, normalization, fill_strategy=1):
    # rearrange the 19 electrodes in the img
    tactile_image = np.zeros(shape=(TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
    tactile_image[ELECTRODES_INDEX_ROWS, ELECTRODES_INDEX_COLS] = finger_biotac
    
    if fill_strategy == 'cero2lesscontact':
        # Strategy 1 - Fill with less contacted value
        # The maximum value corresponds to the less contacted electrode
        max_value = np.max(finger_biotac)
        tactile_image[tactile_image == 0] = max_value
    elif fill_strategy == 'cero2mean':
        # Strategy 2 - Fill with neighbours average
        tactile_image = ceros_2_mean(tactile_image)
        
        # Repeat in case that there were cells with no values as neighbours, they will now
        if np.min(tactile_image) == 0.0:
            tactile_image = ceros_2_mean(tactile_image)
    
    # TODO: find a proper value to replace when std is 0, which leads to nan tactile_img
    if normalization == 'stdnorm':
        tactile_image = (tactile_image - np.mean(tactile_image)) / np.max([np.std(tactile_image), 0.1])
    elif normalization == 'featurescaling':
        tactile_image = (tactile_image - np.min(tactile_image)) / (np.max(tactile_image) - np.min(tactile_image))
    
    return tactile_image

def pt2img(names, test_inds, fout, generate_label=False):

    # save labels

    train_labels = {}
    test_labels = {}

    train_count= 0
    test_count = 0

    data_dir = 'pt_data/'
    train_ids = []
    test_ids = []
    
    print("generate BioTac data for: ")
    print(names)

    print("reverse engineering to generate tactile images")
    for obj_name in names:
        # read object data file
        for sample_num in range(1, num_sample+1):
            name_convention = obj_name + '_' + str(sample_num)
            # load torch tensor and convert to numpy array
            df_arr = torch.load('pt_data/smooth_ele/' + name_convention + '.pt').numpy()

            # # print(df_arr.shape) #(600, 44)
            # # convert numpy array to pandas dataframe
            # df = pd.DataFrame(df_arr[:,25:],dtype=np.int8)

            df = pd.DataFrame(df_arr,dtype=np.int8)
            # create tactile image
            # print("df shape", df.shape) # (600, 19)
            tactile_images = np.zeros(shape=(df.shape[0], TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
            for sample in range(df.shape[0]):
                one_grasp = df.iloc[sample].values
                # BioTac (we are using) 19, while BioTacSP is 24
                tactile_images[sample] = create_finger_tactile_image(one_grasp[0:19], normalization=NORM, fill_strategy=FILL_STRATEGY)
            
            # # print a sample tactile image
            # some_grasp = 0
            # print("a sample tacktile image from", tactile_images.shape)
            # print(tactile_images[some_grasp])
            
            # save the images
            whole_images = tactile_images
            torch.save(torch.from_numpy(whole_images), 'tactile_img_data_smooth/' + name_convention + '.pt')

            if sample_num in test_inds:
                test_ids.append(name_convention)
                test_labels[name_convention] = names.index(obj_name)
                test_count += 1
            else:
                train_ids.append(name_convention)
                train_labels[name_convention] = names.index(obj_name)
                train_count += 1


            
    if generate_label:
        pickle.dump([train_ids, train_labels, test_ids, test_labels], open(fout,'wb'))



def check_pt2img(obj_name, sample_num):

    data_dir = 'pt_data/'
    name_convention = obj_name + '_' + str(sample_num)

    print("check BioTac data for: ", name_convention)

    # load torch tensor and convert to numpy array
    df_arr = torch.load('pt_data/' + name_convention + '.pt').numpy()
    # print(df_arr.shape) #(600, 44)
    # convert numpy array to pandas dataframe
    df = pd.DataFrame(df_arr[:,25:],dtype=np.int8)
    # print(df.shape) # (600, 19)
    # print(df.describe())

    # create tactile image
    tactile_images = np.zeros(shape=(df.shape[0], TACTILE_IMAGE_ROWS, TACTILE_IMAGE_COLS))
    for sample in range(df.shape[0]):
        one_grasp = df.iloc[sample].values
        is_nan = np.isnan(one_grasp)
        data_nan = is_nan==1

        if True in data_nan:
            print("nan data", sample)
            result = np.where(data_nan==True)
            print(result)
        # BioTac (we are using) 19, while BioTacSP is 24
        tactile_img = create_finger_tactile_image(one_grasp[0:19], normalization=NORM, fill_strategy=FILL_STRATEGY)

        is_nan = np.isnan(tactile_img)
        data_nan = is_nan==1

        if True in data_nan:
            print("nan img", sample)
            result = np.where(data_nan==True)
            print(result)


# fout = 'BioTac.pkl'
# random_idxes = np.random.choice([i for i in range(1, num_sample+1)], int(num_sample*0.2))
# generate_BioTac_data(object_names, random_idxes, fout)
# pt2img(object_names, random_idxes, fout, generate_label=False)

' generate a new datafile with specified pt img data folder '
# data_dir = "/home/ruihan/BioTac-classifier/tactile_img_comb/"
# fout = "BioTest.pkl"
# generate_datafile_from_pt(data_dir=data_dir, fout=fout)
# # check the datafile
# with open(fout, 'rb') as fin:
#     datafile = pickle.load(fin)
# print("check datafile")
# print(datafile)