import numpy as np
import os, sys
import pandas as pd
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

import pickle
from random import shuffle
import copy
import glob
import shutil

obj_name_dict = {
    'bathTowel': 3,
    'carpetNet': 1,
    'cork': 12,
    'cotton': 2,
    'cuisionFoam': 19,
    'eva': 13,
    'fakeLeather': 4,
    'felt': 6,
    'fiberBoard': 15,
    'metal': 20,
    'paper1': 14,
    'paper2': 8,
    'polypropileno': 5,
    'polypropilenoSmooth': 9,
    'softMaterial1': 11,
    'softMaterial2': 7,
    'spongeWhiteSmall': 18,
    'styrofoam': 17,
    'thinPolypropylene': 10,
    'woodHard': 16,
}


def rename_pt_slide_data(fout=None):
    data_dir = 'slide_6_10/'
    dest_dir = 'slide_6_10_rename/'
    class_names = set()
    
    files = glob.glob(data_dir+"*.pt")
    for file in files:
        filename = file.split("/")[-1]
        class_name = filename.split("_")[0]
        class_names.add(class_name)
        sample_num = int(filename.split("_")[-1].split(".pt")[0])
        if class_name in obj_name_dict:
            # print("find key ", class_name, obj_name_dict[class_name])
            new_filename = "mat" + str(obj_name_dict[class_name]-1) + "_" + str(sample_num) + ".pt"
            dest = dest_dir + new_filename
            newPath = shutil.copy(file, dest)

        else:
            print("{} not found in dict, skip".format(class_name))
    num_class = len(class_names)
    num_sample = len(files)//num_class
    print("In {} we have {} samples for {} classes".format(data_dir, num_sample, num_class))


def rename_pt_BioTac_data():
    ''' swap the indexes of the metal and the transparent plastic '''
    data_dir = 'tactile_img_Feb/'
    dest_dir = 'tactile_img_Feb_rename/'
    class_names = set()
    
    files = glob.glob(data_dir+"*.pt")
    for file in files:
        filename = file.split("/")[-1]
        class_name = int(filename.split("_")[0][3:])
        class_names.add(class_name)
        sample_num = int(filename.split("_")[-1].split(".pt")[0])
        if class_name == 20:
            class_name = 21
        elif class_name == 21:
            class_name= 20
    
        new_filename = "mat" + str(class_name-1) + "_" + str(sample_num-1) + ".pt"
        dest = dest_dir + new_filename
        newPath = shutil.copy(file, dest)

    num_class = len(class_names)
    num_sample = len(files)//num_class
    print("In {} we have {} samples for {} classes".format(dest_dir, num_sample, num_class))


def create_equal_dataset(num_class=20, num_sample=50):
    # data_dir1 = 'tactile_img_Feb_rename/'
    # data_dir2 = 'slide_6_10_rename/'
    dest_dir = 'data/'
    data_dir3 = 'BioTac_19_rename/'

    # for file in glob.glob(data_dir1+"*.pt"):
    #     filename = file.split("/")[-1]
    #     matNum, sampleNum = filename.split("_")
    #     matNum = int(matNum[3:])
    #     sampleNum = int(sampleNum[:-3])
    #     if matNum < num_class and sampleNum < num_sample:
    #         new_filename = "BioTac_" + filename
    #         dest = dest_dir + new_filename
    #         newPath = shutil.copy(file, dest)

    # for file in glob.glob(data_dir2+"*.pt"):
    #     filename = file.split("/")[-1]
    #     matNum, sampleNum = filename.split("_")
    #     matNum = int(matNum[3:])
    #     sampleNum = int(sampleNum[:-3])
    #     if matNum < num_class and sampleNum < num_sample:
    #         new_filename = "Icub_" + filename
    #         dest = dest_dir + new_filename
    #         newPath = shutil.copy(file, dest)
    
    for file in glob.glob(data_dir3+"*.pt"):
        filename = file.split("/")[-1]
        matNum, sampleNum = filename.split("_")
        matNum = int(matNum[3:])
        sampleNum = int(sampleNum[:-3])
        if matNum < num_class and sampleNum < num_sample:
            new_filename = dest_dir + "BT19/" + "BT19_" + filename
            dest = new_filename
            newPath = shutil.copy(file, dest)
        

def generate_pkl_from_pt(data_dir, sensors, fout_postfix, num_class=20, num_sample=50, test_ratio=0.2, valid_ratio=0.2, seed=1):
    
    if seed:
        np.random.seed(seed)
    ids = [i for i in range(num_sample)]
    test_inds = np.random.choice(ids, int(num_sample*test_ratio), replace=False)
    print("test_inds are: ", test_inds)
    for x in test_inds:
        ids.remove(x) 
    valid_inds = np.random.choice(ids, int(num_sample*valid_ratio), replace=False)
    print("valid_inds are: ", valid_inds)

    files = sorted(glob.glob(data_dir+"*.pt"))
    for sensor in sensors:

        train_labels = {}
        valid_labels = {}
        test_labels = {}

        train_count= 0
        valid_count = 0
        test_count = 0

        train_ids = []
        valid_ids = []
        test_ids = []

        for file in files:
            ss, matName, sampleNum = file.split("/")[-1].split("_")
            name_convention = file.split("/")[-1][:-3]
            sampleNum = int(sampleNum[:-3])
            if ss == sensor:
                if sampleNum in test_inds:
                    test_ids.append(name_convention)
                    test_labels[name_convention] = int(matName[3:])
                    test_count += 1
                elif sampleNum in valid_inds:
                    valid_ids.append(name_convention)
                    valid_labels[name_convention] = int(matName[3:])
                    valid_count += 1
                else:
                    train_ids.append(name_convention)
                    train_labels[name_convention] = int(matName[3:])
                    train_count += 1
        fout = sensor + fout_postfix
        print("fout ", fout)
        assert len(train_ids) == len(train_labels), "different train len: {} ids, {} labels".format(len(train_ids),len(train_labels))
        assert len(test_ids) == len(test_labels), "different valid len: {} ids, {} labels".format(len(valid_ids),len(valid_labels))
        assert len(test_ids) == len(test_labels), "different test len: {} ids, {} labels".format(len(test_ids),len(test_labels))
        pickle.dump([train_ids, train_labels, valid_ids, valid_labels, test_ids, test_labels], open(fout,'wb'))
        print("dump train & valid & test ids to {}, with {} train, {} valid and {} test".format(fout, train_count, valid_count, test_count))


def copy_equal_dataset(num_class=20, num_sample=50):
    data_dir = 'data/'

    for file in glob.glob(data_dir+"*.pt"):
        ss, matName, sampleNum = file.split("/")[-1].split("_")
        # name_convention = file.split("/")[-1][:-3]
        # sampleNum = int(sampleNum[:-3])
        
        dest = data_dir + ss + "/" + file.split("/")[-1]
        newPath = shutil.copy(file, dest)




def generate_raw_BioTac_data():
    data_dir = "/home/ruihan/Documents/material_data_Feb/"
    dst_dir = "/home/ruihan/Joint-classifier/BioTac_19/"
    num_class = 21
    num_sample = 50
    names = []
    for i in range(1,num_class+1):
        names.append('mat'+ str(i))

    for obj_name in names:
        # read object data file
        for sample_num in range(1, num_sample+1):
            parent_name = data_dir + obj_name + '/' + obj_name + '_' + str(sample_num)
            csv_file = parent_name +'_bio.csv'

            # read csv and filter by time window
            df = pd.read_csv(csv_file, header=0, index_col=False)
            df = df.drop(labels=['timestamp'], axis=1)
            
            name_convention = obj_name + '_' + str(sample_num) # save .pt files together outside the subfolders of each material

            if df.shape[0] == 799:
                last_row = df.iloc[-1:]
                df = df.append(last_row)
            df_arr = df.values.astype(int)
        
            df = pd.DataFrame(df_arr[:,25:],dtype=np.int8)
            print(name_convention, "df_arr", df_arr.shape, "df", df.shape)
            torch.save(torch.from_numpy(df.to_numpy()), dst_dir + name_convention + '.pt')
    
def rename_raw_BioTac_data():

    ''' swap the indexes of the metal and the transparent plastic '''
    data_dir = 'BioTac_19/'
    dest_dir = 'BioTac_19_rename/'
    class_names = set()
    
    files = glob.glob(data_dir+"*.pt")
    for file in files:
        filename = file.split("/")[-1]
        class_name = int(filename.split("_")[0][3:])
        class_names.add(class_name)
        sample_num = int(filename.split("_")[-1].split(".pt")[0])
        if class_name == 20:
            class_name = 21
        elif class_name == 21:
            class_name= 20
    
        new_filename = "mat" + str(class_name-1) + "_" + str(sample_num-1) + ".pt"
        dest = dest_dir + new_filename
        # print(file, dest)
        newPath = shutil.copy(file, dest)

    num_class = len(class_names)
    num_sample = len(files)//num_class
    print("In {} we have {} samples for {} classes".format(dest_dir, num_sample, num_class))

# rename_pt_slide_data()
# rename_pt_BioTac_data()
# create_equal_dataset()

# data_dir = 'data/'
# sensors = ['BioTac', 'Icub']
# fout_postfix = "_20_50.pkl"
# generate_pkl_from_pt(data_dir, sensors, fout_postfix, num_class=20, num_sample=50)

# # check the pkl file
# for sensor in sensors:
#     datafile = sensor + fout_postfix
#     [train_ids, train_labels, test_ids, test_labels] = pickle.load(open(datafile, 'rb'))
#     assert len(train_ids) == len(train_labels), "different train len: {} ids, {} labels".format(len(train_ids),len(train_labels))
#     assert len(test_ids) == len(test_labels), "different test len: {} ids, {} labels".format(len(test_ids),len(test_labels))
#     print("For {}: {} train {} test".format(sensor, len(train_ids), len(test_ids)))

# copy_equal_dataset()
# generate_raw_BioTac_data()
# rename_raw_BioTac_data()
create_equal_dataset()