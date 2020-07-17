import numpy as np 
import os,sys
import pickle

dir = "/home/ruihan/project/transfer_learning/timeseries-clustering-vae/models_and_stats"
filename = "BT19Icub_joint_ae_wrB_0.01_wcB_1_wrI_0.01_wcI_1_wC_1_0.pkl"
with open(os.path.join(dir, filename), "rb") as fin:
    logs = pickle.load(fin)
print(len(logs))
print(logs)