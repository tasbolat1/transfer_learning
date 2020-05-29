import os,sys
import numpy as np 
import shutil

src_dir = "/home/ruihan/BioTac-classifier/tactile_img_data/"
dest_dir = "/home/ruihan/BioTac-classifier/tactile_img_comb/"
for matNum in range(1, 22):
    for sampleNum in range(1, 6):
        # copy file from src_dir to dest_dir
        src = src_dir + "mat" + str(matNum) + "_" + str(sampleNum) + ".pt"
        dest = dest_dir + "mat" + str(matNum) + "_" + str(sampleNum+50) + ".pt"
        newPath = shutil.copy(src, dest)
        print("process ", newPath)
