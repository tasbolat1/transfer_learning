import os, sys
import numpy 
import glob

counter = 0

for file in glob.glob("/home/students/student6_16/projects/BioTac-classifier/BioTac_info_400/*.ckpt"):
    num_epochs = file.split('_')[-1][:-5]
    if num_epochs.isdigit():
        num_epochs = int(num_epochs)
        if num_epochs%100!=0:
            # remove the file
            os.remove(file)
        print("remove ", file)
    else:
        print("not interested pass {}".format(file))

