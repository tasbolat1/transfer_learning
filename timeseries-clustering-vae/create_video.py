import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os, glob

img_dir = "/home/ruihan/Desktop/MLDA/Dec_imgs"
cl = "C2"

S = ['S0', 'S1', 'S2']
T = 75
size= (300, 450)
for s in S:
    video_name = "/home/ruihan/Desktop/MLDA/Dec_imgs/Dec_{}_{}_T{}.avi".format(cl, s, str(T))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size, True)
    
    for t in range(T):
        name = "/home/ruihan/Desktop/MLDA/Dec_imgs/Dec_{}_{}_T{}_c.png".format(cl, s, str(t))
        # print(name)
        img = cv2.imread(name)

        # print(img.shape) # (288, 432, 3)
        height, width, frames = img.shape

        img_frame = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

        stacked_img = np.stack(img_frame, axis=-1)
        out.write(np.uint8(stacked_img))

        # cv2.imshow("img", img_frame)
        # cv2.waitKey()
        # break

    out.release() 


sys.exit()



# tactile_img_pt1 = "/home/ruihan/BioTac-classifier/tactile_img_data/mat7_6.pt"
# tactile_img1 = torch.load(tactile_img_pt1)

# tactile_img_pt2 = "/home/ruihan/BioTac-classifier/tactile_img_data/mat6_7.pt"
# tactile_img2 = torch.load(tactile_img_pt2)

# print(tactile_img.size()) #torch.Size([600, 8, 5])
# tactile_img.unsqueeze_(0)
# print("unsqueeze", tactile_img.size()) #torch.Size([600, 8, 5])

''' display tactile_img[seq, :, :] using matplotlib '''
# fig = plt.figure()

# plt.axis('off')

# plt.subplot(2, 2, 1)
# plt.imshow(tactile_img1[0, :, :], interpolation='nearest', )

# plt.subplot(2, 2, 2)
# plt.imshow(tactile_img1[447, :, :], interpolation='nearest', )

# plt.subplot(2, 2, 3)
# plt.imshow(tactile_img2[0, :, :], interpolation='nearest', )

# plt.subplot(2, 2, 4)
# plt.imshow(tactile_img2[447, :, :], interpolation='nearest', )

# plt.colorbar()
# plt.show()

'''creat video from image sequence'''
# convert to numpy
tactile_im1 = tactile_img2.numpy()

# change shape (600, 8, 5) to (8, 5, 600)
tactile_im1 = np.moveaxis(tactile_im1, 0, -1)
height, width, frames = tactile_im1.shape
print("tactile_im1", tactile_im1.shape) # (8, 5, 600)

scale_percent = 2000 # percent of original size
width = int(tactile_im1.shape[1] * scale_percent / 100)
height = int(tactile_im1.shape[0] * scale_percent / 100)
size = (width, height)
print("size", size)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter("tactile_mat6_7.avi", fourcc, 30, size)

for i in range(frames):
    # if keyboard.is_pressed('q'):
    #     print("exit")
    #     sys.exit()
    # resize image
    img = cv2.resize(tactile_im1[:,:,i], size, interpolation = cv2.INTER_AREA)
    stacked_img = np.stack((img*200,)*3, axis=-1)
    # print("stacked_img", stacked_img.shape)
    out.write(np.uint8(stacked_img))
    # cv2.imshow("img", np.uint8(img*200))
    # cv2.waitKey(0)
    # print(img)
    
out.release() 