# Calculating Optical Flow via TV-L1

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import os

base_path = '../optflow_vis_test/'
path_list = os.listdir(base_path)

print(path_list)
dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()

for i in range(len(path_list)):

    if not os.path.isdir('../optflow_vis_test/'+path_list[i]+'/linear_o'):
        os.makedirs('../optflow_vis_test/'+path_list[i]+'/linear_o')

    
    new_path = '../optflow_vis_test/'+path_list[i]+'/linear_o/'
    
    path = glob.glob(base_path+path_list[i]+'/*.npy')
    path = sorted(path)
    print(path)
    for j in range(0,18):
        # Reading input data
        img0 = np.load(path[j]) # input image at t-10
        img1 = np.load(path[j+1]) # input image at t

        flow_1 = dtvl1.calc(img1, img0, None)*1.

        # saving optical flow field
        np.save(new_path+'o_vector_202007301000+'+str((j+1)*10)+'min.npy',flow_1)

        h,w,_ = flow_1.shape
        flow_1[:,:,0] += np.arange(w)
        flow_1[:,:,1] += np.arange(h)[:,np.newaxis]
        prevImg = cv2.remap(img1, flow_1, None, cv2.INTER_CUBIC)
        
        #saving future frame at t+10
        np.save(new_path+'o_precipitation_202007301000+'+str(10*(j+1))+'min.npy',prevImg)
