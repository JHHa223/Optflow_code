import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import os
import shutil

base_path = '../optflow_vis_test/'
path_list = os.listdir(base_path)

print(path_list)

# The function for linear regression stage
def MLR_optflow(img1,img2,img3,img4,img_gt):
        
    W = np.array([1/4,1/4,1/4,1/4])
    learning_rate = 1e-7
    error_prev = 1

    x_size = img1.shape[0]
    y_size = img1.shape[1]
    
    for epoch in range(100):

        img1_r = img1.reshape(x_size*y_size)
        img2_r = img2.reshape(x_size*y_size)
        img3_r = img3.reshape(x_size*y_size)
        img4_r = img4.reshape(x_size*y_size) 
    
        I_arr = np.r_[[img1_r],
                      [img2_r],
                      [img3_r],
                      [img4_r]]


        I_f = W[0]*img1_r + W[1]*img2_r + W[2]*img3_r + W[3]*img4_r
        I_f = I_f/np.sum(W)

        error = np.mean((I_f.astype(np.float64)-img_gt.astype(np.float64).reshape(x_size*y_size))**2)

        if error > error_prev:
            print(epoch, error, error_prev, "break")
            break
    
        W_grad = learning_rate*I_arr[:,:]@((I_f.astype(np.float64) - img_gt.astype(np.float64).reshape(x_size*y_size)))

        W = W - W_grad

        error_prev = error
        
        print(W/(np.sum(W)), error)

    return I_f.reshape(x_size,y_size)


for i in range(len(path_list)):

    if not os.path.isdir('../optflow_vis_test/'+path_list[i]+'/linear_o_opt'):
        os.makedirs('../optflow_vis_test/'+path_list[i]+'/linear_o_opt')

    
    new_path = '../optflow_vis_test/'+path_list[i]+'/linear_o_opt/'
    
    path = glob.glob(base_path+path_list[i]+'/*.npy')
    path = sorted(path)

# Reading input dataset
    img0 = np.load(path[0]) # input image: t - 10
    img1 = np.load(path[1]) # input image: t
    img_gt = np.load(path[2]) # ground truth: t + 10

# Optical flow calculation
#   TV-L1
    dtvl1_0= cv2.optflow.DualTVL1OpticalFlow_create()
    flow_tvl1_0 = dtvl1_0.calc(img1, img0, None)
#   DeepFlow        
    d_deepflow_0 = cv2.optflow.createOptFlow_DeepFlow()  
    flow_deepflow_0 = d_deepflow_0.calc(img1, img0, None)
#   Farneback
    dfar_0 = cv2.optflow.createOptFlow_Farneback()
    flow_far_0 = dfar_0.calc(img1, img0, None)
#   PCA Flow
    d_pca_0 = cv2.optflow.createOptFlow_PCAFlow()
    flow_pca_0 = d_pca_0.calc(img1, img0, None)
        
    h,w,_ = flow_tvl1_0.shape
    flow_tvl1_0[:,:,0] += np.arange(w)
    flow_tvl1_0[:,:,1] += np.arange(h)[:,np.newaxis]
        
    h,w,_ = flow_deepflow_0.shape
    flow_deepflow_0[:,:,0] += np.arange(w)
    flow_deepflow_0[:,:,1] += np.arange(h)[:,np.newaxis]         

    h,w,_ = flow_far_0.shape
    flow_far_0[:,:,0] += np.arange(w)
    flow_far_0[:,:,1] += np.arange(h)[:,np.newaxis] 

    h,w,_ = flow_pca_0.shape
    flow_pca_0[:,:,0] += np.arange(w)
    flow_pca_0[:,:,1] += np.arange(h)[:,np.newaxis]     

    # Generating future frames via various optical flow algorithms 
    I_tvl1_0 = cv2.remap(img1, flow_tvl1_0, None, cv2.INTER_CUBIC)
    I_deepflow_0 = cv2.remap(img1, flow_deepflow_0, None, cv2.INTER_CUBIC)
    I_far_0 = cv2.remap(img1, flow_far_0, None, cv2.INTER_CUBIC)
    I_pca_0 = cv2.remap(img1, flow_pca_0, None, cv2.INTER_CUBIC)

    # Linear regression stage        
    img_f = MLR_optflow(I_tvl1_0,I_deepflow_0,I_far_0,I_pca_0,img_gt)

    # Saving future frame at t + 10    
    np.save(new_path+'o_precipitation_202007301000+10min.npy',img_f)    

        
        
