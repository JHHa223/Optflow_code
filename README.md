# Optflow v1.0: A Deep Learning Model for Precipitation Nowcasting Using Multiple Optical Flow Algorithms

## Introduction
Optflow v1.0 is a deep learning model that utilizes the optical flow method to enhance the performance of precipitation nowcasting.

## Model Architecture 
A Deep Learning Model for Precipitation Nowcasting consists of two parts.

Part I. Optical flow calculation and linear regression stage

Part II. U-Net architecture for training the nonlinear motion of precipitation fields

## Part I. Optical flow calculation and linear regression stage

OpenCV library is used here to estimate the optical flow field.

The codes for optical flow calculation are provided at the directory "optical_flow".

### Example: Optical flow estimation
```python
#### Single-Temporal Model ####

# Reading input data
img0 = np.load('path/img/radar_001.npy') # input image at t-10
img1 = np.load('path/img/radar_002.npy') # input image at t

# Calculating optical flow field
dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
flow_1 = dtvl1.calc(img1, img0, None)

# Saving optical flow field
np.save('o_vector_202007301000+10min.npy',flow_1)

# Generating future frame using optical flow field
h,w,_ = flow_1.shape
flow_1[:,:,0] += np.arange(w)
flow_1[:,:,1] += np.arange(h)[:,np.newaxis]
img_f = cv2.remap(img1, flow_1, None, cv2.INTER_CUBIC)
        
#saving future frame at t+10
np.save('o_precipitation_202007301000+10min.npy',img_f)
```

### Example: Linear regression stage
```python
# Generating future frames via various optical flow algorithms 
I_tvl1_0 = cv2.remap(img1, flow_tvl1_0, None, cv2.INTER_CUBIC)
I_deepflow_0 = cv2.remap(img1, flow_deepflow_0, None, cv2.INTER_CUBIC)
I_far_0 = cv2.remap(img1, flow_far_0, None, cv2.INTER_CUBIC)
I_pca_0 = cv2.remap(img1, flow_pca_0, None, cv2.INTER_CUBIC)

# Linear regression stage        
img_f = MLR_optflow(I_tvl1_0,I_deepflow_0,I_far_0,I_pca_0,img_gt)

# Saving future frame at t + 10    
np.save('o_precipitation_202007301000+10min.npy',img_f)
```
## Part II. U-Net architecture for training the nonlinear motion of precipitation fields

The codes are provided at the directory "U-Net".

Run the U-Net model through the command provided below.
```shell
export CUDA_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7
python main_Optflow.py
```
