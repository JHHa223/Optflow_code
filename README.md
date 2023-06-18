# Optflow v1.0: A Deep Learning Model for Precipitation Nowcasting Using Multiple Optical Flow Algorithms

## Introduction
Optflow v1.0 is a deep learning model that utilizes the optical flow method to enhance the performance of precipitation nowcasting.

## Model Architecture 
A Deep Learning Model for Precipitation Nowcasting consists of two parts.

Part I. Optical flow calculation and linear regression stage: /optical_flow

Part II. U-Net architecture for training the nonlinear motion of precipitation fields: /U-NET

## Part I. Optical flow calculation and linear regression stage

OpenCV library is used here to estimate the optical flow field.

```python
#### Single-Temporal Model ####
dtvl1=cv2.optflow.DualTVL1OpticalFlow_create()
    for j in range(0,18):
        # Reading input data
        img0 = np.load(path[j]) # input image at t-10
        img1 = np.load(path[j+1]) # input image at t

        flow_1 = dtvl1.calc(img1, img0, None)

        # saving optical flow field
        np.save(new_path+'o_vector_202007301000+'+str((j+1)*10)+'min.npy',flow_1)

        # Generating future frame using optical flow field
        h,w,_ = flow_1.shape
        flow_1[:,:,0] += np.arange(w)
        flow_1[:,:,1] += np.arange(h)[:,np.newaxis]
        Img_f = cv2.remap(img1, flow_1, None, cv2.INTER_CUBIC)
        
        #saving future frame at t+10
        np.save(new_path+'o_precipitation_202007301000+'+str(10*(j+1))+'min.npy',Img_f)
```

## Part II. U-Net architecture for training the nonlinear motion of precipitation fields
