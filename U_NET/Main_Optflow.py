import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import torch
from torch import optim
from Data_Loader import Loader
from UNET_Optflow import *


#### TRAINING #####
def train(epoch):
    model.train()
    loss_tr = 0
    for batch_idx, inputs in enumerate(train_loader):
        inputs, targets = inputs[:,:3,:,:].float().to(device), inputs[:,3:,:,:].float().to(device)        
        optimizer.zero_grad()
        output = model(inputs)
        loss = l1_loss(output, targets)
        loss_tr += loss
        loss.backward()
        optimizer.step()
    loss_train.append(loss_tr/len(train_loader))
    scheduler.step()
    state = model.state_dict()
    torch.save(state, './ckpt_lr4b8_srf_aug_st20_g0.5.pth')
    print("training epoch: {} | loss: {:.6f}".format(epoch, loss_tr/len(train_loader)))
    
#### VALIDATION ####
def validation(epoch):
    model.eval()
    loss_v = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(val_loader):
            inputs, targets = inputs[:,:3,:,:].float().to(device), inputs[:,3:,:,:].float().to(device)
            output = model(inputs)
            loss = l1_loss(output, targets)
            loss_v += loss
            total += targets.size(0)
    loss_val.append(loss_v/len(test_loader))
    print("validation epoch: {} | loss: {:.6f}".format(epoch, loss_v/len(test_loader)))  

device = 'cuda'

#### Importing U-NET MODEL ####
model = UNetOptflow()
model = model.to(device)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

#### OPTIMIZER & SCHEDULER ####
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#### LOSS FUNCTION ####
l1_loss = nn.L1Loss()

#### DATA LOADER ####
train_loader = torch.utils.data.DataLoader(
    Loader(dummy=0),
    batch_size=8, shuffle=True,
    num_workers=32, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    Loader(dummy=1),
    batch_size=8, shuffle=True,
    num_workers=32, pin_memory=True)  

#### TRAINING & VALIDATIONO LOSSES ####
loss_train = []
loss_val = []

#### MAIN LOOP ####
for epoch in range(120):
    train(epoch)
    validation(epoch)

# FOR SAVING TRAINING & VALIDATIONO LOSSES
with open("learning/loss_train.pkl","wb") as f:
    pickle.dump(loss_train,f)
with open("learning/loss_validation.pkl","wb") as g:    
    pickle.dump(loss_val,g)
    
