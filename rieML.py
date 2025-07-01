
import pytorch_network as pyt
from importlib import reload
import sys
import os
sys.stderr = open(os.devnull, 'w')
import torch
#torch.backends.nnpack.enabled = False
reload(pyt)
import pdb
import numpy as np


import tube_loader
import rieML_model
reload(rieML_model)
#data = tube_loader.load_many()
data,parameters = tube_loader.read_one_parameters("tubes_take5.h5")
data = torch.tensor(data,dtype=torch.float32)
parameters = torch.tensor(parameters,dtype=torch.float32)
#data = data[1:]
#parameters = parameters[1:]

if 1:
    Ntrain=1200 #len(data) - 10
    train = data[:Ntrain]
    test = data[Ntrain:]
    test_parameters = parameters[Ntrain:]
    train_parameters = parameters[:Ntrain]
##model = pyt.Conv1DThreeChannel()
#model = pyt.NikhilsUnet()
#model = pyt.TwoU(base_filters=64)
testnum=95
new_model = 0
train_model = 0
import mixednn 

reload(mixednn)
if new_model:

    #hidden_dims=512,512
    #conv_channels=64
    hidden_dims = 128, 128
    conv_channels = 32
    model = mixednn.HybridShockTubeNN(hidden_dims=hidden_dims, conv_channels=conv_channels)

if 0:
    hidden_dims = 256,512,512,256
    #hidden_dims = 64,128,128,64
    conv_channels=6
    model = rieML_model.SixToThreeChannelNN(1000, hidden_dims=hidden_dims, conv_channels=conv_channels)
    #model = rieML_model.SixToThreeB(1000, hidden_dims = (256,512,1024,512,256))
if train_model:
    epoch = 1000
    batch_size=100
    lr = 1e-3
    rieML_model.train(model,train,train_parameters,lr=lr, epochs = epoch, batch_size=batch_size, test_num=testnum, 
                     weight_decay=1e-4)
if 1:
    subset = slice(0,5)
    characteristic=False
    delta = True
    #zzz=rieML_model.test_plot(train[subset], train_parameters[subset], model, fname='test_%d_train'%testnum, 
    #                          characteristic=characteristic,delta=delta)
    zzz=rieML_model.error_plot(train[subset], train_parameters[subset], model, fname='%d_train'%testnum)
if 1:
    pass
    zzz=rieML_model.test_plot(test[subset], test_parameters[subset], model, fname="test_%d_test"%testnum,
                              characteristic=characteristic,delta=delta)
    zzz=rieML_model.error_plot(test[subset], test_parameters[subset], model, fname='%d_test'%testnum)
