
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
    Ntrain=2 #len(data) - 10
    testnum=34
    train = data[:Ntrain]
    test = data[Ntrain:]
    test_parameters = parameters[Ntrain:]
    train_parameters = parameters[:Ntrain]
##model = pyt.Conv1DThreeChannel()
#model = pyt.NikhilsUnet()
#model = pyt.TwoU(base_filters=64)
if 0:
    #hidden_dims = 256,512,512,256
    #hidden_dims = 64,128,128,64
    #model = rieML_model.SixToThreeChannelNN(1000, hidden_dims=hidden_dims)
    model = rieML_model.SixToThreeB(1000, hidden_dims = (256,512,512,256))
if 0:
    epoch = 300
    batch_size=20
    lr = 1e-3
    rieML_model.train(model,train,train_parameters,lr=lr, epochs = epoch, batch_size=batch_size, test_num=testnum, 
                     weight_decay=1e-4)
if 1:
    subset = slice(0,10)
    zzz=rieML_model.test_plot(train[subset], train_parameters[subset], model, fname='test_%d_train'%testnum)
    zzz=rieML_model.test_plot(test[subset], test_parameters[subset], model, fname="test_%d_test"%testnum)
