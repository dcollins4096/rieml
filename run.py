
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
import matplotlib.pyplot as plt
import time
import plot
reload(plot)


import tube_loader
import networks.net300 as net
reload(net)

new_model = 1
load_model = 0
train_model = 1

testnum=net.idd
#data = tube_loader.load_many()
if 'data' not in dir():
    alldata,allparameters = tube_loader.read_good_parameters("tubes_take5.h5")
    alldata = torch.tensor(alldata,dtype=torch.float32)
    allparameters = torch.tensor(allparameters,dtype=torch.float32)
if 1:
    Ntrain=1000 #len(data) - 10
    Nvalid = 1100 #leaving 49 for validation
    data={}
    parameters={}
    data['train'] = alldata[:Ntrain]
    data['test'] = alldata[Ntrain:Nvalid]
    data['validate'] = alldata[Nvalid:]
    parameters['train'] = allparameters[:Ntrain]
    parameters['test'] = allparameters[Ntrain:Nvalid]
    parameters['validate'] = allparameters[Nvalid:]

if new_model:

    model = net.thisnet()

if load_model:

    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

total_time='no'
if train_model:

    t0 = time.time()

    net.train(model,data['train'],parameters['train'], data['validate'],parameters['validate'])

    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)

if 1:
    print('losses')
    loss_train = plot.compute_losses(model, data['train'],parameters['train'])
    loss_test = plot.compute_losses(model, data['test'],parameters['test'])
    loss_validate = plot.compute_losses(model, data['validate'],parameters['validate'])
    args_train = torch.argsort(loss_train)
    args_test = torch.argsort(loss_test)
    args_validate = torch.argsort(loss_validate)
if 1:
    print('plot')
    plot.plot_hist(loss_train,loss_test,loss_validate,net.idd)

    zzz=plot.test_plot(data['test'][args_test[:5]], parameters['test'][args_test[:5]], model, fname="test%d_test_best"%testnum)
    zzz=plot.test_plot(data['test'][args_test[-5:]], parameters['test'][args_test[-5:]], model, fname="test%d_test_worst"%testnum)
    zzz=plot.test_plot(data['train'][args_train[:5]], parameters['train'][args_train[:5]], model, fname="test%d_train_best"%testnum)
    zzz=plot.test_plot(data['train'][args_train[-5:]], parameters['train'][args_train[-5:]], model, fname="test%d_train_worst"%testnum)
    zzz=plot.test_plot(data['validate'], parameters['validate'], model, fname="test%d_avalidate"%testnum)

if not load_model:
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model test{testnum:d} with {nparam:,d} parameters elapsed {total_time:s}")
    oname = "models/test%d.pth"%testnum
    torch.save(model.state_dict(), oname)
    print("model saved ",oname)
