
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
import networks.net405 as net
reload(net)

new_model = 1
load_model = 0
train_model = 1
make_plot = 1
save_model = 0
pdb.set_trace()
print('you dont want this one')

testnum=net.idd
#data = tube_loader.load_many()
if 'data' not in dir():
    read=False
    if hasattr(net,'time_data'):
        if net.time_data:
            print('read time')
            data, parameters,numbers= tube_loader.read_good_parameters("tubes_take8.h5", nvalid=200, ntest=500)
            #data, parameters= tube_loader.read_good_parameters("tubes_take8.h5", nvalid=50, ntest=100)
            #for 404
            #data['train'] = data['train'][:3000]
            #parameters['train'] = parameters['train'][:3000]
            read=True
    if not read:
        print('read no time')
        data, parameters= tube_loader.read_good_parameters("tubes_take6.h5", nvalid=50, ntest=100)
    #data, parameters= tube_loader.read_good_parameters("tubes_take8.h5", nvalid=200, ntest=500)

if new_model:

    model = net.thisnet()

if load_model:

    model.load_state_dict(torch.load("models/test%d.pth"%net.idd))

total_time='no'
if train_model:

    t0 = time.time()

    net.train(model,data['train'],parameters['train'], data['validate'],parameters['validate'])

    if save_model:
        oname = "models/test%d.pth"%testnum
        torch.save(model.state_dict(), oname)
        print("model saved ",oname)

    t1 = time.time() - t0
    hrs = t1//3600
    minute = (t1-hrs*3600)//60
    sec = (t1 - hrs*3600-minute*60)#//60
    total_time="%02d:%02d:%02d"%(hrs,minute,sec)

if make_plot:
    print('losses', len(data['train']), len(data['test']), len(data['validate']))
    loss_train = plot.compute_losses(model, data['train'][::10],parameters['train'][::10])
    print('done with the long one')
    loss_test = plot.compute_losses(model, data['test'],parameters['test'])
    loss_validate = plot.compute_losses(model, data['validate'],parameters['validate'])
    args_train = torch.argsort(loss_train)
    args_test = torch.argsort(loss_test)
    #args_valiiate = torch.argsort(loss_validate)

    plot.plot_hist(loss_train,loss_test,loss_validate,net.idd)

    zzz=plot.test_plot(data['test'][args_test[:5]], parameters['test'][args_test[:5]], model, fname="test%d_test_best"%testnum)
    zzz=plot.test_plot(data['test'][args_test[-5:]], parameters['test'][args_test[-5:]], model, fname="test%d_test_worst"%testnum)
    zzz=plot.test_plot(data['train'][args_train[:5]], parameters['train'][args_train[:5]], model, fname="test%d_train_best"%testnum)
    zzz=plot.test_plot(data['train'][args_train[-5:]], parameters['train'][args_train[-5:]], model, fname="test%d_train_worst"%testnum)
    zzz=plot.test_plot(data['validate'], parameters['validate'], model, fname="test%04d_avalidate"%testnum)

if 1:
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model test{testnum:d} with {nparam:,d} parameters elapsed {total_time:s}")
    #oname = "models/test%d.pth"%testnum
    #torch.save(model.state_dict(), oname)
    #print("model saved ",oname)
