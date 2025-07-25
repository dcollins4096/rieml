
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

new_model = 1
load_model = 0
train_model = 1
save_model = 1
make_plot = 0
nframes=11
#data = tube_loader.load_many()
if new_model:
    import networks.net502 as net
    reload(net)
    model = net.thisnet()

testnum=net.idd

if 'data' not in dir():
    read=False
    if hasattr(net,'time_data'):
        if net.time_data:
            print('read time')
            data, parameters= tube_loader.read_good_parameters_by_tube("tubes_take8.h5", nvalid=5*nframes, ntest=20*nframes)
            #data, parameters= tube_loader.read_good_parameters("tubes_take8.h5", nvalid=50, ntest=100)
            #for 404
            #data['train'] = data['train'][:3000]
            #parameters['train'] = parameters['train'][:3000]
            read=True
    if not read:
        print('read no time')
        data, parameters= tube_loader.read_good_parameters("tubes_take6.h5", nvalid=50, ntest=100)
    #data, parameters= tube_loader.read_good_parameters("tubes_take8.h5", nvalid=200, ntest=500)


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
    if 'all_loss' not in dir():
        all_loss = plot.loss_by_tube(model,data,parameters)
    plot.plot_hist2(all_loss, model.idd)

    #zzz=plot.plot_by_tube(data['validate'], parameters['validate'], model, fname="test%d_validate"%testnum)
    #zzz=plot.plot_by_tube(data['test'], parameters['test'], model, fname="test%d_test"%testnum)
    worst_worst = torch.argsort(all_loss['train']['max'])[-5:]
    worst_mean =  torch.argsort(all_loss['train']['mean'])[-5:]
    #zzz=plot.test_plot(data['validate'], parameters['validate'], model, fname="test%04d_avalidate"%testnum)
    #zzz=plot.test_plot(data['test'], parameters['test'], model, fname="test%d_test_worst"%testnum)
    zzz=plot.plot_by_tube(data['train'], parameters['train'], model, fname="test%d_train_worst_worst"%testnum, 
                      tubelist=worst_worst)
    zzz=plot.plot_by_tube(data['train'], parameters['train'], model, fname="test%d_train_worst_mean"%testnum,
                      tubelist=worst_mean)
    #zzz=plot.test_plot(data['train'], parameters['train'], model, fname="test%d_train_worst"%testnum)

if 1:
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model test{testnum:d} with {nparam:,d} parameters elapsed {total_time:s}")
    #oname = "models/test%d.pth"%testnum
    #torch.save(model.state_dict(), oname)
    #print("model saved ",oname)
