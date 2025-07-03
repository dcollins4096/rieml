
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


import tube_loader
import rieML_model
reload(rieML_model)
#data = tube_loader.load_many()
if 'data' not in dir():
    data,parameters = tube_loader.read_good_parameters("tubes_take5.h5")
    data = torch.tensor(data,dtype=torch.float32)
    parameters = torch.tensor(parameters,dtype=torch.float32)
#data = data[1:]
#parameters = parameters[1:]

if 1:
    Ntrain=1100 #len(data) - 10
    train = data[:Ntrain]
    test = data[Ntrain:]
    test_parameters = parameters[Ntrain:]
    train_parameters = parameters[:Ntrain]
##model = pyt.Conv1DThreeChannel()
#model = pyt.NikhilsUnet()
#model = pyt.TwoU(base_filters=64)
testnum=177
new_model = 1
train_model = 1
import mixednn 

reload(mixednn)
if new_model:

    #hidden_dims=512,512
    #conv_channels=64
    hidden_dims = 1024   ,
    #hidden_dims = 128,256,512,256,128
    #hidden_dims = 256,512,1024,512,256
    #hidden_dims = 6000,
    conv_channels = 64
    model = mixednn.HybridShockTubeNN(hidden_dims=hidden_dims, conv_channels=conv_channels)

if 0:
    hidden_dims = 256,512,512,256
    #hidden_dims = 64,128,128,64
    conv_channels=6
    model = rieML_model.SixToThreeChannelNN(1000, hidden_dims=hidden_dims, conv_channels=conv_channels)
    #model = rieML_model.SixToThreeB(1000, hidden_dims = (256,512,1024,512,256))
if train_model:
    epoch = 50000
    batch_size=3
    lr = 1e-3
    losses=rieML_model.train(model,train,train_parameters,lr=lr, epochs = epoch, batch_size=batch_size, test_num=testnum, 
                     weight_decay=1e-4)
if 1:
    las = torch.argsort(losses)
    subset = list(range(5)) 
    characteristic=False
    delta = True
    zzz=rieML_model.ft_plot(test[subset], test_parameters[subset], model, fname="ft_%d"%testnum)
if 1:
    zzz=rieML_model.test_plot(test[subset], test_parameters[subset], model, fname="test_%d_test"%testnum,
                              characteristic=characteristic,delta=delta)
    zzz=rieML_model.test_plot(train[subset], train_parameters[subset], model, fname='test_%d_train'%testnum, 
                              characteristic=characteristic,delta=delta)
    subset = las[:5]
    zzz=rieML_model.test_plot(train[subset], train_parameters[subset], model, fname='test_%d_best'%testnum, 
                              characteristic=characteristic,delta=delta)
    subset = las[-5:]
    zzz=rieML_model.test_plot(train[subset], train_parameters[subset], model, fname='test_%d_worst'%testnum, 
                              characteristic=characteristic,delta=delta)
if 0:
    zzz=rieML_model.error_plot(train[subset], train_parameters[subset], model, fname='%d_train'%testnum)
    zzz=rieML_model.error_plot(test[subset], test_parameters[subset], model, fname='%d_test'%testnum)
if 1:
    bmin = min([min(losses),1e-4])
    bmax = max([max(losses),1e-1])
    bins = np.geomspace(bmin,bmax,64)
    fig,ax=plt.subplots(1,1)
    hist, bins, obj=ax.hist(losses.detach().numpy(), bins=bins)
    bc = 0.5*(bins[1:]+bins[:-1])
    Lmax = bc[np.argmax(hist)]
    ax.set(xlabel='loss',xscale='log', title = "mean %0.2e peak %0.2e std %0.2e"%(losses.mean(), Lmax, losses.std()))
    fig.savefig('%s/plots/errhist_test%d'%(os.environ['HOME'],testnum))
if 1:
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model test{testnum:d} with {nparam:,d} parameters")
