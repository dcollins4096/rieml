
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from importlib import reload
import pdb
import numpy as np
import matplotlib.pyplot as plt
plot_dir = "%s/plots"%os.environ['HOME']

def compute_losses_faster(model, data, parameters):
    guess = model(parameters)
    L1 = torch.abs( guess-data).mean(axis=-1).mean(axis=-1)
    return L1

def compute_losses_old(model,data,parameters):

    size=parameters[0].shape[0]
    guesses = [model(param.view(1,size)) for param in parameters]
    losses = torch.tensor([model.criterion(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(guesses,data)])
    return losses
def compute_losses_good(model,data,parameters):
    

    size=parameters[0].shape[0]
    guesses = [model(param.view(1,size)) for param in parameters]
    if len(data.shape) == 4:
        losses = torch.tensor([model.criterion(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(guesses,data)])
    elif len(data.shape)==3:
        losses = torch.tensor([model.criterion(mod.view(1,3,1000), dat.view(1,3,1000)) for mod,dat in zip(guesses,data)])
    else:
        pring('oops, something broke.')
        pdb.set_trace()

    return losses
def compute_losses(model,data,parameters):
    
    size=parameters[0].shape[0]
    guesses = [model(param.view(1,size)) for param in parameters]
    if len(data.shape) == 4:
        losses = torch.tensor([model.criterion(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(guesses,data)])
    elif len(data.shape)==3:
        losses = torch.tensor([model.criterion(mod.view(1,3,1000), dat.view(1,3,1000)) for mod,dat in zip(guesses,data)])
    else:
        pring('oops, something broke.')
        pdb.set_trace()

    return losses

nframes=11
def compute_losses_next(model, data, parameters, tubelist=None):
    ntube = len(data)//nframes  
    if tubelist is None:
        tubelist = torch.arange(ntube)
    framelist = torch.arange(nframes-1)
    loss_by_tube=torch.zeros(len(tubelist))
    for nt in tubelist:
        loss=0
        for nf in framelist:
            i = nt*nframes+nf
            datum_n = data[i]
            datum_np1 = data[i+1]
            p_n = parameters[i]
            p_np1= parameters[i+1]
            out = model(datum_n, p_n, p_np1)
            loss+=  model.criterion(out,datum_np1)
        loss_by_tube[nt] = loss/nframes
    return loss_by_tube


def plot_by_tube(data, parameters,model,fname="tube", tubelist=None):
    ntube = len(data)//nframes  
    if tubelist is None:
        tubelist = torch.arange(ntube)
    framelist = torch.arange(nframes-1)
    for nt in tubelist:
        for nframe in framelist:
            i = nt*nframes+nframe
            rows=2
            fields = ['density','pressure','velocity']
            fig,axes=plt.subplots(rows,3,figsize=(12,4))
            ax0,ax1=axes
            ymax = [2,2,1.1]
            datum_n = data[i]
            datum_np1 = data[i+1]
            z = model(datum_n, parameters[i], parameters[i+1])[0]

            for nf,field in enumerate(fields):
                ax0[nf].plot( datum_n[nf], c='k')
                ax1[nf].plot( datum_np1[nf], c='k')
                ymax[nf]=max([ymax[nf],datum_n[nf].max().item()])
                ax0[nf].set(ylabel=field)
                ymax[nf]=max([ymax[nf],datum_np1[nf].max().item()])
                if 1:
                    zzz = z[nf].detach().numpy()
                    if np.isnan(zzz).sum() > 0:
                        print("Is nan", np.isnan(zzz).sum(), nd, nf)
                    ax1[nf].plot( zzz, c='r')
                    ymax[nf]=max([ymax[nf],z.max().item()])
                    ax1[nf].set(ylabel=field)
            ax0[0].set(ylim=[0,ymax[0]])
            ax0[1].set(ylim=[0,ymax[1]])
            ax0[2].set(ylim=[-1.1,ymax[2]])
            ax1[0].set(ylim=[0,ymax[0]])
            ax1[1].set(ylim=[0,ymax[1]])
            ax1[2].set(ylim=[-1.1,ymax[2]])
            oname="%s/%s_t%04d_f%04d"%(plot_dir,fname,nt,nframe)
            print(oname)
            fig.savefig(oname)






def plot_hist(loss_train,loss_test,loss_validate,testnum):
    everything = torch.cat([loss_train, loss_test,loss_validate])
    bmin = min([min(everything),1e-4])
    bmax = max([max(everything),1e-1])
    bins = np.geomspace(bmin,bmax,64)
    fig,ax=plt.subplots(1,1)
    for nl in [0,1,2]:
        lll = [loss_train, loss_test, loss_validate][nl]
        hist, bins, obj=ax.hist(lll.detach().numpy(), bins=bins, histtype='step', 
                                label= "mean %0.2e"%(lll.mean()))
        bc = 0.5*(bins[1:]+bins[:-1])
        #Lmax = bc[np.argmax(hist)]
    ax.legend(loc=0)
    ax.set(xlabel='loss',xscale='log')
    fig.savefig('%s/plots/errhist_test%d'%(os.environ['HOME'],testnum))


def test_plot(datalist, parameters,model, fname="plot", characteristic=False, delta=False):
    nd=-1
    for datum, param1 in zip(datalist,parameters):
        size = param1.shape[0]
        param = param1.view(1,size)
        nd+=1
        z = model(param)
        #loss = model.criterion(z, datum[1], initial=datum[0])
        loss=-1
        z=z.view(3,1000)
        rows=1
        if characteristic:
            rows=2
        if delta:
            rows += 1


        fig,axes=plt.subplots(rows,3,figsize=(12,4))
        if characteristic or delta:
            ax,axb=axes
        else:
            ax=axes

        fields = ['density','pressure','velocity']
        ymax = [2,2,1.1]
        for nf,field in enumerate(fields):
            if len(datum.shape) == 3:
                ax[nf].plot( datum[0][nf], c='k')
                ymax[nf]=max([ymax[nf],datum[0][nf].max().item()])
                ax[nf].plot( datum[1][nf], c='k', linestyle='--')
                ymax[nf]=max([ymax[nf],datum[1][nf].max().item()])
            elif len(datum.shape) == 2:
                ax[nf].plot( datum[nf], c='k')
                ymax[nf]=max([ymax[nf],datum[nf].max().item()])
            zzz = z[nf].detach().numpy()
            if np.isnan(zzz).sum() > 0:
                print("Is nan", np.isnan(zzz).sum(), nd, nf)
            ax[nf].set(title='error %0.2e'%loss)
            ax[nf].plot( zzz, c='r')
            ymax[nf]=max([ymax[nf],z.max().item()])
            ax[nf].set(ylabel=field)
            if delta:
                ddd = datum[1][nf] - zzz
                axb[nf].plot(ddd**2)
                errp=(ddd**2).mean()
                axb[nf].set(title="%0.2e"%errp)
        ax[0].set(ylim=[0,ymax[0]])
        ax[1].set(ylim=[0,ymax[1]])
        ax[2].set(ylim=[-1.1,ymax[2]])
        if characteristic:
            ICchar = ptoc.primitive_to_characteristic(datum[0])
            REchar = ptoc.primitive_to_characteristic(datum[1])
            MOchar = ptoc.primitive_to_characteristic(z)
            #pdb.set_trace()

            fields = ['w1','w2','w3']
            for nf,field in enumerate(fields):
                axb[nf].plot( ICchar[nf], c='k')
                #pdb.set_trace()
                axb[nf].plot( REchar[nf], c='k', linestyle='--')
                #print("Is nan", np.isnan(zzz).sum(), nd, nf)
                zzz = MOchar[nf].detach().numpy()
                axb[nf].set(title='error %0.2e'%loss)
                axb[nf].plot( zzz, c='r')
                axb[nf].set(ylabel=field)
            #ax[0].set(ylim=[0,2])
            #ax[1].set(ylim=[0,2])
            #ax[2].set(ylim=[-1.1,1.1])
        fig.tight_layout()
        oname="%s/rieML_%s_%04d"%(plot_dir,fname,nd)
        fig.savefig(oname)
        print(oname)
        plt.close(fig)
    return zzz
