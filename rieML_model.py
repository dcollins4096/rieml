# PyTorch MLP: 6 inputs → 1D output array with 3 channels
# ---------------------------------------------------------
# This module defines a neural network that maps a 6-dimensional input
# vector to an output tensor of shape (batch_size, 3, L), where L is the
# 1D length of each channel.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import pdb
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import ptoc
import time
from importlib import reload
reload(ptoc)
plot_dir = "%s/plots"%os.environ['HOME']

def train(model, data,parameters, epochs=1, lr=1e-3, batch_size=10, test_num=0, weight_decay=None):
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    fptr = open('output','w')
    fptr.close()
    #optimizer = optim.AdamW( model.parameters(), lr=lr, weight_decay = 1e-2)
    optimizer = optim.Adam( model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25000, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
            milestones=[25000, 35000,45000],  # change after N and N+M steps
                gamma=0.1             # multiply by gamma each time
                )
    n=-1
    losses=[]
    a = torch.arange(len(data))
    N = len(data)
    seed = 8675309
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    t0 = time.time()
    loss_batch=[]
    minlist=[];meanlist=[];maxlist=[];stdlist=[]
    for epoch in range(epochs):
        subset = torch.tensor(random.sample(list(a),batch_size))
        data_subset =  data[subset]
        param_subset = parameters[subset]
        optimizer.zero_grad()
        output1=model(param_subset)
        loss = model.criterion1(output1, data_subset[:,1,:,:], initial=data_subset[:,0,:,:])
        loss.backward()
        optimizer.step()
        scheduler.step()
        tnow = time.time()
        tel = tnow-t0
        loss_batch.append(loss.item())
        if epoch>0 and epoch%10==0:
            time_per_epoch = tel/epoch
            epoch_remaining = epochs-epoch
            time_remaining_s = time_per_epoch*epoch_remaining
            hrs = time_remaining_s//3600
            minute = (time_remaining_s-hrs*3600)//60
            sec = (time_remaining_s - hrs*3600-minute*60)#//60
            time_remaining="%02d:%02d:%02d"%(hrs,minute,sec)

            mean = np.mean(loss_batch)
            std = np.std(loss_batch)
            mmin = min(loss_batch)
            mmax = max(loss_batch)
            minlist.append(mmin)
            maxlist.append(mmax)
            meanlist.append(mean)
            stdlist.append(std)
            print(time_remaining)
            print("Epoch %d loss %0.2e LR %0.2e time left %8s loss mean %0.2e var %0.2e min %0.2e max %0.2e"%
                  (epoch,loss, optimizer.param_groups[0]['lr'], time_remaining, mean, std, mmin, mmax))
            loss_batch=[]
    print("Run time", tel)
    plt.clf()
    plt.plot(meanlist,c='k')
    plt.plot(np.array(meanlist)+np.array(stdlist),c='b')
    plt.plot(np.array(meanlist)-np.array(stdlist),c='b')
    plt.plot(minlist,c='k')
    plt.plot(maxlist,c='k')
    plt.yscale('log')
    plt.savefig('%s/errortime_test%d'%(plot_dir,test_num))

    models = [model(param.view(1,6)) for param in parameters]
    losses = torch.tensor([model.criterion1(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(models,data)])
    aas = torch.argsort(losses)[:50]
    if 0:
        for cur in range(100):
            print('more',cur)
            models = [model(param.view(1,6)) for param in parameters]
            losses = torch.tensor([model.criterion1(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(models,data)])
            aas = torch.argsort(losses)[:300]
            for repeat in range(10):
                for ind in range(len(aas)-batch_size):
                    subset = torch.tensor(aas[ind:ind+batch_size])
                    data_subset =  data[subset]
                    param_subset = parameters[subset]
                    optimizer.zero_grad()
                    output1=model(param_subset)
                    loss = model.criterion1(output1, data_subset[:,1,:,:], initial=data_subset[:,0,:,:])
                    loss.backward()
                    optimizer.step()
    if 0:
        for cur in range(50):
            print('more',cur)
            models = [model(param.view(1,6)) for param in parameters]
            losses = torch.tensor([model.criterion1(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(models,data)])
            aas = torch.argsort(losses)[-300:]
            for repeat in range(10):
                for ind in range(len(aas)-batch_size):
                    subset = torch.tensor(aas[ind:ind+batch_size])
                    data_subset =  data[subset]
                    param_subset = parameters[subset]
                    optimizer.zero_grad()
                    output1=model(param_subset)
                    loss = model.criterion1(output1, data_subset[:,1,:,:], initial=data_subset[:,0,:,:])
                    loss.backward()
                    optimizer.step()

    if 0:
        for epoch in range(epochs):
            subset = torch.tensor(random.sample(list(a),batch_size))
            data_subset =  data[subset]
            param_subset = parameters[subset]
            optimizer.zero_grad()
            output1=model(param_subset)
            loss = model.criterion2(output1, data_subset[:,1,:,:], initial=data_subset[:,0,:,:])
            loss.backward()
            optimizer.step()
            print("Epoch2 %d loss %0.2e LR %0.2e"%(epoch,loss, optimizer.param_groups[0]['lr']))

    plt.clf()
    plt.plot(losses[aas])
    plt.yscale('log')
    plt.savefig("%s/loss_test_%d_%d"%(plot_dir,test_num,n))
    return losses


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(128, 256, 128)):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SixToThreeB(nn.Module):
    def __init__(self, output_length: int, hidden_dims=(128, 256, 128)):
        super().__init__()
        self.output_length = output_length

        # Shared encoder for input → latent vector
        self.encoder = MLP(6, hidden_dims[-1], hidden_dims[:-1])

        # Separate decoder heads for each output channel
        self.heads = nn.ModuleList([
            MLP(hidden_dims[-1], output_length, hidden_dims[::-1])  # decoder per channel
            for _ in range(3)
        ])
        self.mse = nn.MSELoss()
    def criterion(self,target,guess):
        mse=self.mse(target,guess)
        N = target.shape[1]
        high_k = high_frequench_penalty(guess)
        #mmax = torch.abs(torch.max(target-guess))
        #print("Mse %0.2e max %0.2e"%(mse,mmax))
        #print("Mse %0.2e l2  %0.2e"%(mse,l2))
        return mse+high_k

    def forward(self, x):
        """
        x: shape (batch_size, 6)
        returns: (batch_size, 3, output_length)
        """
        batch_size = x.size(0)
        latent = self.encoder(x)  # (batch_size, latent_dim)

        # Pass through each head
        outs = [head(latent).unsqueeze(1) for head in self.heads]  # list of (batch,1,L)
        y = torch.cat(outs, dim=1)  # (batch_size, 3, output_length)
        y=y.T
        return y

def high_frequency_penalty(output, cutoff_ratio=0.3):
    """
    Penalize high-frequency components in the output.
    
    Args:
        output: (batch_size, channels, length)
        cutoff_ratio: fraction of frequencies to treat as low-frequency (0.3 means first 30%)
        
    Returns:
        scalar penalty term
    """

    # FFT along spatial dimension
    fft = torch.fft.rfft(output, dim=1)
    
    # Power spectrum
    power = torch.abs(fft) ** 2
    
    # Determine cutoff index
    n_freqs = fft.shape[-1]
    cutoff = int(n_freqs * cutoff_ratio)

    # High-frequency power only
    high_freq_power = power[..., cutoff:]
    
    # Return mean power in high frequencies as penalty
    return high_freq_power.mean()



def smoothness_loss(y):
    diff = y[:, 1:] - y[:, :-1]
    return torch.mean(diff ** 2)
def fourth(x,y):
    return torch.mean( (x-y)**4)


class SixToThreeChannelNN(nn.Module):
    def __init__(self, output_length: int, conv_channels=3,hidden_dims=(64, 128, 128,64)):
        """
        Args:
            output_length: Length L of the 1D output array per channel.
            hidden_dims: Tuple specifying hidden layer sizes.
        """
        super(SixToThreeChannelNN, self).__init__()
        self.output_length = output_length

        # Build fully-connected layers
        dims = [6] + list(hidden_dims) + [3 * output_length]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            # Apply activation after every layer except the last
            if out_dim != dims[-1]:
                layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(
                nn.Conv1d(3, conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(conv_channels, 3, kernel_size=3, stride=1,padding=1)
            )
        self.net = nn.Sequential(*layers)
        self.mse = nn.MSELoss()

    def criterion(self,target,guess):
        mse = self.mse(target,guess)
        high_k = high_frequency_penalty(guess)
        #smooth = smoothness_loss(guess)
        #fourth_loss = fourth(target,guess)
        #output = self.mse(target,guess) + smoothness_loss(guess)
        #print("MSE %0.2e smooth %0.2e"%(mse,smooth))
        #print("Mse %0.2e k  %0.2e"%(mse,high_k))
        return mse+high_k
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 6)
        Returns:
            Tensor of shape (batch_size, 3, output_length)
        """
        # Forward pass through MLP → (batch_size, 3*L)
        x = self.net(x)
        x = x.view(1, 3, self.output_length) 
        x = x+ self.conv(x)

        x = x.view( 3, self.output_length)
        return x
def ft_plot(datalist, parameters,model, fname="plot"):
    nd=-1
    for datum, param1 in zip(datalist,parameters):
        param = param1.view(1,6)
        nd+=1
        #pdb.set_trace()
        z = model(param)[0]
        rows=1
        fig,axes=plt.subplots(rows,3,figsize=(12,4))
        axa=axes#[0]
        #axb=axes[1]
        #axc=axes[2]

        fields = ['density','pressure','velocity']

        for nf,field in enumerate(fields):
            q = z[nf].detach().numpy()
            ft_z = np.fft.rfft(q)
            ft_d = np.fft.rfft(datum[1][nf])
            p_z = np.abs(ft_z)**2
            p_d = np.abs(ft_d)**2
            axa[nf].plot(p_d,c='k')
            axa[nf].plot(p_z,c='r')
            axa[nf].set(xscale='log',yscale='log')
            
        fig.tight_layout()
        fig.savefig("%s/ft_%s_%d"%(plot_dir,fname,nd))
        plt.close(fig)
def error_plot(datalist, parameters,model, fname="plot"):
    nd=-1
    for datum, param in zip(datalist,parameters):
        nd+=1
        #pdb.set_trace()
        z = model(param)
        loss = model.criterion(z, datum[1], initial=datum[0])
        print(loss)
        rows=3
        fig,axes=plt.subplots(rows,3,figsize=(12,4))
        axa=axes[0]
        axb=axes[1]
        axc=axes[2]

        fields = ['density','pressure','velocity']
        dat = datum[1]
        dx_target,dx_guess = model.sobolev_derivatives(dat, z) #target guess
        dsob = (dx_target-dx_guess)**2

        for nf,field in enumerate(fields):
            mse = (z[nf]- dat[nf])**2
            mse = mse.detach().numpy()
            mse_weight, sobolev_weight = model.convex_combination()
            sob = model.sobolev(dat,z)
            
            axa[nf].plot( dx_guess[nf].detach().numpy(), c='g', label='dx guess %0.2e'%sob)
            axa[nf].plot( dx_target[nf].detach().numpy()[10:], c='r', label='dx target %0.2e'%sobolev_weight)
            axb[nf].plot( mse, c='k', label='mse %0.2e'%mse.mean())
            axc[nf].plot(dsob[nf].detach().numpy(),label='sob %0.2e'%sob.mean())
            #pdb.set_trace()
            axa[nf].legend(loc=0)
            axb[nf].legend(loc=0)
            axc[nf].legend(loc=0)
        fig.tight_layout()
        fig.savefig("%s/error_%s_%d"%(plot_dir,fname,nd))
        plt.close(fig)

def test_plot(datalist, parameters,model, fname="plot", characteristic=False, delta=False):
    nd=-1
    for datum, param1 in zip(datalist,parameters):
        param = param1.view(1,6)
        nd+=1
        z = model(param)
        z=z.view(3,1000)
        loss = model.criterion1(z, datum[1], initial=datum[0])
        print(loss)
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
            ax[nf].plot( datum[0][nf], c='k')
            ymax[nf]=max([ymax[nf],datum[0][nf].max().item()])
            ax[nf].plot( datum[1][nf], c='k', linestyle='--')
            ymax[nf]=max([ymax[nf],datum[1][nf].max().item()])
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
        fig.savefig("%s/rieML_%s_%d"%(plot_dir,fname,nd))
        plt.close(fig)
    return zzz
