import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from importlib import reload
import pdb
import random
import matplotlib.pyplot as plt
import numpy as np
import plot
plot_dir = "%s/plots"%os.environ['HOME']

idd = 293
what = "290 with more parameters"

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    hidden_dims = 1024,1024
    conv_channels = 128
    model = main_net(hidden_dims=hidden_dims, conv_channels=conv_channels)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs = 50000
    lr = 1e-4
    batch_size=3
    trainer(model,data,parameters,validatedata,validateparams,epochs=epochs,lr=lr,batch_size=batch_size)

def trainer(model, data,parameters, validatedata,validateparams,epochs=1, lr=1e-3, batch_size=10, test_num=0, weight_decay=None):
    optimizer = optim.Adam( model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25000, 35000,45000],  # change after N and N+M steps
        gamma=0.1             # multiply by gamma each time
    )
    losses=[]
    a = torch.arange(len(data))
    N = len(data)
    seed = 8675309
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    t0 = time.time()
    minlist=[];meanlist=[];maxlist=[];stdlist=[]
    for epoch in range(epochs):
        subset = torch.tensor(random.sample(list(a),batch_size))
        data_subset =  data[subset]
        param_subset = parameters[subset]
        optimizer.zero_grad()
        output1=model(param_subset)
        loss = model.criterion(output1, data_subset[:,1,:,:], initial=data_subset[:,0,:,:])
        loss.backward()
        optimizer.step()
        scheduler.step()
        tnow = time.time()
        tel = tnow-t0
        if (epoch>0 and epoch%100==0) or epoch==10:
            model.eval()
            validate_losses = plot.compute_losses(model, validatedata, validateparams)
            model.train()

            time_per_epoch = tel/epoch
            epoch_remaining = epochs-epoch
            time_remaining_s = time_per_epoch*epoch_remaining
            hrs = time_remaining_s//3600
            minute = (time_remaining_s-hrs*3600)//60
            sec = (time_remaining_s - hrs*3600-minute*60)#//60
            time_remaining="%02d:%02d:%02d"%(hrs,minute,sec)

            mean = validate_losses.mean()
            std = validate_losses.std()
            mmin = validate_losses.min()
            mmax = validate_losses.max()
            minlist.append(mmin)
            maxlist.append(mmax)
            meanlist.append(mean)
            stdlist.append(std)
            print("test%d Epoch %d loss %0.2e LR %0.2e time left %8s loss mean %0.2e var %0.2e min %0.2e max %0.2e"%
                  (idd,epoch,loss, optimizer.param_groups[0]['lr'], time_remaining, mean, std, mmin, mmax))
            loss_batch=[]
    print("Run time", tel)
    plt.clf()
    plt.plot(meanlist,c='k')
    plt.plot(np.array(meanlist)+np.array(stdlist),c='b')
    plt.plot(np.array(meanlist)-np.array(stdlist),c='b')
    plt.plot(minlist,c='r')
    plt.plot(maxlist,c='r')
    plt.yscale('log')
    plt.savefig('%s/errortime_test%d'%(plot_dir,idd))


class main_net(nn.Module):
    def __init__(self, output_length=1000, hidden_dims=(128, 128), conv_channels=32, characteristic=False):
        super().__init__()
        self.output_length = output_length

        # Project 6 input values to a pseudo-spatial format (3 channels)
        if 0:
            self.fc1 = nn.Linear(6, 3 * output_length)
            self.relu1 = nn.ReLU()

        # Conv block 1 (acts on the "3 x output_length" format)
        if 0:
            dil = 1
            kern = 5
            padding = dil*(kern-1)//2
            dil2 = 2
            padding2 = dil2*(kern-1)//2
            dil3 = 4
            padding3 = dil3*(kern-1)//2
            dil4 = 8
            padding4 = dil4*(kern-1)//2
            self.conv1 = nn.Sequential(
                nn.Conv1d(3, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
                nn.ReLU(),
                #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding2, dilation=dil2),
                #nn.ReLU(),
                #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding3, dilation=dil3),
                #nn.ReLU(),
                #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding4, dilation=dil4),
                #nn.ReLU(),
                nn.Conv1d(conv_channels, 3, kernel_size=kern, padding=padding, dilation=dil),
                nn.ReLU()
            )
            self.conv1.apply(init_weights_constant)

        if 1:
            in_dim = 3*output_length
            out_dim = 3*output_length
            layers=[]
            dims = [6] + [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.fc2 = nn.Sequential(*layers)
            self.fc2.apply(init_weights_constant)

        # Conv block 2
        if 0:
            dil = 1
            kern = 3
            padding = dil*(kern-1)//2
            self.conv2 = nn.Sequential(
                nn.Conv1d(3, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
                nn.ReLU(),
                nn.Conv1d(conv_channels, 3, kernel_size=kern, padding=padding, dilation=dil)
            )
            self.conv2.apply(init_weights_constant)
        if 1:
            dil1 = 2
            kern1 = 3
            padding1 = dil1*(kern1-1)//2
            dil2 = 2
            kern2 = 3
            padding2 = dil2*(kern2-1)//2
            dil3 = 5
            kern3 = 7
            padding3 = dil3*(kern3-1)//2
            if 1:
                self.conv3 = nn.Sequential(
                    nn.Conv1d(3, conv_channels, kernel_size=kern1, padding=padding1, dilation=dil1),
                    nn.ReLU(),
                    nn.Conv1d(conv_channels, 2*conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                    nn.ReLU(),
                    nn.Conv1d(2*conv_channels, 4*conv_channels, kernel_size=kern3, padding=padding3, dilation=dil3),
                    nn.ReLU(),
                    nn.Conv1d(4*conv_channels, 2*conv_channels, kernel_size=kern3, padding=padding3, dilation=dil3),
                    nn.ReLU(),
                    nn.Conv1d(2*conv_channels, conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                    nn.ReLU(),
                    nn.Conv1d(conv_channels, 3, kernel_size=kern1, padding=padding1, dilation=dil1)
                )
            self.conv3.apply(init_weights_constant)
        #if characteristic:
        #    self.forward=self.char_forward
        #else:
        #    self.forward=self.prim_forward
        self.T = nn.Parameter(torch.eye(3) + 0.01 * torch.randn(3, 3)) 

        self.mse=nn.MSELoss()
        self.log_derivative_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_tv_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_high_k_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.criterion = self.criterion1

        #self.attn = SelfAttention1D(channels=3, length=output_length)
        #self.attn_rho = SelfAttention1D(channels=1, length=output_length)
        #self.attn_vel = SelfAttention1D(channels=1, length=output_length)
        #self.attn_pres = SelfAttention1D(channels=1, length=output_length)
        embed_dim=output_length
        num_heads = 4
        self.hl = nn.HuberLoss(delta=0.2)
        self.l1 = nn.L1Loss()

        #self.attn_rho =  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)
        #self.attn_vel =  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)
        #self.attn_pres=  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)

    def criterion(self,guess,target, initial=None):
        #mse_weight, sobolev_weight = self.convex_combination()
        #mse_weight=1
        #mse = mse_weight*self.mse(target,guess)
        #sobolev_weight = torch.exp(self.log_derivative_weight)
        #sobolev_weight=1
        #sobolev = sobolev_weight*self.sobolev(target,guess)
        #md = self.maxdiff(target,guess)
        #phys = average_flux_conservation_loss(initial, guess)
        #return sobolev+mse+0.1*md+0.1*phys
        #print('mse %0.2e phys %0.2e'%(mse,phys))
        #hl = self.hl(guess,target)
        #tv = torch.abs(guess[...,1:]-guess[...,:-1]).mean()
        L1 = self.l1(target,guess)
        #high_k = self.fft_penalty(target,guess)
        #print("L1 %0.2e sob %0.2e tv %0.2e"%(L1,sobolev,tv))
        #if torch.isnan(tv):
        #    pdb.set_trace()
        return L1#+mse#+sobolev#+0.1*tv#+high_k#
    def forward(self, x):
        batch_size=x.shape[0]

        # FC2 block: reprocess globally
        x_flat = self.fc2(x)
        x = x_flat.view(batch_size,3, self.output_length)
        #x = self.attn(x)

        # Conv block 2: refine locally again
        #x = x + self.conv2(x)
        #conv 3
        x = x + self.conv3(x)

        #x = x.view(3,self.output_length)

        return x  # shape (B, 3, output_length)

