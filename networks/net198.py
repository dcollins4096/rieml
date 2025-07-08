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

idd = 198
idd_text = "197 plus the FC layer"

def init_weights_kaming(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    hidden_dims = 1024  ,
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
        milestones=[10000, 20000],  # change after N and N+M steps
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
        if epoch>0 and (epoch%100==0 or epoch == 10):
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class main_net(nn.Module):
    def __init__(self, hidden_dims=[1024],output_length=1000, conv_channels=16):
        super().__init__()
        self.output_length = output_length

        # Project 6 global features to pseudo-spatial format (3 channels)
        self.fc1 = nn.Linear(6, 3 * output_length)
        self.relu1 = nn.ReLU()

        # Initial residual conv block
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=3, padding=1)
        )

        # Dilated residual block
        self.dilated_block = nn.Sequential(
            DilatedConvBlock(3, conv_channels, kernel_size=3, dilations=[1, 2, 4, 8]),
            nn.Conv1d(conv_channels, 3, kernel_size=1)  # compress back to 3 channels
        )

        # FC block 2: merge spatial info
        if 1:
            in_dim = 3*output_length
            out_dim = 3*output_length
            layers=[]
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.fc2 = nn.Sequential(*layers)

        # Pointwise refinement
        self.pointwise = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=1)
        )

        # Losses
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.2)

        # Weight init
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def criterion(self, guess, target, initial=None):
        L1 = self.l1(guess, target)
        return L1

    def forward(self, x):
        batch_size = x.shape[0]

        # Project to pseudo-spatial
        x = self.fc1(x)  # (B, 3*L)
        x = self.relu1(x)
        x = x.view(batch_size, 3, self.output_length)

        # Initial residual conv
        x = x + self.conv1(x)

        x_flat = x.view(batch_size, -1)
        x_flat = self.fc2(x_flat)
        x = x_flat.view(batch_size,3, self.output_length)

        # Dilated residual conv
        x = x + self.dilated_block(x)

        # Pointwise refinement
        x = x + self.pointwise(x)

        return x  # (B, 3, L)


class DilatedConvBlock(nn.Module):
    """
    Stack of dilated convolutions with increasing dilation rates.
    """
    def __init__(self, in_channels, conv_channels, kernel_size=3, dilations=[1,2,4,8]):
        super().__init__()
        layers = []
        for i, d in enumerate(dilations):
            pad = d * (kernel_size - 1) // 2
            layers.append(nn.Conv1d(
                in_channels if i == 0 else conv_channels,
                conv_channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=d
            ))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

