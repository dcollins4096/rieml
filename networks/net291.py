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

idd = 291
what = "proper Unet and pooling, less FC"

def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)

def thisnet():

    hidden_dims = 512,512   
    conv_channels = 16
    model = main_net(base_channels=conv_channels)
    return model

def train(model,data,parameters, validatedata, validateparams):
    epochs = 30000
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class main_net(nn.Module):
    def __init__(self, output_length=1000, base_channels=16):
        super().__init__()
        self.output_length = output_length

        # Project global 6D input → pseudo-spatial representation
        self.fc1 = nn.Linear(6, 3 * output_length)
        self.relu1 = nn.ReLU()

        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(3, base_channels)            # L
        self.enc2 = ConvBlock(base_channels, base_channels*2)  # L/2
        self.enc3 = ConvBlock(base_channels*2, base_channels*4) # L/4

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*4, base_channels*8)

        # Decoder (Upsampling path)
        self.up3 = UpBlock(base_channels*8, base_channels*4)
        self.up2 = UpBlock(base_channels*4, base_channels*2)
        self.up1 = UpBlock(base_channels*2, base_channels)

        # Output projection
        self.out_conv = nn.Conv1d(base_channels, 3, kernel_size=1)

        # Loss functions
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.2)

        # Weight initialization
        self.apply(self.init_weights)

    @staticmethod
    def init_weights_kamming(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.constant_(m.bias, 0.1)

    def criterion(self, guess, target, initial=None):
        L1 = self.l1(guess, target)
        return L1

    def forward(self, x):
        batch_size = x.shape[0]

        # Project global → spatial
        x = self.fc1(x)
        x = self.relu1(x)
        x = x.view(batch_size, 3, self.output_length)

        # Encoder
        e1 = self.enc1(x)                # (B, C, L)
        e2 = self.enc2(F.avg_pool1d(e1, 2))  # (B, 2C, L/2)
        e3 = self.enc3(F.avg_pool1d(e2, 2))  # (B, 4C, L/4)

        # Bottleneck
        b = self.bottleneck(F.avg_pool1d(e3, 2))  # (B, 8C, L/8)

        # Decoder
        d3 = self.up3(b, e3)             # (B, 4C, L/4)
        d2 = self.up2(d3, e2)           # (B, 2C, L/2)
        d1 = self.up1(d2, e1)           # (B, C, L)

        # Final output
        out = self.out_conv(d1)         # (B, 3, L)

        return out


class ConvBlock(nn.Module):
    """
    Two-layer convolutional block with ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block with skip connection
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv = ConvBlock(in_channels=out_channels*2, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # If needed, crop skip to match x
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            skip = skip[..., :x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

