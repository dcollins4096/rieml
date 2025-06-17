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
plot_dir = "%s/plots"%os.environ['HOME']

def train(model, data,parameters, epochs=1, lr=1e-3, batch_size=10, test_num=0, weight_decay=None):
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    fptr = open('output','w')
    fptr.close()
    #optimizer = optim.AdamW( model.parameters(), lr=lr, weight_decay = weight_decay)
    optimizer = optim.Adam( model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    n=-1
    losses=[]
    a = torch.arange(len(data))
    N = len(data)
    for epoch in range(epochs):
        subset = torch.tensor(random.sample(list(a),batch_size))
        #random.shuffle(a)
        #subset = a[:batch_size]
        data_subset =  data[subset]
        param_subset = parameters[subset]
        #print(subset)
        #pdb.set_trace()

        local_losses=[]
        for datum, param in zip(data_subset, param_subset):
        #for datum, param in zip(data, parameters):
            #pdb.set_trace()
            n+=1
            optimizer.zero_grad()
            output1=model(param)
            #output = output1.view(3,1000)
            #loss = model.criterion(output1, datum[0], param)
            loss = model.criterion(output1, datum[1])
            #print(datum[0][0][:10])

            loss.backward()
            optimizer.step()
            local_losses.append(loss.item())
        scheduler.step()
        losses += local_losses

        print("Epoch %d set %d %0.2e"%(epoch, n,np.mean(local_losses)))
        fptr = open("output","r+")
        fptr.write("Epoch %d set %d %0.2e\n"%(epoch, n,np.mean(local_losses)))
        fptr.close()
            #if n%10000 == 0 and n>1:
            #    plt.clf()
            #    plt.plot(losses)
            #    plt.yscale('log')
            #    plt.savefig("%s/loss_test_%d_%d"%(plot_dir,test_num,n))
    print("Epoch %d set %d %0.2e"%(epoch, n,np.mean(local_losses)))
    plt.clf()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig("%s/loss_test_%d_%d"%(plot_dir,test_num,n))


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
        #x = x.view(-1,3,self.output_length)
        x = x+ self.conv(x)

        # Reshape to (batch_size, 3, L)
        #x = x.view(-1, 3, self.output_length)
        x = x.view( 3, self.output_length)
        return x

def test_plot(datalist, parameters,model, fname="plot"):
    nd=-1
    for datum, param in zip(datalist,parameters):
        nd+=1
        #pdb.set_trace()
        z = model(param)
        loss = model.criterion(z, datum[0])
        print(loss)

        fig,ax=plt.subplots(1,3,figsize=(12,4))
        fields = ['density','pressure','velocity']
        for nf,field in enumerate(fields):
            ax[nf].plot( datum[0][nf], c='k')
            ax[nf].plot( datum[1][nf], c='k', linestyle='--')
            zzz = z[nf].detach().numpy()
            print("Is nan", np.isnan(zzz).sum(), nd, nf)
            ax[nf].set(title='error %0.2e'%loss)
            ax[nf].plot( zzz, c='r')
            ax[nf].set(ylabel=field)
        ax[0].set(ylim=[0,2])
        ax[1].set(ylim=[0,2])
        ax[2].set(ylim=[-1.1,1.1])
        fig.tight_layout()
        fig.savefig("%s/rieML_%s_%d"%(plot_dir,fname,nd))
        plt.close(fig)
    return zzz
