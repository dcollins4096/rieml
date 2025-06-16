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
    optimizer = optim.AdamW( model.parameters(), lr=lr, weight_decay = weight_decay)
    n=-1
    losses=[]
    a = torch.arange(len(data))
    N = len(data)
    for epoch in range(epochs):
        subset = torch.randint(0, N, (batch_size,))
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
            output = model(param).view(3,1000)
            loss = model.criterion(output, datum[0])
            #print(datum[0][0][:10])

            loss.backward()
            optimizer.step()
            local_losses.append(loss.item())
        losses += local_losses

        print("Epoch %d set %d %0.2e"%(epoch, n,np.mean(local_losses)))
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








class SixToThreeChannelNN(nn.Module):
    def __init__(self, output_length: int, hidden_dims=(64, 128, 256,128, 64)):
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
        self.net = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, 6)
        Returns:
            Tensor of shape (batch_size, 3, output_length)
        """
        # Forward pass through MLP → (batch_size, 3*L)
        y = self.net(x)
        # Reshape to (batch_size, 3, L)
        y = y.view(-1, 3, self.output_length)
        return y

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
            zzz = z[0][nf].detach().numpy()
            print("Is nan", np.isnan(zzz).sum(), nd, nf)
            ax[nf].set(title='error %0.2e'%loss)
            ax[nf].plot( zzz, c='r')
            ax[nf].set(ylabel=field)
        fig.tight_layout()
        fig.savefig("%s/rieML_%s_%d"%(plot_dir,fname,nd))
        plt.close(fig)
    return zzz
