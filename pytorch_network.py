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

plot_dir = "%s/plots"%os.environ['HOME']

class NikhilsUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,base_filters=32):
        super().__init__()
        self.enc1= nn.Sequential(
            nn.Conv1d(in_channels,base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters,base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters*2, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv2 = nn.ConvTranspose1d(base_filters*2, base_filters,
                                          kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Conv1d(base_filters, out_channels, kernel_size=1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.upconv2(e2)
        d2 = torch.cat([d1,e1],dim=1) #dcc question is this right?
        #print("SHAPES", d1.shape, e1.shape, d2.shape)
        d2 = self.dec2(d2)

        return self.outconv(d2)

def test_plot(datalist, model):
    for nd, data in enumerate(datalist):
        print('test',nd)
        x,y=data
        print(x.shape)
        myx=torch.tensor(x, dtype=torch.float32).view(1,3,1000)
        #pdb.set_trace()
        z = model(myx)

        fig,ax=plt.subplots(1,3,figsize=(12,4))
        fields = ['density','pressure','velocity']
        for nf,field in enumerate(fields):
            ax[nf].plot( x[nf] , c='k')
            ax[nf].plot( y[nf] , c='k', linestyle='--')
            zzz = z[0][nf].detach().numpy()
            print("Is nan", np.isnan(zzz).sum())
            ax[nf].plot( zzz, c='r')
            ax[nf].set(ylabel=field)
        fig.tight_layout()
        fig.savefig("%s/test_%d"%(plot_dir,nd))
        plt.close(fig)
    return zzz

class Conv1DThreeChannel(nn.Module):
    def __init__(self):
        super(Conv1DThreeChannel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # input channels = 3
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 3, kernel_size=3, padding=1)   # output channels = 3
        )

    def forward(self, x):
        # x shape: (batch_size, 3, input_length)
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # shape: (batch_size, 3, input_length)



# Example usage:
#model = Conv1DThreeChannel(input_length=128)
#example_input = torch.randn(10, 3, 128)  # batch of 10, 3 channels, length 128
#output = model(example_input)
#print(output.shape)  # should be (10, 3, 128)

class FeedforwardNN(nn.Module):
    def __init__(self, sizes):
        super(FeedforwardNN, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_network(model, training_data, epochs=1, mini_batch_size=1, lr=1, test_data=None):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #optimizer = optim.Adam( model.parameters, lr=lr)

    for epoch in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

        errs = []
        for mini_batch in mini_batches:
            #x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x, y in mini_batch])
            x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x, y in mini_batch])
            y_batch = torch.stack([torch.tensor(y, dtype=torch.float32) for x, y in mini_batch])

            optimizer.zero_grad()
            output = model(x_batch)
            #pdb.set_trace()
            oot = output.detach().numpy()
            if np.isnan(oot).any():
                print('nan')
                pdb.set_trace()
            loss = criterion(output, y_batch)
            errs.append(loss.item())
            loss.backward()
            optimizer.step()

        if test_data:
            test_results = [(torch.argmax(model(torch.tensor(x, dtype=torch.float32)), dim=0) == torch.argmax(torch.tensor(y, dtype=torch.float32))).item()
                            for x, y in test_data]
            accuracy = sum(test_results) / len(test_results)
            print(f"Epoch {epoch + 1}: {accuracy * 100:.2f}% accuracy")
        else:
            print(f"Epoch {epoch + 1} complete {np.mean(errs):.2e} {np.std(errs):.2e}")
