
import pytorch_network as pyt
from importlib import reload
import torch
torch.backends.nnpack.enabled = False
reload(pyt)

import mnist_loader
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
#import network
#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#model = pyt.FeedforwardNN([784,30,10])
if 0:
    #MNIST number tool
    model = pyt.FeedforwardNN([784,30,10])
    pyt.train_network(model, training_data,30,11,3)

import tube_loader
import pdb
#data = tube_loader.load_many()
data = tube_loader.read_one("tubes_take2.h5")
Ntrain=109
train = data[:Ntrain]
test = data[Ntrain:]
##model = pyt.Conv1DThreeChannel()
model = pyt.NikhilsUnet()
pyt.train_network(model,data,epochs=100,mini_batch_size=5,lr=1e-2)
mini_batch = train[0:1]
#testme = torch.stack([torch.tensor(x, dtype=torch.float32) for x, y in mini_batch])
testme = train[0:1]
zzz=pyt.test_plot(testme,model)
