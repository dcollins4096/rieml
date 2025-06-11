
import mnist_loader
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
#import network
import pytorch_network as pyt
from importlib import reload
reload(pyt)
#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#model = pyt.FeedforwardNN([784,30,10])
model = pyt.FeedforwardNN([784,30,10])
pyt.train_network(model, training_data,30,11,3)

