import torch
import torch.nn as nn
import torch.optim as optim
import random
import pdb

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

def train_network(model, training_data, epochs, mini_batch_size, lr, test_data=None):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

        for mini_batch in mini_batches:
            #x_batch = torch.stack([torch.tensor(x, dtype=torch.float32) for x, y in mini_batch])
            x_batch = torch.stack([torch.tensor(x, dtype=torch.float32).view(-1) for x, y in mini_batch])
            y_batch = torch.stack([torch.tensor(y, dtype=torch.float32).view(-1) for x, y in mini_batch])
            #pdb.set_trace()

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if test_data:
            test_results = [(torch.argmax(model(torch.tensor(x, dtype=torch.float32)), dim=0) == torch.argmax(torch.tensor(y, dtype=torch.float32))).item()
                            for x, y in test_data]
            accuracy = sum(test_results) / len(test_results)
            print(f"Epoch {epoch + 1}: {accuracy * 100:.2f}% accuracy")
        else:
            print(f"Epoch {epoch + 1} complete")
