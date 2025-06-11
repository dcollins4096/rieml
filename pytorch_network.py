import torch
import torch.nn as nn
import torch.optim as optim
import random
import pdb
import torch
import torch.nn as nn

class Conv1DThreeChannel(nn.Module):
    def __init__(self, input_length):
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
