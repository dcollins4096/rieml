import torch
import torch.nn as nn
import torch.nn.functional as F

class main_net(nn.Module):
    def __init__(self, output_length=1000, conv_channels=16):
        super().__init__()
        self.output_length = output_length

        # Project 6 global features to pseudo-spatial format (3 channels)
        self.fc1 = nn.Linear(6, 3 * output_length)
        self.relu1 = nn.ReLU()

        # Initial conv block (residual)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=3, padding=1)
        )

        # Replace fc2 with lightweight pointwise convs
        self.pointwise = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=1)
        )

        # Shallow U-Net style context capture
        self.down = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose1d(conv_channels, conv_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=3, padding=1)
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
        # You can also add gradient loss if desired
        L1 = self.l1(guess, target)
        return L1

    def forward(self, x):
        batch_size = x.shape[0]

        # FC1 to expand global features into spatial representation
        x = self.fc1(x)  # (B, 3*output_length)
        x = self.relu1(x)
        x = x.view(batch_size, 3, self.output_length)  # (B, 3, L)

        # Residual conv1
        x = x + self.conv1(x)

        # Downsample for context
        x_down = self.down(x)

        # Upsample and merge
        x_up = self.up(x_down)

        # Combine with pointwise conv for refinement
        x = x + self.pointwise(x_up)

        return x  # (B, 3, output_length)

