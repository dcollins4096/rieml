import torch
import torch.nn as nn
import torch.nn.functional as F
import rieML_model
import ptoc

class HybridShockTubeNN(nn.Module):
    def __init__(self, output_length=1000, hidden_dims=(128, 128), conv_channels=32, characteristic=False):
        super().__init__()
        self.output_length = output_length

        # Project 6 input values to a pseudo-spatial format (3 channels)
        self.fc1 = nn.Linear(6, 3 * output_length)
        self.relu1 = nn.ReLU()

        # Conv block 1 (acts on the "3 x output_length" format)
        dil = 1
        kern = 5
        padding = dil*(kern-1)//2
        dil2 = 2
        padding2 = dil2*(kern-1)//2
        dil3 = 4
        padding3 = dil3*(kern-1)//2
        dil4 = 8
        padding4 = dil4*(kern-1)//2
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU(),
            #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding2, dilation=dil2),
            #nn.ReLU(),
            #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding3, dilation=dil3),
            #nn.ReLU(),
            #nn.Conv1d(conv_channels, conv_channels, kernel_size=kern, padding=padding4, dilation=dil4),
            #nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU()
        )

        # FC block 2: merge spatial info
        self.fc2 = nn.Sequential(
            nn.Linear(3 * output_length, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 3 * output_length)
        )

        # Conv block 2
        dil = 1
        kern = 3
        padding = dil*(kern-1)//2
        self.conv2 = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=kern, padding=padding, dilation=dil)
        )
        #if characteristic:
        #    self.forward=self.char_forward
        #else:
        #    self.forward=self.prim_forward
        self.T = nn.Parameter(torch.eye(3) + 0.01 * torch.randn(3, 3)) 
        self.forward = self.prim_forward

        self.mse=nn.MSELoss()
        #self.log_derivative_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_tv_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_high_k_weight = nn.Parameter(torch.tensor(0.0)) 

    def criterion(self,target,guess):
        mse = self.mse(target,guess)
        #dx_target = target[:,1:]-target[:,:-1]
        #dx_guess  = guess[:,1:]-guess[:,:-1]
        #sobolev_weight = torch.exp(self.log_derivative_weight)
        #sobolev = sobolev_weight*self.mse(dx_target,dx_guess)
        #sobolev = self.mse(dx_target,dx_guess)

        #high_k_weight = torch.exp(self.log_high_k_weight)
        #high_k = high_k_weight*rieML_model.high_frequency_penalty(guess)

        #tv_weight = torch.exp(self.log_tv_weight)
        #tv = tv_weight*torch.abs(guess[1:]-guess[:-1]).mean()
        #smooth = smoothness_loss(guess)
        #fourth_loss = fourth(target,guess)
        #output = self.mse(target,guess) + smoothness_loss(guess)
        #print("MSE %0.2e smooth %0.2e"%(mse,smooth))
        #print("Mse %0.2e k  %0.2e"%(mse,high_k))
        #print("Mse %0.2e sob  %0.2e"%(mse,sobolev))
        #return mse+tv+high_k+sobolev
        return mse#+sobolev

    def prim_forward(self, x):

        # FC1 to expand global features into spatial representation
        x = self.fc1(x)  # (batch_size, 3*output_length)
        x = self.relu1(x)
        x = x.view(1, 3, self.output_length)  # shape (B, 3, L)

        # Conv block 1: local patterns
        x = x + self.conv1(x)  # Residual connection

        # FC2 block: reprocess globally
        x_flat = x.view( -1)
        x_flat = self.fc2(x_flat)
        x = x_flat.view(1,3, self.output_length)

        # Conv block 2: refine locally again
        x = x + self.conv2(x)
        x = x.view(3,self.output_length)

        return x  # shape (B, 3, output_length)

    def char_forward(self, x_in):
        d1,d2,p1,p2,v1,v2=x_in
        left = torch.tensor([d1,v1,p1],dtype=torch.float32) 
        right = torch.tensor([d2,v2,p2],dtype=torch.float32) 
        mean = 0.5*(left-right)
        #left = left - mean
        #right = right-mean
        mean_rho = mean[0]
        mean_p   = mean[2]

        leftc = ptoc.primitive_to_characteristic(left)
        rightc = ptoc.primitive_to_characteristic(right)
        x = torch.cat([leftc,rightc])

        # FC1 to expand global features into spatial representation
        x = self.fc1(x)  # (batch_size, 3*output_length)
        x = self.relu1(x)
        x = x.view(1, 3, self.output_length)  # shape (B, 3, L)

        # Conv block 1: local patterns
        x = x + self.conv1(x)  # Residual connection

        # FC2 block: reprocess globally
        x_flat = x.view( -1)
        x_flat = self.fc2(x_flat)
        x = x_flat.view(1,3, self.output_length)

        # Conv block 2: refine locally again
        x = x + self.conv2(x)
        x = x.view(3,self.output_length)
        
        x = ptoc.characteristic_to_primitive(x, mean_rho, mean_p)

        return x  # shape (B, 3, output_length)

