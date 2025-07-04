import torch
import torch.nn as nn
import torch.nn.functional as F
import rieML_model
import ptoc
import pdb


gamma_default=1.6666666667
def compute_flux(rho, u, p, gamma=gamma_default):
    momentum = rho * u
    energy = 0.5 * rho * u**2 + p / (gamma - 1)
    return torch.stack([
        momentum,
        momentum * u + p,
        u * (energy + p)
    ], dim=1)  # shape (batch, 3, L)
def average_flux_conservation_loss(U_init, U_final_pred, t_final=0.1, gamma=gamma_default):
    """
    Enforce U_final = U_init - 0.5 * t_final * (∂F_init/∂x + ∂F_final/∂x)
    """
    # U_init and U_final_pred: (batch, 3, L)
    rho_i, p_i, u_i = U_init[:, 0], U_init[:, 1], U_init[:, 2]
    rho_f, p_f, u_f = U_final_pred[:, 0], U_final_pred[:, 1], U_final_pred[:, 2]

    F_i = compute_flux(rho_i, u_i, p_i, gamma)
    F_f = compute_flux(rho_f, u_f, p_f, gamma)

    # Central differences for ∂F/∂x
    dFdx_i = F_i[..., 2:] - F_i[..., :-2]
    dFdx_f = F_f[..., 2:] - F_f[..., :-2]
    dFdx_avg = 0.5 * (dFdx_i + dFdx_f)

    # Adjust U_init and U_final to match shape
    U_i_crop = U_init[..., 1:-1]
    U_f_crop = U_final_pred[..., 1:-1]

    # Residual from conservation
    residual = U_f_crop - U_i_crop + t_final * dFdx_avg
    #print("WTF",residual.requires_grad)
    #pdb.set_trace()
    return torch.mean(residual ** 2)
def init_weights_constant(m):
    if isinstance(m, nn.Linear):
        #nn.init.constant_(m.weight, 0.5)
        nn.init.constant_(m.bias, 0.1)
class SelfAttention1D(nn.Module):
    def __init__(self, channels, length):
        super().__init__()
        self.query = nn.Linear(length, length)
        self.key = nn.Linear(length, length)
        self.value = nn.Linear(length, length)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, channels, length)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = self.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5))
        out = attn @ V
        return out + x  # residual


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
        if 0:
            self.fc2 = nn.Sequential(
                nn.Linear(3 * output_length, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], 3 * output_length)
            )
        if 0:
            in_dim = 3*output_length
            out_dim = 3*output_length
            layers=[]
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
                    print('relu')
            self.fc2 = nn.Sequential(*layers)
        if 1:
            in_dim = 3*output_length
            out_dim = 3*output_length
            layers=[]
            dims = [in_dim] + list(hidden_dims) + [out_dim]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.fc2 = nn.Sequential(*layers)

        # Conv block 2
        dil = 1
        kern = 3
        padding = dil*(kern-1)//2
        self.conv2 = nn.Sequential(
            nn.Conv1d(3, conv_channels, kernel_size=kern, padding=padding, dilation=dil),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 3, kernel_size=kern, padding=padding, dilation=dil)
        )
        dil1 = 2
        kern1 = 3
        padding1 = dil*(kern-1)//2
        dil2 = 2
        kern2 = 3
        padding2 = dil*(kern-1)//2
        if 1:
            self.conv3 = nn.Sequential(
                nn.Conv1d(3, conv_channels, kernel_size=kern1, padding=padding1, dilation=dil1),
                nn.ReLU(),
                nn.Conv1d(conv_channels, 2*conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                nn.ReLU(),
                nn.Conv1d(2*conv_channels, conv_channels, kernel_size=kern2, padding=padding2, dilation=dil2),
                nn.ReLU(),
                nn.Conv1d(conv_channels, 3, kernel_size=kern1, padding=padding1, dilation=dil1)
            )
        self.conv1.apply(init_weights_constant)
        self.conv2.apply(init_weights_constant)
        self.conv3.apply(init_weights_constant)
        self.fc2.apply(init_weights_constant)
        #if characteristic:
        #    self.forward=self.char_forward
        #else:
        #    self.forward=self.prim_forward
        self.T = nn.Parameter(torch.eye(3) + 0.01 * torch.randn(3, 3)) 

        self.mse=nn.MSELoss()
        self.log_derivative_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_tv_weight = nn.Parameter(torch.tensor(0.0)) 
        #self.log_high_k_weight = nn.Parameter(torch.tensor(0.0)) 
        self.criterion = self.criterion1

        #self.attn = SelfAttention1D(channels=3, length=output_length)
        #self.attn_rho = SelfAttention1D(channels=1, length=output_length)
        #self.attn_vel = SelfAttention1D(channels=1, length=output_length)
        #self.attn_pres = SelfAttention1D(channels=1, length=output_length)
        embed_dim=output_length
        num_heads = 4
        self.hl = nn.HuberLoss(delta=0.2)
        self.l1 = nn.L1Loss()

        #self.attn_rho =  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)
        #self.attn_vel =  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)
        #self.attn_pres=  nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,batch_first=True)


    def fft_penalty(self,target,guess, cutoff_ratio=0.3):
        """
        Penalize high-frequency components in the output.
        
        Args:
            output: (batch_size, channels, length)
            cutoff_ratio: fraction of frequencies to treat as low-frequency (0.3 means first 30%)
            
        Returns:
            scalar penalty term
        """

        # FFT along spatial dimension
        fft_t = torch.fft.rfft(target, dim=1)
        fft_g = torch.fft.rfft(guess, dim=1)
        
        # Power spectrum
        power_t = torch.abs(fft_t) ** 2
        power_g = torch.abs(fft_g) ** 2
        
        return self.l1(power_t,power_g)

    def sobolev_derivatives(self,target,guess):
        dx_target = target[:,1:]-target[:,:-1]
        dx_guess  = guess[:,1:]-guess[:,:-1]
        return dx_target, dx_guess
    def sobolev(self,target,guess):
        dx_target,dx_guess = self.sobolev_derivatives(target,guess)
        #sobolev = self.mse(dx_target,dx_guess)
        sobolev = self.l1(dx_target,dx_guess)
        return sobolev
    def convex_combination(self):
        theta = self.log_derivative_weight
        mse_weight = torch.cos(theta)
        sobolev_weight = 1-torch.cos(theta)
        mse_weight, sobolev_weight = 0.8,0.1
        return mse_weight, sobolev_weight

    def maxdiff(self,target,guess):
        return ((target-guess)**2).max()

    def criterion1(self,guess,target, initial=None):
        #mse_weight, sobolev_weight = self.convex_combination()
        #mse_weight=1
        #mse = mse_weight*self.mse(target,guess)
        #sobolev_weight = torch.exp(self.log_derivative_weight)
        #sobolev_weight=1
        #sobolev = sobolev_weight*self.sobolev(target,guess)
        #md = self.maxdiff(target,guess)
        #phys = average_flux_conservation_loss(initial, guess)
        #return sobolev+mse+0.1*md+0.1*phys
        #print('mse %0.2e phys %0.2e'%(mse,phys))
        #hl = self.hl(guess,target)
        #tv = torch.abs(guess[...,1:]-guess[...,:-1]).mean()
        L1 = self.l1(target,guess)
        #high_k = self.fft_penalty(target,guess)
        #print("L1 %0.2e sob %0.2e tv %0.2e"%(L1,sobolev,tv))
        #if torch.isnan(tv):
        #    pdb.set_trace()
        return L1#+high_k#+sobolev+0.1*tv
    def criterion2(self,guess,target, initial=None):
        mse_weight, sobolev_weight = self.convex_combination()
        mse_weight = 1
        mse = mse_weight*self.mse(target,guess)
        #sobolev = self.mse(dx_target,dx_guess)
        #sobolev_weight = torch.exp(self.log_derivative_weight)
        #sobolev = self.sobolev(target,guess)
        #md = self.maxdiff(target,guess)
        #phys = average_flux_conservation_loss(initial, guess)
        #return sobolev+mse+0.1*md+0.1*phys
        #print('mse %0.2e phys %0.2e'%(mse,phys))
        #tv = sobolev_weight*torch.abs(guess[1:]-guess[:-1]).mean()
        #print("mse %0.2e tv %0.2e"%(mse,tv))
        return mse


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
        #return mse#+sobolev

    def forward(self, x):
        batch_size=x.shape[0]
        # FC1 to expand global features into spatial representation
        x = self.fc1(x)  # (batch_size, 3*output_length)
        x = self.relu1(x)
        x = x.view(batch_size, 3, self.output_length)  # shape (B, 3, L)

        if 0:
            rho, vel, pres = torch.chunk(x, 3, dim=1)
            rho, weights = self.attn_rho(rho,rho,rho)
            vel, weights = self.attn_vel(vel,vel,vel)
            pres, weights=self.attn_pres(pres,pres,pres)
            x = torch.cat([rho, vel, pres], dim=1)

        # Conv block 1: local patterns
        x = x + self.conv1(x)  # Residual connection

        # FC2 block: reprocess globally
        x_flat = x.view(batch_size, -1)
        x_flat = self.fc2(x_flat)
        x = x_flat.view(batch_size,3, self.output_length)
        #x = self.attn(x)

        # Conv block 2: refine locally again
        x = x + self.conv2(x)
        #conv 3
        x = x + self.conv3(x)

        #x = x.view(3,self.output_length)

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

