import sys
import struct

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def load_parameters(filename):

    with open(filename, 'rb') as f:
        
        nanims, nparams = struct.unpack('II', f.read(8))
        return np.frombuffer(f.read(nanims*nparams*4), dtype=np.float32, count=nanims*nparams).reshape([nanims, nparams])
        
def save_network(filename, layers, mean_in, std_in, mean_out, std_out):
    
    with torch.no_grad():
        
        with open(filename, 'wb') as f:
            f.write(struct.pack('I', mean_in.shape[0]) + mean_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_in.shape[0]) + std_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', mean_out.shape[0]) + mean_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', std_out.shape[0]) + std_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack('I', len(layers)))
            for layer in layers:
                f.write(struct.pack('II', *layer.weight.T.shape) + layer.weight.T.cpu().numpy().astype(np.float32).ravel().tobytes())
                f.write(struct.pack('I', *layer.bias.shape) + layer.bias.cpu().numpy().astype(np.float32).ravel().tobytes())

# Networks

class Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(Network, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x
        
# Training procedure

if __name__ == '__main__':
    
    # Load data
    
    P = load_parameters('./parameters_speedturn.bin').astype(np.float32)
    # P = load_parameters('./parameters_traj.bin').astype(np.float32)
    
    nanims = P.shape[0]
    nparams = P.shape[1]
    
    # Training Parameters
    
    seed = 1234
    batchsize = 32
    lr = 0.001
    niter = 500000
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    # Make PyTorch tensors
    
    P = torch.as_tensor(P)
    Pmin = torch.min(P, dim=0).values - 0.1
    Pmax = torch.max(P, dim=0).values + 0.1
    
    D = torch.sqrt(torch.sum(torch.square(P[None,:] - P[:,None]), dim=-1))
    
    # Make network
    
    network = Network(nanims, nanims)
    network_mean_in = D.mean() * torch.ones(nanims, dtype=torch.float32)
    network_std_in = D.std() * torch.ones(nanims, dtype=torch.float32)
    network_mean_out = 0.5 * torch.ones(nanims, dtype=torch.float32)
    network_std_out = 0.5 * torch.ones(nanims, dtype=torch.float32)
    network_name = 'network_speedturn.bin' if nparams == 2 else 'network_traj.bin'
    
    # Train
    
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(
        list(network.parameters()),
        lr=lr,
        amsgrad=True,
        weight_decay=0.001)
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: 1.0 - (i / (niter - 1)))
        
    rolling_loss = None
    
    sys.stdout.write('\n')
    
    for i in range(niter):
    
        optimizer.zero_grad()
        
        # Sample random points in the parameter space
        Prand = (Pmax - Pmin) * torch.rand(batchsize, nparams, dtype=torch.float32) + Pmin

        # Compute the distances for those random points
        Drand = torch.sqrt(torch.sum(torch.square(P[None,:] - Prand[:,None]), dim=-1))

        # Concatenate to the points and distances from our animations
        Pbatch = torch.cat([P, Prand], dim=0)
        Dbatch = torch.cat([D, Drand], dim=0)

        # Compute the weights by going through the neural network
        W = (network((Dbatch - network_mean_in) / network_std_in) * 
            network_std_out + network_mean_out)

        # Compute the normalized weights (i.e positive and sum to one)
        Wnorm = torch.maximum(W, torch.zeros_like(W))
        Wnorm = Wnorm / torch.maximum(
            Wnorm.sum(dim=-1)[...,None], 1e-4 * torch.ones_like(Wnorm[:,:1]))

        # Loss to make weights at the animations 1 for that anim and 0 elsewhere
        loss_data = torch.mean(torch.abs(W[:nanims] - torch.eye(nanims)))

        # Loss to make the weights sum to one
        loss_sum = 0.01 * torch.mean(torch.abs(W.sum(dim=-1) - 1))

        # Loss to make the weights positive
        loss_pos = 0.01 * torch.mean(torch.abs(torch.minimum(W, torch.zeros_like(W))))

        # Loss to make the input match the output in parameter space
        loss_proj = 0.1 * torch.mean(
            torch.sqrt(torch.sum(torch.square((Wnorm @ P) - Pbatch), dim=-1) + 1e-4))

        # loss_proj_l1 = 0.1 * torch.mean(
            # torch.mean(torch.abs((Wnorm @ P) - Pbatch), dim=-1))

        # Add together losses
        loss = loss_data + loss_sum + loss_pos + loss_proj
        loss.backward()

        optimizer.step()
        scheduler.step()
    
        # Logging
        
        writer.add_scalar('blendspace/loss', loss.item(), i)
        
        writer.add_scalars('blendspace/loss_terms', {
            'data': loss_data.item(),
            'sum': loss_sum.item(),
            'pos': loss_pos.item(),
            'proj': loss_proj.item()
        }, i)
        
        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %6.4f' % (i, rolling_loss))
        
        if i % 1000 == 0:
            save_network(network_name, [
                network.linear0, 
                network.linear1,
                network.linear2,
                network.linear3],
                network_mean_in,
                network_std_in,
                network_mean_out,
                network_std_out)
            