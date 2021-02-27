# %%

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random

# %%

manualSeed = 443
print(f"Seed: {manualSeed}")
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# %%

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# %%

netG = torch.load('../../data/models/dcgan/dcgan_gen.pt')
netG.to(device)

# %%

latent_space = torch.randn(64, nz, 1, 1, device=device)

# %%

face_output = netG(latent_space).detach().cpu()

# %%

output = vutils.make_grid(face_output, padding=2, normalize=True)

fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.imshow(np.transpose(output,(1,2,0)), animated=True)
# %%
