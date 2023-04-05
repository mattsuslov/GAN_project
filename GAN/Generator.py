import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, n_classes, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.fc = nn.Sequential(
            nn.Linear(self.n_classes + latent_size, 512*4*4),
            nn.ReLU()
        )
    
        def block(in_channels, out_channels, kernel_size, padding, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        self.deconv = nn.Sequential(
            block(512, 256, 4, 1, 2),
            block(256, 128, 4, 1, 2),
            block(128, 64, 4, 1, 2),
            block(64, 32, 4, 1, 2),
            nn.ConvTranspose2d(32, 3, 5, 1, 2)
        )
    
    def forward(self, z, label):
        label = label.reshape(label.shape[0], self.n_classes, 1, 1)
        x = self.fc(torch.cat([z, label], dim=1).view(-1, self.latent_size + self.n_classes))
        x = x.view(-1, 512, 4, 4)
        x = self.deconv(x)
        return torch.tanh(x)