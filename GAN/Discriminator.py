import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes, photo_size):
        super(Discriminator, self).__init__()

        def block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2))
        
        self.conv = nn.Sequential(
            block(3 + 1, 32, 4, 1, 1),
            block(32, 64, 4, 2, 1),
            block(64, 128, 4, 2, 1),
            block(128, 256, 4, 2, 1),
            block(256, 512, 4, 2, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512*3*3, 1),
            nn.Sigmoid()
        )
        
        self.embed = nn.Embedding(n_classes, photo_size ** 2)
        
    def forward(self, x, label):
        y = self.embed(label).view(-1, 1, 64, 64)
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        x = x.view(-1, 512*3*3)
        x = self.fc(x)
        return x