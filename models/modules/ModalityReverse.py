import math

import torch.nn as nn


class ModalityReverse(nn.Module):
    def __init__(self):
        super(ModalityReverse, self).__init__()

        # Common
        self.relu = nn.ReLU(inplace=True)

        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

        self.to_depth = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb_embedding, depth_embedding):
        return self.to_rgb(depth_embedding), self.to_depth(rgb_embedding)
