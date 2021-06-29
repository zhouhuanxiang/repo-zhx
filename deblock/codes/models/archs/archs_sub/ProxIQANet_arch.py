import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torchvision.models.resnet as resnet


class ProxIQANet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, use_sigmoid):
        super(ProxIQANet, self).__init__()

        conv1 = [
            nn.Conv2d(6, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        self.conv1 = nn.Sequential(*conv1)

        conv2 = [
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        self.conv2 = nn.Sequential(*conv2)

        conv3 = [
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        self.conv3 = nn.Sequential(*conv3)

        conv4 = [
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        self.conv4 = nn.Sequential(*conv4)

        self.fc = nn.Linear(32768, 1)

        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.fc], 0.1)

    def forward(self, x, y):

        concat = torch.cat((x, y), 1)
        out = self.conv1(concat)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape([out.shape[0], -1])
        out = self.fc(out)

        if self.use_sigmoid:
            out = self.sigmoid(out)

        return out
