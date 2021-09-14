import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['lenet', 'resnet18', 'resnet18cifar']


class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        resnet = models.resnet18(pretrained=False, num_classes=10)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.maxpool = torch.nn.Identity()
        self.model = resnet

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

class Resnet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        res_mod = models.resnet34(pretrained=False)
        num_ftrs = res_mod.fc.in_features
        res_mod.fc = nn.Linear(num_ftrs, num_classes)
        self.model = res_mod

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)




def lenet(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)

def resnet18(num_classes, **kwargs):
    return Resnet18(num_classes, **kwargs)

def resnet18cifar(num_classes, **kwargs):
    return Resnet18CIFAR(num_classes, **kwargs)
