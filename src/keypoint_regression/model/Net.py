import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchsummary import summary


class MobilenetV2_fully_connected(nn.Module):
    def __init__(self, pretrained = True):
        super(MobilenetV2_fully_connected, self).__init__()
        self.pretrained = pretrained
        self.model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=self.pretrained)
        self.linear = nn.Linear(1000, 50)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.model(x)
        x = self.linear(x)
        x = self.relu(x)
        return x


class MobilenetV2_feature_map(nn.Module):
    def __init__(self, layers):
        super(MobilenetV2_feature_map, self).__init__()
        # layers = list(mobilenet_v2.children())[0]
        self.layers = layers
        self.layers[18][0] = nn.Conv2d(320, 512, (1,1), (1,1), bias = False)
        self.layers[18][1] = nn.BatchNorm2d(512)
        self.layers[18][2] = nn.ReLU6(True)
        self.layers.add_module("global_pool-157", nn.MaxPool2d(2,2))
        self.layers.add_module("Conv2d-158", nn.Conv2d(512,10, (3,3), stride =(1,1), padding = (1,1), bias = False))
        self.layers.add_module("BatchNorm2d-159", nn.BatchNorm2d(10))
        self.layers.add_module("ReLU-160", nn.ReLU6(True))
        # self.layers.add_module("global_pool-160", nn.MaxPool2d(2,2))
        # self.layers.add_module("Conv2d-161", nn.Conv2d(128,32, (3,3), stride =(1,1), padding = (1,1), bias = False))
        # self.layers.add_module("BatchNorm2d-162", nn.BatchNorm2d(32))
        # self.layers.add_module("ReLU-163", nn.ReLU6(True))
        # self.classifier = nn.Sequential(
        #     nn.Linear(32 * 4 * 4, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 10),
        # )
    def forward(self,x):
        x = self.layers(x)

        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = MobilenetV2_fully_connected()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # print(model)
    print(summary(model, (3, 224, 224)))
