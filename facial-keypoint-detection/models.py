## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=5,
                               stride=2) # 110x110x32
        
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2) # 54x54x64
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2) # 26x26x128
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.dense1 = nn.Linear(in_features=26*26*128, out_features=1000)
        self.dense1_bn = nn.BatchNorm1d(1000)
        self.dense2 = nn.Linear(in_features=1000, out_features=512)
        self.dense2_bn = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(in_features=512, out_features=256)
        self.dense3_bn = nn.BatchNorm1d(256)
        self.dense4 = nn.Linear(256, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = F.relu(self.conv1(x))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.conv3_bn(F.relu(self.conv3(x)))
        
        # fully-connected
        x = x.view(x.size(0), -1)
        x = self.dense1_bn(F.relu(self.dense1(x)))
        x = self.dense2_bn(F.relu(self.dense2(x)))
        x = self.dense3_bn(F.relu(self.dense3(x)))
        x = self.dense4(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
