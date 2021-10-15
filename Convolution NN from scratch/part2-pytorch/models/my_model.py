import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.mp1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.mp2 = nn.MaxPool2d(2, stride=1)
        self.fc = nn.Linear(64*16*16,32)
        self.fc2 = nn.Linear(32,10)
        
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=2)
        # self.mp1 = nn.MaxPool2d(2, stride=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        # self.mp2 = nn.MaxPool2d(2, stride=1)
        # self.fc = nn.Linear(64*8*8,128)
        # self.fc2 = nn.Linear(128,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.mp1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.mp2(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        outs = self.fc2(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs