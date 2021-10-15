import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(32*13*13,10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # xreshaped = x.reshape(x[0], -1)
        a1 = torch.nn.functional.relu(self.conv(x))
        mp1 = self.mp(a1)
        outs = self.fc(mp1.reshape(mp1.shape[0],-1))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs