from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    '''
    Max Pooling of input
    '''
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        '''
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        '''
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        h0 = x

        for m in self.modules:
            h1 = m.forward(h0)
            h0 = h1

        last_layer = self.criterion
        probs, loss = last_layer.forward(h0, y)
        last_layer.backward()

        last_dx = last_layer.dx
        self.last_dx = last_dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        '''
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        current_dout = self.last_dx
        self.modules.reverse() #Reversing the order of modules. 
        for m in self.modules:
            m.backward(current_dout) #Updating dx
            current_dout = m.dx #assigning updated dx to dout
        self.modules.reverse()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################