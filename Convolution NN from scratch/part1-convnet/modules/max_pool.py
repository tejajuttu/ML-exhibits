import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N,C,H,W = x.shape
        out = np.zeros((N,C, int((H-self.kernel_size)/self.stride)+1, int((W-self.kernel_size)/self.stride)+1))

        hs = 0
        ws = 0
        for n in range(N):
            for c in range(C):
                hs=0
                ws=0
                for h in range(0,H, self.stride):
                    ws=0
                    for w in range(0,W,self.stride):
                        max_val = np.max(x[n,c,h:h+self.kernel_size,w:w+self.kernel_size])
                        out[n,c,hs,ws] = max_val
                        ws+=1
                    hs+=1


        H_out = 0
        W_out = 0
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N,C,H,W = x.shape
        dx = np.zeros(x.shape)
        hmax, wmax = (int((H-self.kernel_size)/self.stride)+1, int((W-self.kernel_size)/self.stride)+1) #h_out,w_out

        for n in range(N):
            for c in range(C):
                for h in range(hmax):
                    for w in range(wmax):
                        temp = x[n,c,self.stride*h:self.stride*h+self.kernel_size, self.stride*w:self.stride*w+self.kernel_size]
                        req_index = np.unravel_index(np.argmax(temp), temp.shape)
                        dx[n,c,self.stride*h:self.stride*h+self.kernel_size,self.stride*w : self.stride*w+ self.kernel_size][req_index] = dout[n,c,h,w]
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
