from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight
        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                m.velocity = np.zeros(m.weight.shape)
            if hasattr(m, 'bias'):
                m.velocity_bias = np.zeros(m.bias.shape)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                m.velocity = self.momentum*m.velocity - self.learning_rate*m.dw
                m.weight += m.velocity
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                m.velocity_bias = self.momentum*(m.velocity_bias) - self.learning_rate*m.db
                m.bias += m.velocity_bias
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################