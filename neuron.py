

def relu(output):
    if output < 0:
        return 0
    return output


class Neuron(object):
    def __init__(self, weights, bias):
        self.__m_weights = weights
        self.__m_activation = relu
        self.__m_bias = bias

    def __weighted_sum(self, inputs):
        w = self.__m_weights
        assert len(w) == len(inputs)
        return sum((w[i] * inputs[i] for i in range(len(w)) ))

    def forward_evaluate(self, inputs):
        return self.__m_activation(self.__weighted_sum(inputs) + self.__m_bias)
