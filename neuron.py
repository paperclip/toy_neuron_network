

def relu(output):
    if output < 0:
        return 0
    return output


def relu_prime(z):
    if z > 0:
        return 1
    return 0


def cost(yHat, y):
    return 0.5 * (yHat - y)**2


def cost_prime(yHat, y):
    return yHat - y


class Neuron(object):
    def __init__(self, weights, bias):
        self.__m_weights = weights
        self.__m_activation = relu
        self.__m_activation_prime = relu_prime
        self.__m_bias = bias
        self.__m_learning_rate = 0.001

    def __weighted_sum(self, inputs):
        w = self.__m_weights
        assert len(w) == len(inputs)
        return sum((w[i] * inputs[i] for i in range(len(w)) ))

    def weight(self, pos):
        return self.__m_weights[pos]

    def weights(self):
        return self.__m_weights

    def forward_evaluate(self, inputs):
        self.__m_most_recent_inputs = inputs
        output = self.__m_activation(self.__weighted_sum(inputs) + self.__m_bias)
        self.__m_most_recent_output = output
        return output

    def __apply_error(self, error):
        error *= self.__m_activation_prime(self.__m_most_recent_output)
        self.__m_most_recent_error = error

        w = self.__m_weights
        inputs = self.__m_most_recent_inputs
        assert len(w) == len(inputs)

        for i in range(len(w)):
            w[i] -= self.__m_learning_rate * error * inputs[i]

    def back_propogate(self, actual_value):
        """
        output layer back_prop
        """
        ## Possibly actual_value - self.__m_most_recent_output ??
        error = self.__m_most_recent_output - actual_value
        return self.__apply_error(error)

    def most_recent_error(self):
        return self.__m_most_recent_error

    def back_propograte_hidden(self, next_layer, my_position):
        error = 0.0
        for neuron in next_layer:
            error += neuron.weight(my_position) * \
                neuron.most_recent_error()
        return self.__apply_error(error)

