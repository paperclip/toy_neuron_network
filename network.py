#!/bin/env python3

import os
import random
import sys
import time

import neuron

def target_function(input):
    return input[0] + input[1] - input[2]

LEARNING_RATE = 0.001

class Network(object):
    def __init__(self):
        self.__m_layer1 = (
            neuron.Neuron([1,2,3],5),
            neuron.Neuron([1,3,4],5),
            neuron.Neuron([1,2,3],5),
        )
        self.__m_layer2 = [ neuron.Neuron([1,2,3],3) ]

    def forward_evaluate(self, inputs):
        layer1_outputs = [ n.forward_evaluate(inputs) for n in self.__m_layer1 ]
        # print(layer1_outputs)
        layer2_output = self.__m_layer2[0].forward_evaluate(layer1_outputs)
        # print(layer2_output)
        return layer2_output

    def back_propagate(self, actual):
        for n in self.__m_layer2:
            n.back_propogate(actual)
        for i, n in enumerate(self.__m_layer1):
            n.back_propograte_hidden(self.__m_layer2, i)

    def printout(self):
        print("LAYER1:",[ n.weights() for n in self.__m_layer1])
        print("LAYER2:", [n.weights() for n in self.__m_layer2])

def generate_input():
    while True:
     yield [ random.uniform(0.0, 100.0) for _ in range(3) ]

def square_error(predicted, actual):
    return (predicted - actual) ** 2

def train_batch(network, training_function):
    batch_size = 10

    while True:
        n = 0
        sum_of_error = 0.0
        for i in generate_input():
            actual = training_function(i)
            predicted = network.forward_evaluate(i)
            error = square_error(predicted, actual)
            sum_of_error += error
            n += 1
            if n > batch_size:
                break
        
        average_square_error = sum_of_error / n
        network.back_propagate(i, average_square_error)
        time.sleep(0.1)

def train(network, training_function):
    iteration = 0
    for i in generate_input():
        print(iteration)
        iteration += 1
        actual = training_function(i)
        predicted = network.forward_evaluate(i)
        print(i,"->",predicted,"should be",actual)
        network.back_propagate(actual)
        network.printout()
        print()
        time.sleep(0.1)


def main(argv):
    network = Network()
    # print(network.forward_evaluate([0,0,0]))
    # print(network.forward_evaluate([1,2,3]))
    train(network, target_function)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
