#!/bin/env python3

import os
import sys

import neuron

class Network(object):
    def __init__(self):
        self.__m_layer1 = (
            neuron.Neuron([1,2,3],5),
            neuron.Neuron([1,3,4],5),
            neuron.Neuron([1,2,3],5),
        )
        self.__m_layer2 = neuron.Neuron([1,2,3],3)

    def forward_evaluate(self, inputs):
        layer1_outputs = [ n.forward_evaluate(inputs) for n in self.__m_layer1 ]
        # print(layer1_outputs)
        layer2_output = self.__m_layer2.forward_evaluate(layer1_outputs)
        # print(layer2_output)
        return layer2_output


def main(argv):
    network = Network()
    print(network.forward_evaluate([0,0,0]))
    print(network.forward_evaluate([1,2,3]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
