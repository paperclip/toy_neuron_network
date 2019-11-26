
import os
import sys
import unittest

import neuron

class TestNeuron(unittest.TestCase):

    def testConstruction(self):
        neuron.Neuron([1,2,3],5)

    def testEvaluate(self):
        n = neuron.Neuron([1,2,3],5)
        self.assertEqual(n.forward_evaluate([0,0,0]), 5)

    def testZero(self):
        n = neuron.Neuron([0],0)
        self.assertEqual(n.forward_evaluate([15]), 0)


def main(argv):
    unittest.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
