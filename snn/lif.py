import torch


class LIFNeuron:
    def __init__(self, beta=0.9, threshold=1.0):
        self.beta = beta
        self.threshold = threshold
        self.v = 0.0

    def step(self, input_current):
        # Leak
        self.v = self.beta * self.v

        # Integrate input
        self.v += input_current

        # Generate spike
        spike = 1.0 if self.v >= self.threshold else 0.0

        # Reset
        if spike:
            self.v = 0.0

        return spike