import matplotlib
matplotlib.use('WebAgg')

import matplotlib.pyplot as plt

class LIFNeuron:

    def __init__(self, beta=0.9, threshold=1.0):

        self.beta = beta
        self.threshold = threshold
        self.v = 0.0

    def step(self, input_current):

        # Leak
        self.v = self.beta * self.v

        # Integrate
        self.v += input_current

        # Spike
        spike = 1.0 if self.v >= self.threshold else 0.0

        # Reset
        if spike:
            self.v = 0.0

        return spike


# Create neuron
neuron = LIFNeuron(
    beta=0.9,
    threshold=1.0
)


# Constant input
inputs = [0.2] * 50


voltages = []
spikes = []


# Simulate
for input_current in inputs:

    spike = neuron.step(input_current)

    voltages.append(neuron.v)
    spikes.append(spike)


# Plot membrane potential
plt.figure(figsize=(10, 4))

plt.plot(
    voltages,
    label="Membrane potential"
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Threshold"
)

plt.xlabel("Time")
plt.ylabel("Membrane potential")

plt.title("LIF Neuron")

plt.legend()

plt.show()