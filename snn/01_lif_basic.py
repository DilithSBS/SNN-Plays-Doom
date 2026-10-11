class LIFNeuron:
    def __init__(self, beta=0.9, threshold=1.0):
        self.beta = beta
        self.threshold = threshold
        self.v = 0.0

    def step(self, input_current):
        # 1. Leak
        self.v = self.beta * self.v

        # 2. Integrate input
        self.v += input_current

        # 3. Generate spike
        spike = 1.0 if self.v >= self.threshold else 0.0

        # 4. Reset after spike
        if spike:
            self.v = 0.0

        return spike


# Create neuron
neuron = LIFNeuron(
    beta=0.9,
    threshold=1.0
)


# Input current over time
inputs = [
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
]


# Simulate
for t, input_current in enumerate(inputs):

    spike = neuron.step(input_current)

    print(
        f"t={t}, "
        f"input={input_current:.2f}, "
        f"voltage={neuron.v:.2f}, "
        f"spike={spike}"
    )