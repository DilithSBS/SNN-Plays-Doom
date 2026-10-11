import matplotlib
matplotlib.use('WebAgg')
from lif import LIFNeuron
import matplotlib.pyplot as plt

neuron = LIFNeuron(beta=0.9, threshold=1.0)

inputs = [0.2] * 50

voltages = []
spikes = []

for input_current in inputs:

    spike = neuron.step(input_current)
    potential = neuron.v

    print(f" Input current {input_current}, Output Spike {spike}, Potential {potential}")

    voltages.append(potential)
    spikes.append(spike)

plt.plot(voltages)
plt.axhline(1.0, linestyle="--", color="black")


plt.plot(spikes)
plt.ylabel("Membrane potential")
plt.show()