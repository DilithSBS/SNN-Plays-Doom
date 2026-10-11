import matplotlib
matplotlib.use('WebAgg')

import matplotlib.pyplot as plt

from temporal_dataset import create_dataset


X, y = create_dataset(
    num_samples=10,
    time_steps=20,
    input_size=10
)


sample = 0

spikes = X[:, sample, :]


plt.figure(figsize=(10, 5))

plt.imshow(
    spikes.T,
    aspect="auto",
    interpolation="nearest"
)

plt.xlabel("Time")
plt.ylabel("Input neuron")

plt.title(
    f"Sample {sample} - Class {y[sample].item()}"
)

plt.show()