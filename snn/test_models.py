import torch
import matplotlib
matplotlib.use("WebAgg")

from models import SNN, SpikeFunction

T = 20
batch_size = 4
input_size = 10
hidden_size = 32
output_size = 2


model = SNN(
    input_size,
    hidden_size,
    output_size
)

# Random input tensor
x = torch.rand(
    T,                  # number of time steps
    batch_size,         # number of samples
    input_size          # number of input features
)


output = model(x)

print(output.shape)




spike = SpikeFunction.apply(
    membrane,
    self.threshold
)