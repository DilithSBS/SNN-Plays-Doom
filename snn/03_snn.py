import torch
import torch.nn as nn

from spike_function import SpikeFunction

# ============================================================
# LIF Layer
# ============================================================

class LIFLayer(nn.Module):

    def __init__(
        self,
        beta=0.9,
        threshold=1.0
    ):
        super().__init__()

        self.beta = beta
        self.threshold = threshold


    def forward(
        self,
        input_current,
        membrane
    ):

        # Leak
        membrane = self.beta * membrane

        # Integrate
        membrane = membrane + input_current

        # Generate spike
        spike = SpikeFunction.apply(
            membrane,
            torch.tensor(
                self.threshold,
                device=membrane.device
            )
        )

        # Reset
        membrane = membrane * (1 - spike)

        return spike, membrane


# ============================================================
# SNN
# ============================================================

class SNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.lif1 = LIFLayer()

        self.fc2 = nn.Linear(
            hidden_size,
            output_size
        )

        self.lif2 = LIFLayer()


    def forward(self, x):

        # x:
        # [time, batch, input_size]

        batch_size = x.shape[1]

        device = x.device


        # Initial membrane potentials
        membrane1 = torch.zeros(
            batch_size,
            self.fc1.out_features,
            device=device
        )

        membrane2 = torch.zeros(
            batch_size,
            self.fc2.out_features,
            device=device
        )


        outputs = []


        # Process through time
        for t in range(x.shape[0]):

            # First layer
            current1 = self.fc1(x[t])

            spike1, membrane1 = self.lif1(
                current1,
                membrane1
            )


            # Second layer
            current2 = self.fc2(spike1)

            spike2, membrane2 = self.lif2(
                current2,
                membrane2
            )


            # Save spikes
            outputs.append(spike2)


        return torch.stack(outputs)


# ============================================================
# Test
# ============================================================

T = 20

batch_size = 4

input_size = 10

hidden_size = 32

output_size = 2


model = SNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)


# Generate random input
x = torch.rand(
    T,
    batch_size,
    input_size
)


# Run SNN
output = model(x)


print("Input shape :", x.shape)
print("Output shape:", output.shape)


loss = output.sum()

loss.backward()

print("Gradients calculated successfully!")
print(model.fc1.weight.grad)