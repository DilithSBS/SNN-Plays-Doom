import torch
import torch.nn as nn

from spike_function import SpikeFunction


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

        # Integrate input
        membrane = membrane + input_current

        # Generate spike
        spike = SpikeFunction.apply(
            membrane,
            self.threshold
        )

        # Reset membrane
        membrane = membrane * (1 - spike)

        return spike, membrane



class SNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
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

        batch_size = x.shape[1]

        device = x.device

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

        for t in range(x.shape[0]):
            
            current1 = self.fc1(x[t])

            spike1, membrane1 = self.lif1 (current1, membrane1)

            current2 = self.fc2(spike1)

            spike2, membrane2 = self.lif2 (current2, membrane2)

            outputs.append(spike2)

        return torch.stack(outputs)




