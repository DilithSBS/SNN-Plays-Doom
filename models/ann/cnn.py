import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) architecture for processing 
    84x84 grayscale frames. It maps the visual state to action values (Q-values).
    """
    def __init__(self, input_channels, num_actions):
        super(SimpleCNN, self).__init__()
        
        # Input shape: (Batch, 1, 84, 84)
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        
        # After conv layers, spatial dimensions are 9x9 with 32 channels (32 * 9 * 9 = 2592)
        self.fc_net = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Tensor of shape (Batch, Channels, Height, Width)
        Returns:
            Tensor of shape (Batch, num_actions) representing action Q-values.
        """
        conv_out = self.conv_net(x)
        # Flatten the spatial dimensions for the fully connected layers
        flattened = conv_out.view(conv_out.size(0), -1)
        action_values = self.fc_net(flattened)
        return action_values

if __name__ == "__main__":
    # Quick test to verify shapes
    dummy_input = torch.zeros(1, 1, 84, 84)  # 1 batch, 1 channel, 84x84
    model = SimpleCNN(input_channels=1, num_actions=3)
    
    output = model(dummy_input)
    print("Model initialized successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 3)
