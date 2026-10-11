import torch
import torch.nn as nn

from models import SNN
from temporal_dataset import create_dataset


# ============================================================
# Configuration
# ============================================================

TIME_STEPS = 20

INPUT_SIZE = 10

HIDDEN_SIZE = 32

OUTPUT_SIZE = 2

NUM_SAMPLES = 1000

EPOCHS = 30

LEARNING_RATE = 1e-3


# ============================================================
# Dataset
# ============================================================

X, y = create_dataset(
    num_samples=NUM_SAMPLES,
    time_steps=TIME_STEPS,
    input_size=INPUT_SIZE
)


# ============================================================
# Model
# ============================================================

model = SNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    optimizer.zero_grad()


    # Forward pass
    spike_output = model(X)

    print(
        "Spike output requires grad:",
        spike_output.requires_grad
    )

    print(
        "Spike output grad fn:",
        spike_output.grad_fn
    )

    # Sum spikes over time
    output = spike_output.sum(dim=0)

    output = spike_output.sum(dim=0)

    print(
        "Output requires grad:",
        output.requires_grad
    )

    print(
        "Output grad fn:",
        output.grad_fn
    )

    loss = criterion(output, y)

    print(
        "Loss requires grad:",
        loss.requires_grad
    )

    print(
        "Loss grad fn:",
        loss.grad_fn
    )

    # Backpropagation
    loss.backward()


    # Update weights
    optimizer.step()


    # Calculate accuracy
    predictions = output.argmax(dim=1)

    accuracy = (
        predictions == y
    ).float().mean()


    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {loss.item():.4f} | "
        f"Accuracy: {accuracy.item() * 100:.2f}%"
    )