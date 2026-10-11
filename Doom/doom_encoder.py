import torch
import torchvision.transforms as transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TIME_STEPS = 20
IMAGE_SIZE = 84


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# Spike encoder
# --------------------------------------------------

def rate_encode(image, time_steps):
    """
    Convert an image with values [0,1]
    into a spike train.

    Higher pixel intensity → higher firing probability.

    Returns:
        [time_steps, channels, height, width]
    """

    spikes = torch.rand(
        time_steps,
        *image.shape,
        device=image.device
    )

    spikes = spikes < image

    return spikes.float()


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    # Fake image for now
    image = torch.rand(3, 240, 320)

    image = transform(image)

    print("Processed image:")
    print("Shape :", image.shape)
    print("Dtype :", image.dtype)
    print("Min   :", image.min().item())
    print("Max   :", image.max().item())

    spikes = rate_encode(
        image,
        TIME_STEPS
    )

    print("\nSpike train:")
    print("Shape :", spikes.shape)
    print("Dtype :", spikes.dtype)
    print("Device:", spikes.device)

    print(
        "Spike rate:",
        spikes.mean().item()
    )