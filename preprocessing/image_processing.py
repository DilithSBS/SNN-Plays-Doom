import cv2
import numpy as np
import torch

def preprocess_frame(frame, target_size=(84, 84)):
    """
    Preprocesses a raw ViZDoom frame for neural network input.
    
    Args:
        frame: numpy array of shape (C, H, W) e.g., (3, 240, 320)
        target_size: tuple (Height, Width) for resizing
        
    Returns:
        numpy array of shape (1, target_size[0], target_size[1]) normalized to [0, 1]
    """
    if frame is None:
        return None
        
    # ViZDoom returns (Channels, Height, Width). OpenCV expects (Height, Width, Channels)
    # We transpose to (240, 320, 3)
    frame = np.transpose(frame, (1, 2, 0))
    
    # Convert to grayscale
    # ViZDoom RGB frames are sometimes returned in BGR format depending on settings,
    # but grayscale conversion largely handles intensity the same way.
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
    # Resize to reduce input dimensionality
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Add channel dimension back for PyTorch (C, H, W) -> (1, 84, 84)
    frame = np.expand_dims(frame, axis=0)
    
    return frame

if __name__ == "__main__":
    # Test the preprocessor with a dummy frame
    dummy_frame = np.random.randint(0, 256, (3, 240, 320), dtype=np.uint8)
    processed = preprocess_frame(dummy_frame)
    
    print(f"Original shape: {dummy_frame.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Max value: {processed.max():.2f}")
    print(f"Min value: {processed.min():.2f}")
