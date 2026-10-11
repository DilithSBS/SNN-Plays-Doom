import torch
from vizdoom import DoomGame


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("Device          :", device)

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Start ViZDoom
# --------------------------------------------------

game = DoomGame()

game.load_config("basic.cfg")
game.init()

print("\nViZDoom started!")

game.new_episode()


# --------------------------------------------------
# 3. Get a frame
# --------------------------------------------------

state = game.get_state()

frame = state.screen_buffer

print("\nViZDoom frame")
print("Shape :", frame.shape)
print("Dtype :", frame.dtype)
print("Min   :", frame.min())
print("Max   :", frame.max())


# --------------------------------------------------
# 4. Convert NumPy → PyTorch
# --------------------------------------------------

frame_tensor = torch.from_numpy(frame)

print("\nPyTorch tensor")
print("Shape :", frame_tensor.shape)
print("Dtype :", frame_tensor.dtype)
print("Device:", frame_tensor.device)


# --------------------------------------------------
# 5. Move tensor to GPU
# --------------------------------------------------

frame_gpu = frame_tensor.to(device)

print("\nCUDA tensor")
print("Shape :", frame_gpu.shape)
print("Dtype :", frame_gpu.dtype)
print("Device:", frame_gpu.device)


# --------------------------------------------------
# 6. Do a small GPU operation
# --------------------------------------------------

frame_gpu = frame_gpu.float() / 255.0

print("\nNormalized tensor")
print("Shape :", frame_gpu.shape)
print("Dtype :", frame_gpu.dtype)
print("Device:", frame_gpu.device)
print("Min   :", frame_gpu.min().item())
print("Max   :", frame_gpu.max().item())


# --------------------------------------------------
# 7. Close ViZDoom
# --------------------------------------------------

game.close()

print("\nViZDoom closed.")