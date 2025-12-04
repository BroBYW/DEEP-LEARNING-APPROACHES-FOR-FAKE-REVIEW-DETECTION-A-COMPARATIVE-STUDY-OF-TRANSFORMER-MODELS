import torch

# Check CUDA version supported by PyTorch
print(f"PyTorch CUDA version: {torch.version.cuda}")

# Check GPU availability
print(f"Is CUDA available: {torch.cuda.is_available()}")

# Check GPU device details
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")

