import torch
import os

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")

print("Checking PyTorch...")
try:
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch Device Count: {torch.cuda.device_count()}")
        print(f"Torch Device Name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Torch Error: {e}")

print("\nChecking Numba...")
try:
    from numba import cuda
    print(f"Numba CUDA is available: {cuda.is_available()}")
    if cuda.is_available():
        print(f"Numba Device: {cuda.get_current_device()}")
    else:
        print("Numba CUDA is NOT available.")
except Exception as e:
    print(f"Numba Error: {e}")
