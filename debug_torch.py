import sys
import os

print("Python executable:", sys.executable)
print("Python path:", sys.path)
print()

try:
    import torch
    print("PyTorch imported successfully")
    print("PyTorch version:", torch.__version__)
    print("PyTorch file location:", torch.__file__)
    print("CUDA available:", torch.cuda.is_available())
    
    if hasattr(torch.version, 'cuda'):
        print("CUDA version:", torch.version.cuda)
    else:
        print("No CUDA version info")
        
except ImportError as e:
    print("Failed to import PyTorch:", e)

print("\nEnvironment variables:")
for key in ['CUDA_HOME', 'CUDA_PATH', 'PATH']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")