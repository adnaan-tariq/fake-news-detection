import torch

print("=== GPU Availability Check ===")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    # Test tensor creation on GPU
    try:
        test_tensor = torch.randn(3, 3).cuda()
        print("✓ GPU tensor creation successful")
        print(f"Test tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"✗ GPU tensor creation failed: {e}")
else:
    print("No CUDA-capable GPU found")
    print("Training will use CPU")

print(f"PyTorch Version: {torch.__version__}")
print(f"Device being used: {'cuda' if torch.cuda.is_available() else 'cpu'}")