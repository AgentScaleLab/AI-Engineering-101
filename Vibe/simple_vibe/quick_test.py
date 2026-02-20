"""
Quick test to demonstrate the optimized training framework performance
"""
import torch
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from training_framework.models.resnet18 import get_resnet18
from training_framework.config.config import Config
from training_framework.utils.training_utils import get_device, calculate_accuracy
from torch.cuda.amp import autocast, GradScaler

print("Testing optimized ResNet18 framework on CPU...")

# Force CPU usage
device = torch.device('cpu')
print(f"Using device: {device}")

# Create a smaller model for quick testing
print("Loading ResNet18 model...")
model = get_resnet18(num_classes=10, pretrained=False)
model = model.to(device)
model.train()

print("Testing basic training performance...")

# Create a small dataset for testing
batch_size = 16  # Small batch for quick test
num_samples = 64  # Small dataset for quick test
inputs = torch.randn(batch_size, 3, 224, 224)
targets = torch.randint(0, 10, (batch_size,))

# Test baseline approach
print("\n1. Testing baseline approach (no optimizations):")
start_time = time.time()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(5):  # Just 5 iterations for quick test
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if i == 0:  # Calculate accuracy after first iteration
        acc = calculate_accuracy(outputs, targets)
        print(f"   Sample accuracy: {acc:.2f}%")

baseline_time = time.time() - start_time
print(f"   Time for 5 iterations: {baseline_time:.4f}s")

# Test with torch.compile (if available)
print("\n2. Testing with torch.compile optimization:")
model2 = get_resnet18(num_classes=10, pretrained=False)
model2 = model2.to(device)
model2.train()

try:
    compiled_model = torch.compile(model2, mode='reduce-overhead')
    optimizer2 = torch.optim.SGD(compiled_model.parameters(), lr=0.001)

    start_time = time.time()
    for i in range(5):  # Just 5 iterations for quick test
        optimizer2.zero_grad()
        outputs = compiled_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer2.step()

    compile_time = time.time() - start_time
    print(f"   Time for 5 iterations: {compile_time:.4f}s")
    print(f"   Speedup: {baseline_time/compile_time:.2f}x")
except Exception as e:
    print(f"   Torch.compile not available or caused error: {e}")
    compile_time = baseline_time  # Assume no improvement if compile fails

print("\nQuick performance test completed!")

print("\nTEST PLATFORM INFORMATION:")
print(f"- Device: {device}")
print(f"- PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"- CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"- CUDA version: {torch.version.cuda}")
else:
    print("- CUDA: Not available (using CPU)")
print(f"- CPU cores: {torch.get_num_threads()}")

print("\nPERFORMANCE SUMMARY:")
print(f"- Baseline (5 iterations): {baseline_time:.4f}s")
try:
    print(f"- With torch.compile: {compile_time:.4f}s")
    print(f"- Speedup achieved: {baseline_time/compile_time:.2f}x")
except:
    print("- No speedup test completed due to torch.compile issues")