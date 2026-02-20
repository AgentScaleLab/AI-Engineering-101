"""
Benchmark script to measure efficiency improvements of the optimized ResNet18 training framework
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def create_simulated_dataset(dataset_size=1000, num_classes=1000):
    """Create a simulated dataset for benchmarking"""
    images = torch.randn(dataset_size, 3, 224, 224)
    targets = torch.randint(0, num_classes, (dataset_size,))
    return TensorDataset(images, targets)


def baseline_train_step(model, data, target, optimizer, criterion, device):
    """Baseline training step without optimizations"""
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def optimized_train_step(model, data, target, optimizer, criterion, device, use_compile=False):
    """Optimized training step with torch.compile"""
    model.train()
    optimizer.zero_grad()

    if use_compile:
        # Apply torch.compile for optimization
        compiled_model = torch.compile(model, mode='reduce-overhead')
        output = compiled_model(data)
    else:
        output = model(data)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def benchmark_baseline(batch_size=64, num_batches=50):
    """Benchmark baseline training without optimizations"""
    print("Benchmarking baseline implementation...")

    # Setup - force CPU usage due to CUDA compatibility issues with older GPU
    device = torch.device('cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Create small dataset for benchmarking
    dataset = create_simulated_dataset(batch_size * num_batches, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Warmup
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:  # 5 warmup batches
            break
        data, target = data.to(device), target.to(device)
        baseline_train_step(model, data, target, optimizer, criterion, device)

    # Actual benchmark
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()

    batch_times = []
    for i, (data, target) in enumerate(dataloader):
        if i < 5:  # Skip warmup batches
            continue

        data, target = data.to(device), target.to(device)
        batch_start = time.time()
        baseline_train_step(model, data, target, optimizer, criterion, device)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if i >= num_batches + 4:  # Stop after benchmark batches
            break

    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times)
    throughput = batch_size / avg_batch_time

    print(f"Baseline Results:")
    print(f"  Total time: {total_time:.4f}s for {num_batches} batches")
    print(f"  Average batch time: {avg_batch_time:.6f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Memory usage: N/A (CPU mode)")
    print()

    return avg_batch_time, throughput


def benchmark_optimized(batch_size=64, num_batches=50, use_amp=True, use_compile=True):
    """Benchmark optimized training with various techniques"""
    print("Benchmarking optimized implementation...")

    # Setup - force CPU usage due to CUDA compatibility issues with older GPU
    device = torch.device('cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model = model.to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Create small dataset for benchmarking
    dataset = create_simulated_dataset(batch_size * num_batches, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Warmup
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:  # 5 warmup batches
            break
        data, target = data.to(device), target.to(device)
        optimized_train_step(model, data, target, optimizer, criterion, device, use_compile)

    # Actual benchmark
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()

    batch_times = []
    for i, (data, target) in enumerate(dataloader):
        if i < 5:  # Skip warmup batches
            continue

        data, target = data.to(device), target.to(device)
        batch_start = time.time()
        optimized_train_step(model, data, target, optimizer, criterion, device, use_compile)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if i >= num_batches + 4:  # Stop after benchmark batches
            break

    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times)
    throughput = batch_size / avg_batch_time

    print(f"Optimized Results (Compile: {use_compile}):")
    print(f"  Total time: {total_time:.4f}s for {num_batches} batches")
    print(f"  Average batch time: {avg_batch_time:.6f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Memory usage: N/A (CPU mode)")
    print()

    return avg_batch_time, throughput


def main():
    """Main benchmarking function"""
    print("=" * 70)
    print("EFFICIENCY BENCHMARK: ResNet18 Training Framework")
    print("=" * 70)

    # Run benchmarks with smaller parameters for faster execution
    baseline_batch_time, baseline_throughput = benchmark_baseline(batch_size=32, num_batches=20)

    # Test torch.compile optimization
    opt1_batch_time, opt1_throughput = benchmark_optimized(
        batch_size=32, num_batches=20, use_amp=False, use_compile=True
    )

    opt2_batch_time, opt2_throughput = benchmark_optimized(
        batch_size=32, num_batches=20, use_amp=False, use_compile=False
    )

    # Compare larger batch size for optimized version
    opt4_batch_time, opt4_throughput = benchmark_optimized(
        batch_size=64, num_batches=20, use_amp=False, use_compile=True
    )

    print("=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print("Configuration                    | Batch Time | Throughput  | Speedup")
    print("---------------------------------------------------------------------")
    print(f"Baseline (No Optimizations)      | {baseline_batch_time:.6f}s | {baseline_throughput:8.2f} |  1.00x")
    print(f"Compile Only                     | {opt1_batch_time:.6f}s | {opt1_throughput:8.2f} | {baseline_throughput/opt1_throughput:.2f}x")
    print(f"Large Batch (64) + Compile       | {opt4_batch_time:.6f}s | {opt4_throughput:8.2f} | {baseline_throughput/opt4_throughput:.2f}x")
    print()
    print("Speedup calculated relative to baseline implementation")
    print()
    print("OPTIMIZATION IMPROVEMENTS:")
    print(f"- Torch.compile: {(baseline_throughput/opt1_throughput):.2f}x speedup")
    print(f"- Large Batch + Compile: {(baseline_throughput/opt4_throughput):.2f}x speedup")
    print()
    print("Note: AMP (Automatic Mixed Precision) not available on CPU")
    print("=" * 70)


if __name__ == "__main__":
    main()