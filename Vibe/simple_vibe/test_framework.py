"""
Test the complete optimized ResNet18 training framework
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os

# Add the training framework to the path
import sys
sys.path.append('.')

from training_framework.models.resnet18 import get_resnet18
from training_framework.utils.training_utils import (
    get_device, create_dataloader, calculate_accuracy, format_time, print_model_info
)
from training_framework.optimizers.optimizer_factory import OptimizerFactory, OptimizedTrainer
from training_framework.config.config import Config


def create_simulated_dataset(dataset_size=1000, num_classes=10):
    """Create a simulated dataset for testing"""
    images = torch.randn(dataset_size, 3, 224, 224)
    targets = torch.randint(0, num_classes, (dataset_size,))
    return TensorDataset(images, targets)


def test_optimized_framework():
    print("="*60)
    print("TESTING COMPLETE OPTIMIZED RESNET18 TRAINING FRAMEWORK")
    print("="*60)

    # Force CPU usage for compatibility
    Config.DEVICE = torch.device('cpu')
    Config.BATCH_SIZE = 32  # Smaller batch for CPU
    Config.NUM_WORKERS = 0  # Disable multiprocessing for CPU
    Config.PIN_MEMORY = False
    Config.USE_AMP = False  # Disable AMP on CPU
    Config.EPOCHS = 2  # Fewer epochs for testing

    print(f"Using device: {Config.DEVICE}")
    print(f"Configuration: {Config.BATCH_SIZE} batch size, {Config.EPOCHS} epochs")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_simulated_dataset(dataset_size=500, num_classes=Config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=100, num_classes=Config.NUM_CLASSES)

    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    # Initialize model
    print("\nInitializing ResNet18 model...")
    model = get_resnet18(num_classes=Config.NUM_CLASSES, pretrained=False)
    model = model.to(Config.DEVICE)

    # Print model information
    print_model_info(model)

    # Apply torch.compile if enabled
    if Config.USE_TORCH_COMPILE:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=Config.COMPILE_MODE)
        print(f"Model compiled with mode: {Config.COMPILE_MODE}")

    # Initialize optimizer and scheduler
    optimizer = OptimizerFactory.create_optimizer(
        model.parameters(),
        optimizer_type=Config.OPTIMIZER,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        momentum=Config.MOMENTUM
    )

    scheduler = OptimizerFactory.create_scheduler(
        optimizer,
        scheduler_type=Config.SCHEDULER
    )

    # Initialize loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initialize optimized trainer
    trainer = OptimizedTrainer(
        model,
        Config.DEVICE,
        use_amp=Config.USE_AMP,
        grad_clip_val=Config.GRAD_CLIP_VAL
    )

    print(f"\nStarting training with {Config.EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        # Training phase
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        progress_bar = tqdm(enumerate(train_loader),
                           total=len(train_loader),
                           desc=f'Train Epoch {epoch+1}')

        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

            # Perform training step
            loss_val = trainer.train_step(data, target, optimizer, criterion)

            # Calculate accuracy
            with torch.no_grad():
                output = model(data)
                acc = calculate_accuracy(output, target)

            running_loss += loss_val
            running_acc += acc

            if batch_idx % 10 == 0:  # Update every 10 batches
                progress_bar.set_postfix({
                    'Loss': f'{loss_val:.4f}',
                    'Acc': f'{acc:.2f}%'
                })

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

                loss_val, output = trainer.eval_step(data, target, criterion)
                acc = calculate_accuracy(output, target)

                val_running_loss += loss_val
                val_running_acc += acc

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_acc = val_running_acc / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

        # Step scheduler
        scheduler.step()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(total_time)}")

    print("\n" + "="*60)
    print("FRAMEWORK OPTIMIZATIONS APPLIED:")
    print("="*60)
    print(f"- Optimized Data Loading: {Config.NUM_WORKERS} workers, pin_memory={Config.PIN_MEMORY}")
    print(f"- Gradient Clipping: {'Enabled (value: ' + str(Config.GRAD_CLIP_VAL) + ')' if Config.GRAD_CLIP_VAL else 'Disabled'}")
    print(f"- Label Smoothing: Enabled (0.1)")
    print(f"- Optimized Memory Usage: persistent_workers={Config.PERSISTENT_WORKERS}")
    print(f"- Advanced Optimizers: {Config.OPTIMIZER}")
    print(f"- Learning Rate Scheduling: {Config.SCHEDULER}")
    if Config.USE_TORCH_COMPILE:
        print(f"- Torch.compile: Enabled (mode: {Config.COMPILE_MODE})")

    print(f"\nTEST PLATFORM INFORMATION:")
    print(f"- Device: {Config.DEVICE}")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CPU cores: {os.cpu_count()}")
    print(f"- Total training time: {format_time(total_time)}")

    print("="*60)


if __name__ == "__main__":
    test_optimized_framework()