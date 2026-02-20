"""
Example script to run the optimized ResNet18 training framework
"""
import torch
import os
from training_framework.train import EfficientResNet18Trainer, create_simulated_dataset
from training_framework.utils.training_utils import create_dataloader
from training_framework.config.config import Config, update_config


def run_default_training():
    """Run training with default optimized configuration"""
    print("Running training with default optimized configuration...")

    # Create simulated datasets
    print("Creating simulated datasets...")
    train_dataset = create_simulated_dataset(dataset_size=10000, num_classes=Config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=2000, num_classes=Config.NUM_CLASSES)

    # Create optimized data loaders
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

    # Initialize and run trainer
    trainer = EfficientResNet18Trainer()
    trainer.train(train_loader, val_loader)


def run_memory_efficient_training():
    """Run training with memory-efficient configuration"""
    print("Running memory-efficient training configuration...")

    # Update config for memory efficiency
    Config.set_memory_efficient_settings()
    Config.EPOCHS = 5  # Fewer epochs for demo

    # Create smaller datasets for memory-constrained demo
    train_dataset = create_simulated_dataset(dataset_size=2000, num_classes=Config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=500, num_classes=Config.NUM_CLASSES)

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

    trainer = EfficientResNet18Trainer()
    trainer.train(train_loader, val_loader)


def run_gpu_optimized_training():
    """Run training with GPU-optimized configuration"""
    print("Running GPU-optimized training configuration...")

    # Ensure we're using GPU-optimized settings
    Config.set_gpu_optimized_settings()
    Config.BATCH_SIZE = 256  # Larger batch size for better GPU utilization
    Config.EPOCHS = 10  # More epochs for thorough training

    # Create datasets
    train_dataset = create_simulated_dataset(dataset_size=20000, num_classes=Config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=4000, num_classes=Config.NUM_CLASSES)

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

    trainer = EfficientResNet18Trainer()
    trainer.train(train_loader, val_loader)


def run_custom_config_training():
    """Run training with custom configuration"""
    print("Running training with custom configuration...")

    # Create custom config
    custom_config = type('CustomConfig', (), {})()
    custom_config.MODEL_NAME = 'resnet18'
    custom_config.NUM_CLASSES = 10
    custom_config.PRETRAINED = False
    custom_config.EPOCHS = 5
    custom_config.BATCH_SIZE = 128
    custom_config.LEARNING_RATE = 0.01
    custom_config.OPTIMIZER = 'adam'
    custom_config.SCHEDULER = 'cosine'
    custom_config.USE_AMP = True
    custom_config.GRAD_CLIP_VAL = 1.0
    custom_config.USE_TORCH_COMPILE = True
    custom_config.COMPILE_MODE = 'reduce-overhead'
    custom_config.NUM_WORKERS = 4
    custom_config.PIN_MEMORY = True
    custom_config.PERSISTENT_WORKERS = True
    custom_config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    custom_config.LOG_INTERVAL = 10
    custom_config.SAVE_CHECKPOINT_INTERVAL = 2
    custom_config.CHECKPOINT_DIR = './checkpoints'
    custom_config.LOG_DIR = './logs'
    custom_config.ENABLE_PROFILER = False

    train_dataset = create_simulated_dataset(dataset_size=5000, num_classes=custom_config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=1000, num_classes=custom_config.NUM_CLASSES)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=custom_config.BATCH_SIZE,
        shuffle=True,
        num_workers=custom_config.NUM_WORKERS,
        pin_memory=custom_config.PIN_MEMORY
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=custom_config.BATCH_SIZE,
        shuffle=False,
        num_workers=custom_config.NUM_WORKERS,
        pin_memory=custom_config.PIN_MEMORY
    )

    trainer = EfficientResNet18Trainer(config=custom_config)
    trainer.train(train_loader, val_loader)


def main():
    """Main function to demonstrate different training options"""
    print("ResNet18 Efficient Training Framework - Examples")
    print("="*50)
    print("Available training modes:")
    print("1. Default optimized configuration")
    print("2. Memory-efficient configuration")
    print("3. GPU-optimized configuration")
    print("4. Custom configuration")
    print("5. Run all configurations")
    print()

    choice = input("Select option (1-5, default=1): ").strip()

    if choice == "1":
        run_default_training()
    elif choice == "2":
        run_memory_efficient_training()
    elif choice == "3":
        run_gpu_optimized_training()
    elif choice == "4":
        run_custom_config_training()
    elif choice == "5" or choice == "":
        print("Running all configurations sequentially...\n")
        run_default_training()
        print("\n" + "="*50 + "\n")
        run_memory_efficient_training()
        print("\n" + "="*50 + "\n")
        run_gpu_optimized_training()
        print("\n" + "="*50 + "\n")
        run_custom_config_training()
    else:
        print("Invalid choice. Running default configuration...")
        run_default_training()

    print("\nTraining examples completed!")


if __name__ == "__main__":
    main()