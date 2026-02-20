"""
Efficient ResNet18 Training Framework with Performance Optimizations
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import components from our framework
from training_framework.models.resnet18 import get_resnet18
from training_framework.utils.training_utils import (
    get_device, create_dataloader, calculate_accuracy, format_time, print_model_info
)
from training_framework.optimizers.optimizer_factory import OptimizerFactory, OptimizedTrainer
from training_framework.config.config import Config


class EfficientResNet18Trainer:
    def __init__(self, config=None):
        self.config = config or Config
        self.device = self.config.DEVICE

        # Initialize model
        self.model = get_resnet18(
            num_classes=self.config.NUM_CLASSES,
            pretrained=self.config.PRETRAINED
        ).to(self.device)

        # Apply torch.compile if enabled
        if self.config.USE_TORCH_COMPILE and torch.cuda.is_available():
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode=self.config.COMPILE_MODE)
            print(f"Model compiled with mode: {self.config.COMPILE_MODE}")

        # Print model information
        print_model_info(self.model)

        # Initialize optimizer and scheduler
        self.optimizer = OptimizerFactory.create_optimizer(
            self.model.parameters(),
            optimizer_type=self.config.OPTIMIZER,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            momentum=self.config.MOMENTUM
        )

        self.scheduler = OptimizerFactory.create_scheduler(
            self.optimizer,
            scheduler_type=self.config.SCHEDULER
        )

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

        # Initialize optimized trainer
        self.trainer = OptimizedTrainer(
            self.model,
            self.device,
            use_amp=self.config.USE_AMP,
            grad_clip_val=self.config.GRAD_CLIP_VAL
        )

        # Initialize logging
        self.writer = SummaryWriter(log_dir=self.config.LOG_DIR)

        # Initialize metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Performance tracking
        self.batch_times = []
        self.epoch_times = []

    def train_epoch(self, train_loader, epoch_num):
        """Train for a single epoch"""
        start_time = time.time()
        self.model.train()

        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(enumerate(train_loader),
                           total=num_batches,
                           desc=f'Epoch {epoch_num+1}/{self.config.EPOCHS}')

        for batch_idx, (data, target) in progress_bar:
            # Move data to device
            data, target = data.to(self.device, non_blocking=self.config.PIN_MEMORY), \
                          target.to(self.device, non_blocking=self.config.PIN_MEMORY)

            # Record batch start time for performance monitoring
            batch_start_time = time.time()

            # Perform training step
            loss_val = self.trainer.train_step(data, target, self.optimizer, self.criterion)

            # Calculate accuracy
            with torch.no_grad():
                output = self.model(data)
                acc = calculate_accuracy(output, target)

            # Track metrics
            running_loss += loss_val
            running_acc += acc

            # Calculate elapsed batch time
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Acc': f'{acc:.2f}%',
                'Avg Time': f'{np.mean(self.batch_times[-10:]):.3f}s'
            })

            # Log to tensorboard periodically
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = epoch_num * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss_val, global_step)
                self.writer.add_scalar('Train/Accuracy', acc, global_step)
                self.writer.add_scalar('Train/Learning_Rate',
                                      self.optimizer.param_groups[0]['lr'],
                                      global_step)

        epoch_time = time.time() - start_time
        self.epoch_times.append(epoch_time)

        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

        return avg_loss, avg_acc, epoch_time

    def validate(self, val_loader, epoch_num):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device, non_blocking=self.config.PIN_MEMORY), \
                              target.to(self.device, non_blocking=self.config.PIN_MEMORY)

                # Perform evaluation step
                loss_val, output = self.trainer.eval_step(data, target, self.criterion)
                acc = calculate_accuracy(output, target)

                running_loss += loss_val
                running_acc += acc

        avg_loss = running_loss / num_batches
        avg_acc = running_acc / num_batches

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)

        # Log validation metrics
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch_num)
        self.writer.add_scalar('Validation/Accuracy', avg_acc, epoch_num)

        return avg_loss, avg_acc

    def train(self, train_loader, val_loader=None):
        """Main training loop with all optimizations"""
        print(f"Starting training with {self.config.EPOCHS} epochs")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Optimizer: {self.config.OPTIMIZER}")
        print(f"Scheduler: {self.config.SCHEDULER}")
        print(f"Using AMP: {self.config.USE_AMP}")
        print(f"Using torch.compile: {self.config.USE_TORCH_COMPILE}")
        print("-" * 50)

        start_time = time.time()

        for epoch in range(self.config.EPOCHS):
            # Train for one epoch
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader, epoch)

            # Validate if validation loader is provided
            if val_loader:
                val_loss, val_acc = self.validate(val_loader, epoch)
                print(f'Epoch {epoch+1}/{self.config.EPOCHS}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                      f'Time: {format_time(epoch_time)}')
            else:
                print(f'Epoch {epoch+1}/{self.config.EPOCHS}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Time: {format_time(epoch_time)}')

            # Step scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loader:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

            # Save checkpoint periodically
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(epoch)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")

        # Calculate and print performance metrics
        self.print_performance_metrics(total_time)

        # Close tensorboard writer
        self.writer.close()

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'{self.config.MODEL_NAME}_epoch_{epoch+1}.pth'
        )

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def print_performance_metrics(self, total_time):
        """Print performance metrics and efficiency improvements"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS & EFFICIENCY IMPROVEMENTS")
        print("="*60)

        # Calculate average batch time
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            print(f"Average batch processing time: {avg_batch_time:.4f}s")

        # Calculate average epoch time
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            print(f"Average epoch time: {format_time(avg_epoch_time)}")

        # Calculate throughput
        if self.batch_times and self.config.BATCH_SIZE:
            avg_samples_per_second = self.config.BATCH_SIZE / np.mean(self.batch_times)
            print(f"Average samples/second: {avg_samples_per_second:.2f}")

        # Estimate FLOPs utilization if possible
        print(f"\nOPTIMIZATION TECHNIQUES APPLIED:")
        print(f"- Automatic Mixed Precision (AMP): {'Enabled' if self.config.USE_AMP else 'Disabled'}")
        print(f"- Torch.compile: {'Enabled' if self.config.USE_TORCH_COMPILE else 'Disabled'}")
        print(f"- Gradient Clipping: {'Enabled (value: ' + str(self.config.GRAD_CLIP_VAL) + ')' if self.config.GRAD_CLIP_VAL else 'Disabled'}")
        print(f"- Optimized Data Loading: {self.config.NUM_WORKERS} workers, pin_memory={self.config.PIN_MEMORY}")
        print(f"- Label Smoothing: Enabled (0.1)")
        print(f"- Optimized Memory Usage: persistent_workers={self.config.PERSISTENT_WORKERS}")

        print(f"\nPERFORMANCE IMPROVEMENTS:")
        print(f"- Larger batch sizes for better GPU utilization")
        print(f"- Mixed precision training (up to 3x speedup, reduced memory)")
        print(f"- Optimized data loading with multiple workers and pinned memory")
        print(f"- Torch.compile for kernel optimization")
        print(f"- Efficient gradient computation and clipping")
        print("="*60)


def create_simulated_dataset(dataset_size=10000, num_classes=1000):
    """
    Create a simulated dataset for testing the training framework
    """
    # Create random images with shape (dataset_size, 3, 224, 224) - typical for ResNet18
    images = torch.randn(dataset_size, 3, 224, 224)
    # Create random targets with shape (dataset_size,) for classification
    targets = torch.randint(0, num_classes, (dataset_size,))

    from torch.utils.data import TensorDataset
    dataset = TensorDataset(images, targets)

    return dataset


def main():
    """Main function to run the training framework"""
    print("Initializing Efficient ResNet18 Training Framework")

    # Create simulated datasets for demonstration
    print("Creating simulated datasets...")
    train_dataset = create_simulated_dataset(dataset_size=50000, num_classes=Config.NUM_CLASSES)
    val_dataset = create_simulated_dataset(dataset_size=10000, num_classes=Config.NUM_CLASSES)

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

    # Initialize trainer
    trainer = EfficientResNet18Trainer()

    # Run training
    trainer.train(train_loader, val_loader)

    print("Training framework completed successfully!")


if __name__ == "__main__":
    main()