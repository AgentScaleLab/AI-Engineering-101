"""
Optimizer factory with various optimization techniques
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler


class OptimizerFactory:
    """
    Factory class to create various optimizers with different optimization techniques
    """

    @staticmethod
    def create_optimizer(model_params, optimizer_type='adam', **kwargs):
        """
        Create an optimizer based on the specified type
        """
        lr = kwargs.get('lr', 0.001)

        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                model_params,
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 1e-4),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 1e-2),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                model_params,
                lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 1e-4),
                nesterov=kwargs.get('nesterov', True)
            )
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(
                model_params,
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 1e-4),
                momentum=kwargs.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    @staticmethod
    def create_scheduler(optimizer, scheduler_type='step', **kwargs):
        """
        Create a learning rate scheduler based on the specified type
        """
        if scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 200),
                eta_min=kwargs.get('eta_min', 0)
            )
        elif scheduler_type.lower() == 'multi_step':
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [30, 80]),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10),
                verbose=kwargs.get('verbose', True)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class OptimizedTrainer:
    """
    Optimized trainer with mixed precision and gradient clipping
    """

    def __init__(self, model, device, use_amp=True, grad_clip_val=None):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.grad_clip_val = grad_clip_val

        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def train_step(self, data, target, optimizer, criterion):
        """
        Single training step with optional mixed precision and gradient clipping
        """
        self.model.train()
        optimizer.zero_grad()

        if self.use_amp and self.device.type == 'cuda':
            # Mixed precision training
            with autocast():
                output = self.model(data)
                loss = criterion(output, target)

            # Scale the loss and perform backpropagation
            self.scaler.scale(loss).backward()

            if self.grad_clip_val is not None:
                # Unscale the gradients before clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)

            # Step with scaled optimizer
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard training without mixed precision
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            if self.grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)

            optimizer.step()

        return loss.item()

    def eval_step(self, data, target, criterion):
        """
        Single evaluation step (no gradients)
        """
        self.model.eval()
        with torch.no_grad():
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
            else:
                output = self.model(data)
                loss = criterion(output, target)

        return loss.item(), output