"""
Configuration file for the ResNet18 training framework
"""
import torch


class Config:
    # Model configuration
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 1000  # Adjust based on your dataset
    PRETRAINED = False  # Set to True to use pretrained weights

    # Training configuration
    EPOCHS = 90
    BATCH_SIZE = 256  # Larger batch size for better GPU utilization
    LEARNING_RATE = 0.1  # Learning rate for SGD
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # Optimizer configuration
    OPTIMIZER = 'sgd'  # 'sgd', 'adam', 'adamw'
    SCHEDULER = 'cosine'  # 'step', 'cosine', 'multi_step', 'exponential', 'reduce_on_plateau'

    # Advanced optimization settings
    USE_AMP = True  # Use Automatic Mixed Precision
    GRAD_CLIP_VAL = 1.0  # Gradient clipping value (set to None to disable)
    USE_TORCH_COMPILE = True  # Use torch.compile for optimization
    COMPILE_MODE = 'reduce-overhead'  # 'reduce-overhead', 'max-autotune', 'default'

    # Data loading configuration
    NUM_WORKERS = 8  # Number of data loading workers
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging and checkpointing
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'

    # Performance monitoring
    ENABLE_PROFILER = False  # Enable PyTorch profiler for performance analysis
    PROFILER_DIR = './profiles'

    # Optimized settings for different hardware
    @classmethod
    def set_gpu_optimized_settings(cls):
        """Set optimized settings for GPU training"""
        cls.BATCH_SIZE = 256
        cls.NUM_WORKERS = 8
        cls.USE_AMP = True
        cls.PIN_MEMORY = True
        cls.PERSISTENT_WORKERS = True

    @classmethod
    def set_cpu_optimized_settings(cls):
        """Set optimized settings for CPU training"""
        cls.BATCH_SIZE = 64
        cls.NUM_WORKERS = 4
        cls.USE_AMP = False  # AMP only works on CUDA
        cls.PIN_MEMORY = False
        cls.PERSISTENT_WORKERS = False

    @classmethod
    def set_memory_efficient_settings(cls):
        """Set settings for memory-constrained environments"""
        cls.BATCH_SIZE = 32
        cls.NUM_WORKERS = 2
        cls.USE_AMP = True
        cls.PIN_MEMORY = False
        cls.PERSISTENT_WORKERS = False

    # Initialize settings based on available hardware
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

# Function to update config based on command line arguments or other settings
def update_config(**kwargs):
    """Update configuration with provided keyword arguments"""
    for key, value in kwargs.items():
        if hasattr(Config, key.upper()):
            setattr(Config, key.upper(), value)
        elif hasattr(Config, key):
            setattr(Config, key, value)
        else:
            print(f"Warning: Unknown configuration key '{key}'")

# Set default settings based on hardware availability after class definition
if torch.cuda.is_available():
    Config.set_gpu_optimized_settings()
else:
    Config.set_cpu_optimized_settings()