import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = "data"
    dem_path: Optional[str] = None
    rainfall_path: Optional[str] = None
    river_network_path: Optional[str] = None
    soil_properties_path: Optional[str] = None
    land_use_path: Optional[str] = None
    historical_floods_path: Optional[str] = None
    
    # Data preprocessing parameters
    spatial_resolution: float = 30.0  # meters
    temporal_resolution: float = 1.0  # hours
    dem_fill_method: str = "fill_depressions"
    rainfall_interpolation_method: str = "idw"
    
    # Training/validation/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_methods: List[str] = None

@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Network."""
    # Network architecture
    hidden_layers: List[int] = None
    activation: str = "tanh"
    initialization: str = "xavier_normal"
    dropout_rate: float = 0.0
    
    # Physics constraints
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    bc_weight: float = 1.0
    data_weight: float = 1.0
    
    # PDE coefficients (for shallow water equations)
    gravity: float = 9.81  # m/s^2
    manning_coefficient: float = 0.035  # Default Manning's n
    theta: float = 0.5  # Implicit factor (0.5 = Crank-Nicolson)
    
    # Geometry adaptation
    use_geometry_adaptation: bool = True
    geometry_mapping_method: str = "conformal"
    boundary_encoding_method: str = "distance_field"
    
    # Fourier feature embedding
    use_fourier_features: bool = True
    fourier_sigma: float = 1.0
    num_fourier_features: int = 256

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Basic training parameters
    num_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    
    # Optimization
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    distributed: bool = False
    precision: str = "fp32"
    
    # Checkpointing
    save_dir: str = "models/checkpoints"
    save_frequency: int = 10
    
    # Early stopping
    patience: int = 50
    monitor: str = "val_loss"
    
    # Sequence-to-sequence training
    sequence_length: int = 24  # hours
    prediction_horizon: int = 72  # hours
    teacher_forcing_ratio: float = 0.5
    
    # Self-supervised pre-training
    use_pretraining: bool = True
    pretraining_epochs: int = 100

@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    checkpoint_path: str = "models/best_model.pth"
    batch_size: int = 64
    threshold: float = 0.5  # Threshold for flood classification
    uncertainty_estimation: bool = True
    ensemble_size: int = 5
    
    # Temporal marching parameters for long-term forecasting
    time_step: float = 1.0  # hours
    forecast_horizon: int = 120  # hours
    
    # Spatial adaptivity
    adaptive_resolution: bool = True
    refinement_threshold: float = 0.1
    
    # Post-processing
    smooth_output: bool = True
    smooth_kernel_size: int = 3

@dataclass
class APIConfig:
    """Configuration for API service."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    timeout: int = 300
    
    # Authentication
    use_auth: bool = True
    auth_method: str = "api_key"
    
    # Rate limiting
    rate_limit: int = 100  # requests per minute
    
    # Caching
    use_cache: bool = True
    cache_ttl: int = 3600  # seconds

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "flood-warning-system"
    wandb_entity: Optional[str] = None

@dataclass
class SystemConfig:
    """Overall system configuration."""
    data: DataConfig = DataConfig()
    pinn: PINNConfig = PINNConfig(hidden_layers=[128, 128, 128, 128])
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # System-wide settings
    seed: int = 42
    debug: bool = False
    version: str = "0.1.0"
    
    # Deployment
    deployment_type: str = "local"  # ["local", "docker", "kubernetes"]
    
    # Self-healing parameters
    error_tolerance: float = 0.1
    auto_recovery: bool = True
    monitoring_frequency: int = 60  # seconds
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from file."""
        import yaml
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

# Default configuration
default_config = SystemConfig() 