#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the Multi-scale Physics-Informed Neural Network (PINN) for flood modeling.

This script loads ANUGA simulation data, creates a multi-scale PINN model,
and trains it to predict flood dynamics. The model integrates neural operators
with physics-informed constraints from the shallow water equations.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MultiScalePINN model and training functions
from flood_warning_system.models.multi_scale_pinn import (
    MultiScalePINN, 
    train_multi_scale_pinn, 
    plot_training_history
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flood_warning_system/logs/multi_scale_pinn_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FloodDataset(Dataset):
    """
    Dataset for flood simulation data.
    
    This dataset loads pre-processed ANUGA simulation data and prepares it
    for training the multi-scale PINN model.
    """
    def __init__(
        self,
        data_path,
        input_vars=["stage", "xmomentum", "ymomentum"],
        target_vars=["stage", "xmomentum", "ymomentum"],
        sequence_length=10,
        predict_steps=1,
        elevation_var="elevation",
        transform=None,
        normalize=True,
        train=True,
        val_split=0.2
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the netCDF file containing simulation data
            input_vars: List of input variable names
            target_vars: List of target variable names
            sequence_length: Number of time steps to use as context
            predict_steps: Number of time steps to predict
            elevation_var: Name of the elevation variable
            transform: Optional transform to apply to the data
            normalize: Whether to normalize the data
            train: Whether this is a training or validation dataset
            val_split: Fraction of data to use for validation
        """
        self.data_path = data_path
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        self.elevation_var = elevation_var
        self.transform = transform
        self.normalize = normalize
        self.train = train
        self.val_split = val_split
        
        # Initialize normalization parameters
        self.norm_params = {}
        
        # Load the data
        self.load_data()
        
        # Split into train and validation sets
        self.create_time_sequences()
    
    def load_data(self):
        """Load the data from the netCDF file."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load the data
            self.ds = xr.open_dataset(self.data_path)
            
            # Extract the variables
            self.data = {}
            
            # Check if all required variables exist
            missing_inputs = [var for var in self.input_vars if var not in self.ds]
            missing_targets = [var for var in self.target_vars if var not in self.ds]
            
            if missing_inputs or missing_targets:
                logger.warning(f"Missing input variables: {missing_inputs}")
                logger.warning(f"Missing target variables: {missing_targets}")
                
                # If critical variables are missing, raise an error
                critical_vars = ["stage"]  # Variables that must exist
                critical_missing = [var for var in critical_vars if var in missing_inputs + missing_targets]
                
                if critical_missing:
                    raise ValueError(f"Critical variables missing: {critical_missing}")
            
            # Load input and target variables
            all_vars = set(self.input_vars + self.target_vars)
            for var in all_vars:
                if var in self.ds:
                    # Convert to torch tensor
                    self.data[var] = torch.tensor(self.ds[var].values, dtype=torch.float32)
                else:
                    # Use zeros for missing variables
                    shape = self.ds[self.input_vars[0]].shape if var in self.input_vars else self.ds[self.target_vars[0]].shape
                    logger.warning(f"Variable {var} not found in dataset, using zeros")
                    self.data[var] = torch.zeros(shape, dtype=torch.float32)
            
            # Load elevation if available
            if self.elevation_var in self.ds:
                self.data[self.elevation_var] = torch.tensor(
                    self.ds[self.elevation_var].values, dtype=torch.float32
                )
                # Add a time dimension if it doesn't exist
                if len(self.data[self.elevation_var].shape) == 2:
                    self.data[self.elevation_var] = self.data[self.elevation_var].unsqueeze(0)
            
            # Normalize data if requested
            if self.normalize:
                self.normalize_data()
            
            # Get the number of time steps
            self.times = self.ds.time.values
            self.n_times = len(self.times)
            
            logger.info(f"Loaded data with {self.n_times} time steps")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def normalize_data(self):
        """Normalize the data using mean and standard deviation."""
        logger.info("Normalizing data")
        
        # Normalize each variable separately
        for var in self.data:
            if var == self.elevation_var:
                # Don't normalize elevation
                continue
            
            # Compute mean and std over all dims except channels
            dim_indices = tuple(range(1, len(self.data[var].shape)))  # All dims except the first
            mean = self.data[var].mean(dim=dim_indices, keepdim=True)
            std = self.data[var].std(dim=dim_indices, keepdim=True)
            
            # Avoid division by zero
            std = torch.where(std > 1e-5, std, torch.ones_like(std))
            
            # Store normalization parameters
            self.norm_params[var] = {'mean': mean, 'std': std}
            
            # Normalize
            self.data[var] = (self.data[var] - mean) / std
            
            logger.info(f"Normalized {var}: mean={mean.item():.4f}, std={std.item():.4f}")
    
    def denormalize(self, data, var):
        """Denormalize data for a specific variable."""
        if var in self.norm_params:
            mean = self.norm_params[var]['mean']
            std = self.norm_params[var]['std']
            return data * std + mean
        return data
    
    def create_time_sequences(self):
        """Create sequences of time steps for training."""
        # Determine valid indices
        valid_indices = range(self.sequence_length, self.n_times - self.predict_steps)
        n_valid = len(valid_indices)
        
        # Split into train and validation
        split_idx = int(n_valid * (1 - self.val_split))
        
        if self.train:
            self.indices = valid_indices[:split_idx]
        else:
            self.indices = valid_indices[split_idx:]
        
        logger.info(f"Created {'training' if self.train else 'validation'} "
                   f"dataset with {len(self.indices)} samples")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - inputs: Tensor of shape [sequence_length, channels, height, width]
                - targets: Tensor of shape [predict_steps, channels, height, width]
                - elevation: Tensor of shape [1, height, width]
        """
        # Get the time index
        t_idx = self.indices[idx]
        
        # Create input sequence
        input_sequence = []
        for t in range(t_idx - self.sequence_length, t_idx):
            # Stack input variables along channel dimension
            features = []
            for var in self.input_vars:
                features.append(self.data[var][t])
            
            input_frame = torch.stack(features, dim=0)
            input_sequence.append(input_frame)
        
        # Stack along time dimension
        inputs = torch.stack(input_sequence, dim=0)
        
        # Create target sequence
        target_sequence = []
        for t in range(t_idx, t_idx + self.predict_steps):
            # Stack target variables along channel dimension
            features = []
            for var in self.target_vars:
                features.append(self.data[var][t])
            
            target_frame = torch.stack(features, dim=0)
            target_sequence.append(target_frame)
        
        # Stack along time dimension
        targets = torch.stack(target_sequence, dim=0)
        
        # Get elevation if available
        elevation = None
        if self.elevation_var in self.data:
            elevation = self.data[self.elevation_var][0]  # Assuming elevation is constant in time
        
        # Apply transform if provided
        if self.transform:
            inputs, targets, elevation = self.transform(inputs, targets, elevation)
        
        # Prepare the sample
        sample = {
            "inputs": inputs[-1],  # Use only the last time step for PINN
            "targets": targets[0],  # Use only the first prediction step
            "sequence_inputs": inputs,  # Full sequence for recurrent models
            "sequence_targets": targets  # Full sequence for recurrent models
        }
        
        if elevation is not None:
            sample["elevation"] = elevation
        
        return sample


def normalize_data(data, mean=None, std=None):
    """
    Normalize the data using mean and standard deviation.
    
    Args:
        data: Data to normalize
        mean: Mean of the data (if None, computed from data)
        std: Standard deviation of the data (if None, computed from data)
        
    Returns:
        Normalized data, mean, std
    """
    if mean is None:
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
    
    if std is None:
        std = data.std(dim=(0, 2, 3), keepdim=True)
    
    # Avoid division by zero
    std = torch.where(std > 1e-5, std, torch.ones_like(std))
    
    # Normalize
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std


def create_dataloaders(
    data_path,
    batch_size=16,
    sequence_length=10,
    predict_steps=1,
    val_split=0.2,
    num_workers=4
):
    """
    Create training and validation dataloaders.
    
    Args:
        data_path: Path to the netCDF file containing simulation data
        batch_size: Batch size
        sequence_length: Number of time steps to use as context
        predict_steps: Number of time steps to predict
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, data_info
    """
    # Create datasets
    train_dataset = FloodDataset(
        data_path=data_path,
        input_vars=["stage", "xmomentum", "ymomentum"],
        target_vars=["stage", "xmomentum", "ymomentum"],
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        elevation_var="elevation",
        train=True,
        val_split=val_split
    )
    
    val_dataset = FloodDataset(
        data_path=data_path,
        input_vars=["stage", "xmomentum", "ymomentum"],
        target_vars=["stage", "xmomentum", "ymomentum"],
        sequence_length=sequence_length,
        predict_steps=predict_steps,
        elevation_var="elevation",
        train=False,
        val_split=val_split
    )
    
    # Get sample to determine data dimensions
    sample = train_dataset[0]
    data_info = {
        "input_shape": sample["inputs"].shape,
        "target_shape": sample["targets"].shape,
        "has_elevation": "elevation" in sample
    }
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, data_info


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(config, data_info):
    """
    Create a multi-scale PINN model based on the configuration.
    
    Args:
        config: Configuration dictionary
        data_info: Information about the data
        
    Returns:
        MultiScalePINN model
    """
    # Get model parameters from config
    model_params = config.get("model", {})
    physics_params = config.get("physics", {})
    
    # Extract input/output channels from data
    input_channels = data_info["input_shape"][0]
    output_channels = data_info["target_shape"][0]
    grid_size = data_info["input_shape"][1:]
    
    # Create the model with all physics parameters
    model = MultiScalePINN(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=model_params.get("hidden_channels", 64),
        fno_modes=model_params.get("fno_modes", (12, 12)),
        num_scales=model_params.get("num_scales", 3),
        use_gnn=model_params.get("use_gnn", True),
        grid_size=grid_size,
        dx=model_params.get("dx", 1.0),
        dy=model_params.get("dy", 1.0),
        dt=model_params.get("dt", 0.1),
        gravity=model_params.get("gravity", 9.81),
        manning_coef=model_params.get("manning_coef", 0.035),
        physics_weight=physics_params.get("initial_weight", 0.1),
        dropout=model_params.get("dropout", 0.0),
        # Physics parameters
        adaptive_weighting=physics_params.get("adaptive_weighting", True),
        min_physics_weight=physics_params.get("min_weight", 0.01),
        max_physics_weight=physics_params.get("max_weight", 1.0),
        adaptation_rate=physics_params.get("adaptation_rate", 0.05),
        continuity_weight=physics_params.get("continuity_weight", 1.0),
        x_momentum_weight=physics_params.get("x_momentum_weight", 0.5),
        y_momentum_weight=physics_params.get("y_momentum_weight", 0.5),
        boundary_weight=physics_params.get("boundary_weight", 0.5),
        enforce_positivity=physics_params.get("enforce_positivity", True)
    )
    
    return model


def create_optimizer(config, model):
    """
    Create an optimizer based on the configuration.
    
    Args:
        config: Configuration dictionary
        model: Model to optimize
        
    Returns:
        Optimizer
    """
    # Get optimizer parameters from config
    optimizer_params = config.get("optimizer", {})
    optimizer_type = optimizer_params.get("type", "adam").lower()
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 0.001),
            weight_decay=optimizer_params.get("weight_decay", 0.0)
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_params.get("learning_rate", 0.01),
            momentum=optimizer_params.get("momentum", 0.9),
            weight_decay=optimizer_params.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(config, optimizer):
    """
    Create a learning rate scheduler based on the configuration.
    
    Args:
        config: Configuration dictionary
        optimizer: Optimizer
        
    Returns:
        Scheduler or None
    """
    # Get scheduler parameters from config
    scheduler_params = config.get("scheduler", {})
    
    if not scheduler_params or not scheduler_params.get("use_scheduler", False):
        return None
    
    scheduler_type = scheduler_params.get("type", "step").lower()
    
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1)
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("t_max", 100)
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=scheduler_params.get("factor", 0.1),
            patience=scheduler_params.get("patience", 10),
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a multi-scale PINN model for flood modeling"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="flood_warning_system/config/multi_scale_pinn_config.yaml",
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="flood_warning_system/data/processed/simulation_output.nc",
        help="Path to the simulation data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flood_warning_system/models/saved",
        help="Directory to save model checkpoints and logs"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)"
    )
    
    parser.add_argument(
        "--log_physics",
        action="store_true",
        help="Whether to log detailed physics loss components"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Configure logging level based on verbosity
    if args.log_physics:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create dataloaders
    train_loader, val_loader, data_info = create_dataloaders(
        data_path=args.data_path,
        batch_size=config.get("data", {}).get("batch_size", 16),
        sequence_length=config.get("data", {}).get("sequence_length", 10),
        predict_steps=config.get("data", {}).get("predict_steps", 1),
        val_split=config.get("data", {}).get("val_split", 0.2),
        num_workers=config.get("data", {}).get("num_workers", 4)
    )
    
    # Create model
    model = create_model(config, data_info)
    
    # Initialize physics loss module
    physics_config = config.get("physics", {})
    if hasattr(model, "physics_loss_module"):
        # Update physics loss module parameters from config
        model.physics_loss_module.physics_weight = physics_config.get("initial_weight", 0.1)
        model.physics_loss_module.adaptive_weighting = physics_config.get("adaptive_weighting", True)
        model.physics_loss_module.min_physics_weight = physics_config.get("min_weight", 0.01)
        model.physics_loss_module.max_physics_weight = physics_config.get("max_weight", 1.0)
        model.physics_loss_module.adaptation_rate = physics_config.get("adaptation_rate", 0.05)
        # Add equation-specific weights
        model.physics_loss_module.continuity_weight = physics_config.get("continuity_weight", 1.0)
        model.physics_loss_module.x_momentum_weight = physics_config.get("x_momentum_weight", 0.5)
        model.physics_loss_module.y_momentum_weight = physics_config.get("y_momentum_weight", 0.5)
        model.physics_loss_module.boundary_weight = physics_config.get("boundary_weight", 0.5)
        model.physics_loss_module.enforce_positivity = physics_config.get("enforce_positivity", True)
        
        logger.info("Initialized physics loss module with parameters:")
        logger.info(f"  - Initial weight: {model.physics_loss_module.physics_weight}")
        logger.info(f"  - Adaptive weighting: {model.physics_loss_module.adaptive_weighting}")
        logger.info(f"  - Min weight: {model.physics_loss_module.min_physics_weight}")
        logger.info(f"  - Max weight: {model.physics_loss_module.max_physics_weight}")
        logger.info(f"  - Adaptation rate: {model.physics_loss_module.adaptation_rate}")
        logger.info(f"  - Continuity weight: {model.physics_loss_module.continuity_weight}")
        logger.info(f"  - X-momentum weight: {model.physics_loss_module.x_momentum_weight}")
        logger.info(f"  - Y-momentum weight: {model.physics_loss_module.y_momentum_weight}")
        logger.info(f"  - Boundary weight: {model.physics_loss_module.boundary_weight}")
        logger.info(f"  - Enforce positivity: {model.physics_loss_module.enforce_positivity}")
    
    # Create optimizer
    optimizer = create_optimizer(config, model)
    
    # Create scheduler
    scheduler = create_scheduler(config, optimizer)
    
    # Get training parameters
    training_config = config.get("training", {})
    num_epochs = training_config.get("num_epochs", 100)
    save_interval = training_config.get("save_interval", 10)
    early_stopping_patience = training_config.get("early_stopping_patience", 15)
    gradient_clip_val = training_config.get("gradient_clip_val", 1.0)
    
    # Set up visualization parameters
    viz_config = config.get("visualization", {})
    plot_interval = viz_config.get("plot_interval", 10)
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract sample batch for visualization
    viz_batch = next(iter(val_loader))
    
    # Train the model with custom parameters
    history = train_multi_scale_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_dir=args.output_dir,
        save_interval=save_interval,
        device=args.device,
        early_stopping_patience=early_stopping_patience,
        gradient_clip_val=gradient_clip_val,
        log_physics=args.log_physics,
        plot_interval=plot_interval,
        viz_dir=viz_dir,
        viz_batch=viz_batch
    )
    
    # Plot and save training history
    plot_training_history(
        history=history,
        save_path=os.path.join(args.output_dir, "training_history.png"),
        show=False
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    model.save_model(save_dir=args.output_dir, filename="final_model.pt")
    
    # Visualize final physics losses
    if hasattr(model, "visualize_physics_losses"):
        logger.info("Generating final physics loss visualization")
        inputs = viz_batch["inputs"].to(args.device)
        targets = viz_batch["targets"].to(args.device)
        elevation = viz_batch.get("elevation", None)
        if elevation is not None:
            elevation = elevation.to(args.device)
        
        model.visualize_physics_losses(
            inputs=inputs[:1],  # Use just the first sample
            targets=targets[:1],
            elevation=elevation[:1] if elevation is not None else None,
            save_path=os.path.join(args.output_dir, "final_physics_losses.png"),
            show=False
        )
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")


def train_multi_scale_pinn(
    model,
    train_loader,
    val_loader=None,
    optimizer=None,
    scheduler=None,
    num_epochs=100,
    save_dir="models/saved",
    save_interval=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience=15,
    gradient_clip_val=1.0,
    log_physics=False,
    plot_interval=10,
    viz_dir=None,
    viz_batch=None
):
    """
    Train a multi-scale PINN model.
    
    Args:
        model: MultiScalePINN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        save_dir: Directory to save model checkpoints
        save_interval: Interval at which to save checkpoints
        device: Device to use for training
        early_stopping_patience: Number of epochs to wait before early stopping
        gradient_clip_val: Value to clip gradients to
        log_physics: Whether to log detailed physics loss components
        plot_interval: Interval at which to plot training history
        viz_dir: Directory to save visualization plots
        viz_batch: Batch of data for visualization
        
    Returns:
        Training history
    """
    # Set up training parameters
    device = torch.device(device)
    model.to(device)

    if __name__ == "__main__":
        main() 