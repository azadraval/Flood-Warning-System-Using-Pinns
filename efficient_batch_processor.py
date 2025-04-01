#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Efficient batch processor for PINN model training.

This module provides optimization techniques for training data pipelines, including:
- Prefetching and caching
- GPU data pinning
- Mixed precision training
- Gradient accumulation for larger effective batch sizes
- Distributed data parallelism for multi-GPU training
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flood_warning_system/logs/efficient_batch_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PrefetchLoader:
    """
    DataLoader wrapper that prefetches data to GPU.
    
    This helps overlap data transfer with computation to improve throughput.
    """
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
    
    def __iter__(self):
        loader_iter = iter(self.data_loader)
        self.preload(loader_iter)
        batch = self.next_batch
        
        while batch is not None:
            yield batch
            batch = self.preload(loader_iter)
    
    def preload(self, loader_iter):
        try:
            self.next_batch = next(loader_iter)
        except StopIteration:
            self.next_batch = None
            return None
        
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                self.next_batch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in self.next_batch.items()
                }
        
        return self.next_batch
    
    def __len__(self):
        return len(self.data_loader)


class MixedPrecisionTrainer:
    """
    Trainer with mixed precision support for faster training.
    
    Uses PyTorch's automatic mixed precision (AMP) to speed up training
    while maintaining numerical stability.
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_mixed_precision=True,
        gradient_accumulation_steps=1,
        clip_grad_norm=1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda' and torch.cuda.is_available()
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training")
        else:
            logger.info("Using full precision training")
        
        self.model.to(self.device)
    
    def train_step(self, batch, step_idx):
        """
        Perform a single training step with mixed precision and gradient accumulation.
        
        Args:
            batch: Dictionary containing training data
            step_idx: Current step index within the epoch
            
        Returns:
            Dictionary of loss values
        """
        # Set model to training mode
        self.model.train()
        
        # Determine if this step requires a gradient update
        update_gradients = (step_idx + 1) % self.gradient_accumulation_steps == 0
        
        # Compute forward pass with autocast for mixed precision
        with autocast(enabled=self.use_mixed_precision):
            result = self.model.training_step(batch)
            loss = result["loss"]
            
            # Scale the loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient accumulation
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            
            if update_gradients:
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            
            if update_gradients:
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Update the learning rate scheduler if provided and it's time to update
        if self.scheduler is not None and update_gradients:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = result.get("val_loss", result["loss"])
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        
        # Scale the loss back for reporting
        result["loss_step"] = result["loss"].item()  # Original loss for logging
        result["loss"] = loss.item() * self.gradient_accumulation_steps  # Scaled loss for gradient accumulation
        
        return result
    
    def validation_step(self, batch):
        """
        Perform a validation step.
        
        Args:
            batch: Dictionary containing validation data
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # No gradients needed for validation
        with torch.no_grad():
            # Use autocast for mixed precision
            with autocast(enabled=self.use_mixed_precision):
                result = self.model.validation_step(batch)
        
        return result
    
    def setup_distributed(self, rank, world_size):
        """
        Set up distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            
        Returns:
            None
        """
        if world_size > 1:
            # Set environment variables
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # Initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            
            # Wrap model in DDP
            self.model = DDP(self.model, device_ids=[rank])
            
            logger.info(f"Initialized distributed training with rank {rank}/{world_size}")


def create_efficient_dataloader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    prefetch_to_device=None,
    distributed=False,
    rank=0,
    world_size=1
):
    """
    Create an efficient DataLoader with optimizations for training.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_to_device: Device to prefetch data to (None to disable)
        distributed: Whether to use distributed sampling
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        Optimized DataLoader
    """
    # Set up sampler for distributed training
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create basic dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    # Wrap with prefetching if requested
    if prefetch_to_device is not None:
        dataloader = PrefetchLoader(dataloader, prefetch_to_device)
    
    return dataloader


def train_epoch_with_efficiency(
    trainer,
    train_loader,
    val_loader=None,
    epoch=0,
    log_interval=10
):
    """
    Train for one epoch with efficiency optimizations.
    
    Args:
        trainer: MixedPrecisionTrainer instance
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        epoch: Current epoch number
        log_interval: Interval for logging progress
        
    Returns:
        Dictionary of epoch metrics
    """
    # Initialize metrics
    train_losses = []
    data_losses = []
    physics_losses = []
    
    # Train loop
    for step_idx, batch in enumerate(train_loader):
        # Training step
        result = trainer.train_step(batch, step_idx)
        
        # Collect metrics
        train_losses.append(result["loss_step"])
        data_losses.append(result["data_loss"].item())
        physics_losses.append(result["physics_loss"].item())
        
        # Log progress
        if step_idx % log_interval == 0:
            logger.info(
                f"Epoch {epoch}, Step {step_idx}/{len(train_loader)}: "
                f"Loss={result['loss_step']:.6f}, "
                f"Data Loss={result['data_loss'].item():.6f}, "
                f"Physics Loss={result['physics_loss'].item():.6f}, "
                f"Physics Weight={result['physics_weight']:.4f}"
            )
    
    # Compute average metrics
    avg_train_loss = np.mean(train_losses)
    avg_data_loss = np.mean(data_losses)
    avg_physics_loss = np.mean(physics_losses)
    
    # Log epoch summary
    logger.info(
        f"Epoch {epoch} completed: "
        f"Avg Train Loss={avg_train_loss:.6f}, "
        f"Avg Data Loss={avg_data_loss:.6f}, "
        f"Avg Physics Loss={avg_physics_loss:.6f}"
    )
    
    # Initialize validation metrics
    val_metrics = {
        "val_loss": None,
        "val_data_loss": None,
        "val_physics_loss": None
    }
    
    # Run validation if loader is provided
    if val_loader is not None:
        val_losses = []
        val_data_losses = []
        val_physics_losses = []
        
        for batch in val_loader:
            # Validation step
            result = trainer.validation_step(batch)
            
            # Collect metrics
            val_losses.append(result["loss"].item())
            val_data_losses.append(result["data_loss"].item())
            val_physics_losses.append(result["physics_loss"].item())
        
        # Compute average metrics
        val_metrics["val_loss"] = np.mean(val_losses)
        val_metrics["val_data_loss"] = np.mean(val_data_losses)
        val_metrics["val_physics_loss"] = np.mean(val_physics_losses)
        
        # Log validation summary
        logger.info(
            f"Validation Epoch {epoch}: "
            f"Val Loss={val_metrics['val_loss']:.6f}, "
            f"Val Data Loss={val_metrics['val_data_loss']:.6f}, "
            f"Val Physics Loss={val_metrics['val_physics_loss']:.6f}"
        )
    
    # Combine metrics
    metrics = {
        "train_loss": avg_train_loss,
        "train_data_loss": avg_data_loss,
        "train_physics_loss": avg_physics_loss,
        **val_metrics
    }
    
    return metrics


def train_distributed(
    rank,
    world_size,
    model,
    train_dataset,
    val_dataset=None,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    use_mixed_precision=True,
    gradient_accumulation_steps=1,
    log_interval=10
):
    """
    Train a model with distributed data parallelism.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size per GPU
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        log_interval: Interval for logging progress
        
    Returns:
        Trained model and training history
    """
    # Set device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Set up distributed training
    trainer.setup_distributed(rank, world_size)
    
    # Create dataloaders
    train_loader = create_efficient_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_to_device=device,
        distributed=True,
        rank=rank,
        world_size=world_size
    )
    
    if val_dataset is not None:
        val_loader = create_efficient_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_to_device=device,
            distributed=True,
            rank=rank,
            world_size=world_size
        )
    else:
        val_loader = None
    
    # Training history
    history = {
        "train_loss": [],
        "train_data_loss": [],
        "train_physics_loss": [],
        "val_loss": [],
        "val_data_loss": [],
        "val_physics_loss": []
    }
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        metrics = train_epoch_with_efficiency(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            log_interval=log_interval
        )
        
        # Update history
        for key, value in metrics.items():
            if value is not None and key in history:
                history[key].append(value)
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()
    
    return model, history


if __name__ == "__main__":
    # This module is intended to be imported, not run directly
    logger.info("This module provides efficiency optimizations for PINN model training.")
    logger.info("Import the functions and classes from this module in your training script.") 