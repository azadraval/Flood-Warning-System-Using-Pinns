#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory-optimized training script for Physics-Informed Neural Networks on limited hardware.

Specifically optimized for NVIDIA 1650 Max-Q and similar GPUs with limited VRAM.
This script implements:
1. Memory monitoring and optimization
2. Enhanced gradient accumulation
3. Model checkpointing to reduce memory usage
4. Dynamic batch size adjustment
5. Mixed precision training customized for 16xx series GPUs
"""

import os
import sys
import time
import argparse
import logging
import json
import yaml
import numpy as np
import torch
import gc
from pathlib import Path
from functools import partial

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from flood_warning_system.train_multi_scale_pinn import (
    load_config,
    create_dataloaders,
    create_model,
    plot_training_history
)
from flood_warning_system.efficient_batch_processor import (
    MixedPrecisionTrainer,
    create_efficient_dataloader,
    train_epoch_with_efficiency
)
from flood_warning_system.optimize_hyperparameters import (
    create_search_space,
    objective_function,
    save_optimization_results,
    create_optimal_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flood_warning_system/logs/optimized_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import LAMB optimizer if available
try:
    from torch_optimizer import Lamb
    LAMB_AVAILABLE = True
except ImportError:
    LAMB_AVAILABLE = False
    logger.warning("LAMB optimizer not available. Install torch_optimizer for LAMB support.")


def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        A tuple of (total_memory, allocated_memory, free_memory, cached_memory) in MB
    """
    if not torch.cuda.is_available():
        return (0, 0, 0, 0)
    
    device = torch.cuda.current_device()
    
    try:
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
        allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        cached = torch.cuda.memory_reserved(device) / (1024**2)  # MB
        free = total_memory - allocated - cached
        
        return (total_memory, allocated, free, cached)
    except RuntimeError:
        return (0, 0, 0, 0)


def log_memory_usage(tag=""):
    """Log GPU memory usage at the current point in execution"""
    if torch.cuda.is_available():
        total, allocated, free, cached = get_gpu_memory_info()
        logger.info(f"[{tag}] GPU Memory: Total={total:.1f}MB, Used={allocated:.1f}MB, Free={free:.1f}MB, Cached={cached:.1f}MB")


def enable_checkpointing(model):
    """
    Enable gradient checkpointing to reduce memory usage.
    
    Args:
        model: PyTorch model to modify
    """
    # Check if the model has modules that support checkpointing
    checkpointing_enabled = False
    
    # Apply checkpointing to specific module types that support it
    for name, module in model.named_modules():
        # Apply to neural operator modules
        if "neural_operator" in name:
            if hasattr(module, "fno_components") and isinstance(module.fno_components, torch.nn.ModuleList):
                for fno in module.fno_components:
                    if hasattr(fno, "forward"):
                        fno.forward = torch.utils.checkpoint.checkpoint(fno.forward)
                        checkpointing_enabled = True
        
        # Apply to MultiScaleNeuralOperator fusion layer if it exists
        if "fusion_layer" in name and hasattr(module, "forward"):
            if isinstance(module, torch.nn.Sequential):
                # For sequential modules, apply checkpointing to the module
                orig_forward = module.forward
                module.forward = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                    orig_forward, *args, **kwargs)
                checkpointing_enabled = True
    
    if checkpointing_enabled:
        logger.info("Gradient checkpointing enabled to reduce memory usage")
    else:
        logger.warning("Could not enable checkpointing - model may not support it")
    
    return model


def optimize_memory():
    """Free unused memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def lock_gpu_clocks(min_freq=300, max_freq=1245):
    """
    Lock GPU clock speeds to prevent thermal throttling and crashes.
    This is especially important for NVIDIA 1650 Max-Q which can be unstable.
    
    Args:
        min_freq: Minimum clock frequency (MHz)
        max_freq: Maximum clock frequency (MHz)
    
    Returns:
        True if successful, False otherwise
    """
    import platform
    import subprocess
    import shutil
    
    if platform.system() != "Windows" and platform.system() != "Darwin":
        try:
            if shutil.which("nvidia-smi") is not None:
                # Enable persistent mode
                subprocess.run(["nvidia-smi", "-pm", "1"], check=False, capture_output=True)
                
                # Set clock speed range
                subprocess.run(
                    ["nvidia-smi", "-lgc", f"{min_freq},{max_freq}"], 
                    check=False, capture_output=True
                )
                
                logger.info(f"Locked GPU clock speeds to {min_freq}-{max_freq} MHz")
                return True
        except Exception as e:
            logger.warning(f"Could not lock GPU clock speeds: {str(e)}")
    
    return False


def reset_gpu_clocks():
    """Reset GPU clock speeds to default."""
    import platform
    import subprocess
    import shutil
    
    if platform.system() != "Windows" and platform.system() != "Darwin":
        try:
            if shutil.which("nvidia-smi") is not None:
                # Reset clocks to default
                subprocess.run(["nvidia-smi", "-rgc"], check=False, capture_output=True)
                logger.info("Reset GPU clock speeds to default")
                return True
        except Exception as e:
            logger.warning(f"Could not reset GPU clock speeds: {str(e)}")
    
    return False


def create_optimizer(config, model):
    """
    Create an optimizer with advanced options like AdamW and LAMB.
    
    Args:
        config: Configuration dictionary
        model: Model to optimize
        
    Returns:
        Optimizer instance
    """
    optimizer_params = config.get("optimizer", {})
    optimizer_type = optimizer_params.get("type", "adam").lower()
    lr = optimizer_params.get("learning_rate", 1e-3)
    weight_decay = optimizer_params.get("weight_decay", 1e-5)
    
    # Filter parameters that require weight decay (exclude bias and normalization parameters)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "norm" in name or "ln" in name or "batch_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # For limited memory GPUs, AdamW is usually more memory efficient than LAMB
    if optimizer_type == "adamw" or optimizer_type == "adam":
        logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8)
        )
    elif optimizer_type == "lamb" and LAMB_AVAILABLE:
        logger.info(f"Using LAMB optimizer with learning rate {lr} and weight decay {weight_decay}")
        optimizer = Lamb(
            param_groups,
            lr=lr,
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8)
        )
    elif optimizer_type == "sgd":
        logger.info(f"Using SGD optimizer with learning rate {lr} and weight decay {weight_decay}")
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=optimizer_params.get("momentum", 0.9),
            nesterov=optimizer_params.get("nesterov", True)
        )
    else:
        # Default to AdamW for limited memory GPUs
        logger.info(f"Using AdamW optimizer with learning rate {lr} and weight decay {weight_decay}")
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8)
        )
    
    return optimizer


def optimize_hyperparameters(args):
    """
    Run hyperparameter optimization using Bayesian optimization.
    Memory-optimized for limited VRAM.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the optimal configuration file
    """
    from skopt import gp_minimize
    from skopt.utils import use_named_args
    from functools import partial
    
    logger.info("Starting memory-efficient hyperparameter optimization")
    log_memory_usage("Before hyperparameter optimization")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    optimization_dir = os.path.join(args.output_dir, "optimization")
    os.makedirs(optimization_dir, exist_ok=True)
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Reduce model size for optimization phase to save memory
    base_config["model"]["hidden_channels"] = min(base_config["model"].get("hidden_channels", 64), 32)
    base_config["model"]["num_scales"] = min(base_config["model"].get("num_scales", 3), 2)
    base_config["data"]["batch_size"] = min(base_config["data"].get("batch_size", 16), 8)
    
    # Define search space - keep it smaller for limited VRAM
    space = create_search_space(limited_memory=True)
    
    # Create objective function with fixed parameters and memory optimization
    objective = partial(
        objective_function,
        base_config=base_config,
        data_path=args.data_path,
        output_dir=optimization_dir,
        device=args.device,
        limited_memory=True
    )
    
    # Wrap objective function to use named parameters and clear memory between runs
    @use_named_args(space)
    def objective_wrapper(**params):
        optimize_memory()
        result = objective(params)
        optimize_memory()
        return result
    
    # Run Bayesian optimization with fewer iterations for memory constraints
    logger.info(f"Running {min(args.n_calls, 10)} iterations of memory-efficient Bayesian optimization")
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=min(args.n_calls, 10),
        n_random_starts=min(args.n_random_starts, 3),
        random_state=42,
        verbose=True
    )
    
    # Save optimization results
    save_optimization_results(result, optimization_dir)
    
    # Get optimal parameters
    optimal_params = {dim.name: result.x[i] for i, dim in enumerate(space)}
    logger.info(f"Optimal parameters: {optimal_params}")
    
    # Create and save optimal configuration
    optimal_config = create_optimal_config(base_config, optimal_params, optimization_dir)
    optimal_config_path = os.path.join(optimization_dir, "optimal_config.yaml")
    
    logger.info(f"Memory-efficient optimization completed with best loss: {result.fun:.6f}")
    logger.info(f"Optimal configuration saved to {optimal_config_path}")
    log_memory_usage("After hyperparameter optimization")
    
    optimize_memory()
    return optimal_config_path


class MemoryEfficientTrainer(MixedPrecisionTrainer):
    """
    Enhanced trainer class with memory optimizations for limited hardware.
    """
    def __init__(self, *args, memory_efficient=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_efficient = memory_efficient
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # seconds
    
    def train_batch(self, batch):
        """Override train_batch to include memory optimizations"""
        # Periodically check memory usage
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            log_memory_usage("During training")
            self.last_memory_check = current_time
        
        result = super().train_batch(batch)
        
        # Clear unnecessary buffers occasionally
        if self.memory_efficient and np.random.random() < 0.05:  # 5% chance
            optimize_memory()
            
        return result


def find_optimal_batch_size(model, sample_input, max_batch_size=32, start_batch_size=4):
    """
    Find the optimal batch size that fits in GPU memory.
    
    Args:
        model: The model to test
        sample_input: A sample input tensor
        max_batch_size: Maximum batch size to try
        start_batch_size: Initial batch size to try
        
    Returns:
        The optimal batch size
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using default batch size")
        return start_batch_size
    
    logger.info("Finding optimal batch size for GPU memory...")
    model.eval()  # Set to eval mode
    
    # Start with the initial batch size
    batch_size = start_batch_size
    found_limit = False
    
    # Get a single sample to replicate
    if isinstance(sample_input, dict):
        # If input is a dictionary (like in some datasets), use the first tensor
        sample = next(iter(sample_input.values())).squeeze(0)
    else:
        sample = sample_input.squeeze(0)
    
    while batch_size <= max_batch_size and not found_limit:
        try:
            # Clear memory
            optimize_memory()
            
            # Create a batch by repeating the sample
            if isinstance(sample_input, dict):
                # Handle dictionary inputs
                batch = {k: v[0:1].repeat(batch_size, *([1] * (len(v.shape) - 1))) 
                         for k, v in sample_input.items()}
            else:
                # Handle tensor inputs
                batch = sample.unsqueeze(0).repeat(batch_size, *([1] * (len(sample.shape))))
            
            # Move to GPU
            if isinstance(batch, dict):
                batch = {k: v.cuda() for k, v in batch.items()}
            else:
                batch = batch.cuda()
            
            # Try a forward and backward pass
            with torch.cuda.amp.autocast(enabled=True):
                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    outputs = model(batch)
                
                # If output is a dict, use the first tensor for testing
                if isinstance(outputs, dict):
                    loss = sum([torch.mean(v) for v in outputs.values() 
                                if isinstance(v, torch.Tensor) and v.requires_grad])
                else:
                    loss = outputs.mean()
            
            loss.backward()
            
            # If we got here without OOM, increase batch size
            logger.info(f"Batch size {batch_size} fits in memory")
            
            # Increase batch size geometrically
            prev_batch_size = batch_size
            batch_size *= 2
            
            # Clean up
            del batch, loss, outputs
            model.zero_grad()
            optimize_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # We hit an OOM error, use the previous successful batch size
                found_limit = True
                batch_size = max(prev_batch_size, start_batch_size)
                logger.info(f"Memory limit reached. Setting batch size to {batch_size}")
            else:
                # If it's not an OOM error, re-raise
                raise
    
    if batch_size > max_batch_size:
        batch_size = max_batch_size
        logger.info(f"Capping batch size at maximum: {batch_size}")
    
    # Clean up
    model.zero_grad()
    optimize_memory()
    
    return batch_size


def train_single_gpu_memory_efficient(args, config):
    """
    Train model on a single GPU with memory optimization techniques.
    
    Args:
        args: Command-line arguments
        config: Model configuration
        
    Returns:
        Trained model and training history
    """
    logger.info("Starting memory-optimized single-GPU training")
    log_memory_usage("Before training")
    
    # Lock GPU clocks if enabled (for stability)
    if args.lock_gpu_clocks:
        lock_gpu_clocks()
    
    try:
        # Modify the config to use memory-efficient settings if needed
        if args.auto_config:
            logger.info("Automatically configuring for memory efficiency")
            # Adjust batch size based on available memory
            total_mem, _, _, _ = get_gpu_memory_info()
            
            # For GPUs with less than 6GB VRAM
            if total_mem < 6000:
                config["data"]["batch_size"] = min(config["data"].get("batch_size", 16), 8)
                config["model"]["hidden_channels"] = min(config["model"].get("hidden_channels", 64), 48)
                # Increase gradient accumulation to compensate for smaller batches
                args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 4)
                # Disable GNN if memory is very constrained
                if total_mem < 4000:
                    logger.warning("Very limited VRAM detected. Disabling GNN and reducing model size.")
                    config["model"]["use_gnn"] = False
                    config["model"]["num_scales"] = min(config["model"].get("num_scales", 3), 2)
                    config["model"]["hidden_channels"] = min(config["model"].get("hidden_channels", 48), 32)
                    args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 8)
        
        logger.info(f"Using batch size: {config['data']['batch_size']} with "
                    f"{args.gradient_accumulation_steps} gradient accumulation steps "
                    f"(effective batch size: {config['data']['batch_size'] * args.gradient_accumulation_steps})")
        
        # Safety check for potentially problematic configurations
        if config["model"].get("use_gnn", True) and config["model"].get("hidden_channels", 64) > 48:
            logger.warning("Using GNN with large hidden channels may cause OOM errors on 4GB GPUs.")
            logger.warning("Consider setting --auto_config to automatically adjust parameters.")
        
        # Create dataloaders
        try:
            train_loader, val_loader, data_info = create_dataloaders(
                data_path=args.data_path,
                batch_size=config["data"]["batch_size"],
                sequence_length=config["data"].get("sequence_length", 10),
                predict_steps=config["data"].get("predict_steps", 1),
                val_split=config["data"].get("val_split", 0.2),
                num_workers=min(config["data"].get("num_workers", 4), 2),  # Reduce workers for safety
                pin_memory=False  # Disable pin_memory on limited VRAM
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during dataloader creation")
                if not args.auto_config:
                    logger.info("Retry with --auto_config to automatically adjust parameters")
                raise
            else:
                raise
        
        # Clear memory before model creation
        optimize_memory()
        
        # Create model - wrap in try/except to handle OOM errors
        try:
            model = create_model(config, data_info)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during model creation")
                if config["model"].get("use_gnn", True) and not args.auto_config:
                    logger.info("Try setting model.use_gnn: false in config or use --auto_config")
                raise
            else:
                raise
        
        # Enable checkpointing for memory efficiency
        if args.gradient_checkpointing:
            model = enable_checkpointing(model)
        
        # Clear memory before optimizer creation
        optimize_memory()
        
        # Create optimizer
        optimizer = create_optimizer(config, model)
        
        # Create learning rate scheduler
        scheduler_params = config.get("scheduler", {})
        lr = config["optimizer"].get("learning_rate", 1e-3)
        
        if scheduler_params.get("use_scheduler", False):
            scheduler_type = scheduler_params.get("type", "plateau").lower()
            if scheduler_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_params.get("factor", 0.5),
                    patience=scheduler_params.get("patience", 10),
                    verbose=True,
                    min_lr=scheduler_params.get("min_lr", 1e-6)
                )
            elif scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_params.get("t_max", 100),
                    eta_min=scheduler_params.get("min_lr", 1e-6)
                )
            elif scheduler_type == "onecycle":
                training_params = config.get("training", {})
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_params.get("max_lr", lr * 10),
                    total_steps=training_params.get("num_epochs", 100) * len(train_loader) // args.gradient_accumulation_steps,
                    pct_start=scheduler_params.get("pct_start", 0.3),
                    div_factor=scheduler_params.get("div_factor", 25.0),
                    final_div_factor=scheduler_params.get("final_div_factor", 10000.0)
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_params.get("step_size", 30),
                    gamma=scheduler_params.get("gamma", 0.1)
                )
        else:
            scheduler = None
        
        # Get device
        device = torch.device(args.device)
        
        # Special handling for 16xx series GPUs
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "1650" in gpu_name or "1660" in gpu_name or "16 series" in gpu_name:
                logger.info("NVIDIA 16xx series GPU detected: enabling 16xx-specific optimizations")
                # These GPUs don't have Tensor Cores, so use regular FP16 mixed precision only if needed
                # and be extra cautious with memory
                if args.mixed_precision:
                    logger.info("Note: 16xx GPUs don't have Tensor Cores, mixed precision provides less benefit")
        
        # Create trainer with memory optimizations
        training_params = config.get("training", {})
        gradient_clip_val = training_params.get("gradient_clip_val", 1.0)
        trainer = MemoryEfficientTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad_norm=gradient_clip_val,
            memory_efficient=True
        )
        
        # Create efficient dataloaders with smaller prefetch factor for memory efficiency
        try:
            efficient_train_loader = create_efficient_dataloader(
                dataset=train_loader.dataset,
                batch_size=config["data"]["batch_size"],
                num_workers=min(config["data"].get("num_workers", 4), 2),  # Reduce workers
                pin_memory=False,  # Disable pin memory for limited VRAM GPUs
                prefetch_to_device=None,  # Disable prefetch to device for safety
                prefetch_factor=2  # Smaller prefetch factor
            )
            
            efficient_val_loader = create_efficient_dataloader(
                dataset=val_loader.dataset,
                batch_size=config["data"]["batch_size"],
                num_workers=min(config["data"].get("num_workers", 4), 2),  # Reduce workers
                pin_memory=False,  # Disable pin memory for limited VRAM GPUs
                prefetch_to_device=None,  # Disable prefetch to device for safety
                prefetch_factor=2  # Smaller prefetch factor
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during dataloader creation")
                logger.info("Try reducing batch size or model size")
                raise
            else:
                raise
        
        # Training history
        history = {
            "train_loss": [],
            "train_data_loss": [],
            "train_physics_loss": [],
            "val_loss": [],
            "val_data_loss": [],
            "val_physics_loss": []
        }
        
        # Get training parameters
        num_epochs = training_params.get("num_epochs", 100)
        early_stopping_patience = training_params.get("early_stopping_patience", 15)
        log_interval = training_params.get("log_interval", 10)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Train loop
        start_time = time.time()
        last_checkpoint_time = start_time
        
        # Log initial memory state
        log_memory_usage("Training start")
        
        # Use try-except to handle potential OOM errors
        try:
            for epoch in range(num_epochs):
                # Clear memory before each epoch
                optimize_memory()
                
                # Train for one epoch
                try:
                    train_metrics = train_epoch_with_efficiency(
                        trainer=trainer,
                        train_loader=efficient_train_loader,
                        val_loader=efficient_val_loader,
                        epoch=epoch,
                        log_interval=log_interval
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(f"CUDA out of memory during epoch {epoch}")
                        logger.info("Trying to recover by clearing memory and reducing batch size")
                        
                        # Clear CUDA cache and run garbage collection
                        optimize_memory()
                        
                        # Reduce batch size for recovery
                        old_batch_size = config["data"]["batch_size"]
                        config["data"]["batch_size"] = max(1, old_batch_size // 2)
                        logger.info(f"Reduced batch size from {old_batch_size} to {config['data']['batch_size']}")
                        
                        # Recreate dataloaders with smaller batch size
                        efficient_train_loader = create_efficient_dataloader(
                            dataset=train_loader.dataset,
                            batch_size=config["data"]["batch_size"],
                            num_workers=1,  # Reduce workers even more for recovery
                            pin_memory=False,
                            prefetch_to_device=None,
                            prefetch_factor=1
                        )
                        
                        efficient_val_loader = create_efficient_dataloader(
                            dataset=val_loader.dataset,
                            batch_size=config["data"]["batch_size"],
                            num_workers=1,  # Reduce workers even more for recovery
                            pin_memory=False,
                            prefetch_to_device=None,
                            prefetch_factor=1
                        )
                        
                        # Try again with smaller batch size
                        try:
                            train_metrics = train_epoch_with_efficiency(
                                trainer=trainer,
                                train_loader=efficient_train_loader,
                                val_loader=efficient_val_loader,
                                epoch=epoch,
                                log_interval=log_interval
                            )
                            logger.info("Successfully recovered from OOM error")
                        except RuntimeError as e2:
                            logger.error(f"Failed to recover: {str(e2)}")
                            # Save emergency checkpoint before raising
                            try:
                                emergency_path = os.path.join(args.output_dir, "emergency_checkpoint.pt")
                                model.save_model(save_dir=args.output_dir, filename="emergency_checkpoint.pt")
                                logger.info(f"Saved emergency checkpoint to {emergency_path}")
                            except:
                                pass
                            raise e2
                    else:
                        raise
                
                # Log memory usage
                log_memory_usage(f"Epoch {epoch} completed")
                
                # Update history
                for key, value in train_metrics.items():
                    if value is not None and key in history:
                        history[key].append(value)
                
                # Check for early stopping
                val_loss = train_metrics.get("val_loss")
                if val_loss is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model
                        best_model_path = os.path.join(args.output_dir, "best_model.pt")
                        # Clear memory before saving
                        optimize_memory()
                        model.save_model(save_dir=args.output_dir, filename="best_model.pt")
                        logger.info(f"Epoch {epoch}: Saved new best model with val_loss={best_val_loss:.6f}")
                    else:
                        patience_counter += 1
                        logger.info(f"Epoch {epoch}: Early stopping patience {patience_counter}/{early_stopping_patience}")
                        
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                            break
                
                # Auto-save checkpoint periodically
                current_time = time.time()
                if (current_time - last_checkpoint_time) > 1800:  # 30 minutes
                    # Clear memory before saving
                    optimize_memory()
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_time_{int(current_time)}.pt")
                    model.save_model(save_dir=args.output_dir, filename=f"checkpoint_time_{int(current_time)}.pt")
                    logger.info(f"Auto-saved checkpoint at epoch {epoch + 1}")
                    last_checkpoint_time = current_time
                
                # Save checkpoint at intervals
                save_interval = training_params.get("save_interval", 10)
                if (epoch + 1) % save_interval == 0:
                    # Clear memory before saving
                    optimize_memory()
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    model.save_model(save_dir=args.output_dir, filename=f"checkpoint_epoch_{epoch + 1}.pt")
                    logger.info(f"Saved checkpoint at epoch {epoch + 1}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save emergency checkpoint
            emergency_path = os.path.join(args.output_dir, "interrupt_checkpoint.pt")
            try:
                optimize_memory()
                model.save_model(save_dir=args.output_dir, filename="interrupt_checkpoint.pt")
                logger.info(f"Saved checkpoint after interruption to {emergency_path}")
            except:
                logger.error("Could not save checkpoint after interruption")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory error during training: {str(e)}")
                logger.info("Try reducing batch size, model size, or increasing gradient accumulation steps")
                # Save emergency checkpoint
                emergency_path = os.path.join(args.output_dir, "emergency_checkpoint.pt")
                try:
                    optimize_memory()
                    model.save_model(save_dir=args.output_dir, filename="emergency_checkpoint.pt")
                    logger.info(f"Saved emergency checkpoint to {emergency_path}")
                except:
                    logger.error("Could not save emergency checkpoint")
            else:
                # Re-raise if not OOM
                raise
        
        finally:
            # Reset GPU clocks if they were locked
            if args.lock_gpu_clocks:
                reset_gpu_clocks()
        
        # Clear memory before final operations
        optimize_memory()
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model.pt")
        model.save_model(save_dir=args.output_dir, filename="final_model.pt")
        
        # Calculate total training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # Log final memory state
        log_memory_usage("Training end")
        
        # Save training history
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(val) for val in v] for k, v in history.items()}, f, indent=2)
        
        # Plot training history
        optimize_memory()  # Clear memory before plotting
        plot_training_history(
            history=history,
            save_path=os.path.join(args.output_dir, "training_history.png"),
            show=False
        )
        
        return model, history
    
    except Exception as e:
        # Reset GPU clocks if they were locked, even if we hit an exception
        if args.lock_gpu_clocks:
            reset_gpu_clocks()
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Memory-optimized training script for Physics-Informed Neural Networks on limited hardware"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="flood_warning_system/config/multi_scale_pinn_config.yaml",
        help="Path to configuration file"
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
        default="flood_warning_system/models/optimized",
        help="Directory to save model checkpoints and logs"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run memory-efficient hyperparameter optimization before training"
    )
    
    parser.add_argument(
        "--n_calls",
        type=int,
        default=10,  # Reduced for memory efficiency
        help="Number of iterations for Bayesian optimization"
    )
    
    parser.add_argument(
        "--n_random_starts",
        type=int,
        default=3,  # Reduced for memory efficiency
        help="Number of random initial points for Bayesian optimization"
    )
    
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training (recommended for memory efficiency)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Increased default for memory efficiency
        help="Number of steps to accumulate gradients (for larger effective batch size)"
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage"
    )
    
    parser.add_argument(
        "--auto_config",
        action="store_true",
        help="Automatically configure model for best performance on limited VRAM"
    )
    
    parser.add_argument(
        "--lock_gpu_clocks",
        action="store_true",
        help="Lock GPU clock speeds to prevent throttling (helps stability)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Enable mixed precision by default for memory efficiency
    if not args.mixed_precision and torch.cuda.is_available():
        logger.info("Mixed precision is highly recommended for memory efficiency. Enabling by default.")
        args.mixed_precision = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log initial system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem, allocated, free, cached = get_gpu_memory_info()
        logger.info(f"GPU Memory: Total={total_mem:.1f}MB, Used={allocated:.1f}MB, Free={free:.1f}MB")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Save the command-line arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Run hyperparameter optimization if requested
    if args.optimize:
        try:
            optimal_config_path = optimize_hyperparameters(args)
            config = load_config(optimal_config_path)
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            logger.info("Falling back to standard configuration")
            config = load_config(args.config)
    else:
        config = load_config(args.config)
    
    # Save the configuration used for training
    with open(os.path.join(args.output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU training")
        args.device = "cpu"
        args.mixed_precision = False
    
    # Train model with memory optimization
    try:
        train_single_gpu_memory_efficient(args, config)
        logger.info(f"Training complete. Results saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error("Check logs for details and consider reducing model complexity or batch size")
        raise


if __name__ == "__main__":
    main() 