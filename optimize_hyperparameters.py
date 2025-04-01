#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter optimization script for the Multi-scale Physics-Informed Neural Network.

This script uses Bayesian Optimization to find optimal hyperparameters for the PINN model,
focusing on learning rate, physics weight, network architecture, and other key parameters.
"""

import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
import logging
from functools import partial
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and training utilities
from flood_warning_system.models.multi_scale_pinn import MultiScalePINN
from flood_warning_system.train_multi_scale_pinn import (
    load_config, 
    create_dataloaders, 
    create_model, 
    create_optimizer,
    train_multi_scale_pinn
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flood_warning_system/logs/hyperparameter_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_search_space():
    """
    Define the search space for Bayesian optimization.
    
    Returns:
        List of parameter spaces to search
    """
    # Define the hyperparameter search space
    space = [
        # Model architecture
        Integer(32, 128, name="hidden_channels"),
        Integer(2, 5, name="num_scales"),
        Categorical([True, False], name="use_gnn"),
        
        # Optimizer
        Real(1e-5, 1e-2, "log-uniform", name="learning_rate"),
        Real(1e-6, 1e-3, "log-uniform", name="weight_decay"),
        Categorical(["adam", "sgd"], name="optimizer_type"),
        
        # Physics parameters
        Real(0.05, 0.5, name="physics_weight"),
        Real(0.5, 2.0, name="continuity_weight"),
        Real(0.1, 1.0, name="x_momentum_weight"),
        Real(0.1, 1.0, name="y_momentum_weight"),
        
        # Training parameters
        Integer(8, 32, name="batch_size"),
        Real(0.0, 0.3, name="dropout")
    ]
    
    return space


def update_config_with_params(config, params):
    """
    Update configuration dictionary with new parameters.
    
    Args:
        config: Original configuration dictionary
        params: Dictionary of parameters to update
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy of the config
    updated_config = {k: v.copy() if isinstance(v, dict) else v for k, v in config.items()}
    
    # Update model parameters
    if "model" not in updated_config:
        updated_config["model"] = {}
    
    if "hidden_channels" in params:
        updated_config["model"]["hidden_channels"] = params["hidden_channels"]
    if "num_scales" in params:
        updated_config["model"]["num_scales"] = params["num_scales"]
    if "use_gnn" in params:
        updated_config["model"]["use_gnn"] = params["use_gnn"]
    if "dropout" in params:
        updated_config["model"]["dropout"] = params["dropout"]
        
    # Update physics parameters
    if "physics" not in updated_config:
        updated_config["physics"] = {}
    
    if "physics_weight" in params:
        updated_config["physics"]["initial_weight"] = params["physics_weight"]
    if "continuity_weight" in params:
        updated_config["physics"]["continuity_weight"] = params["continuity_weight"]
    if "x_momentum_weight" in params:
        updated_config["physics"]["x_momentum_weight"] = params["x_momentum_weight"]
    if "y_momentum_weight" in params:
        updated_config["physics"]["y_momentum_weight"] = params["y_momentum_weight"]
    
    # Update optimizer parameters
    if "optimizer" not in updated_config:
        updated_config["optimizer"] = {}
    
    if "learning_rate" in params:
        updated_config["optimizer"]["learning_rate"] = params["learning_rate"]
    if "weight_decay" in params:
        updated_config["optimizer"]["weight_decay"] = params["weight_decay"]
    if "optimizer_type" in params:
        updated_config["optimizer"]["type"] = params["optimizer_type"]
    
    # Update data parameters
    if "data" not in updated_config:
        updated_config["data"] = {}
    
    if "batch_size" in params:
        updated_config["data"]["batch_size"] = params["batch_size"]
    
    # Update training parameters
    if "training" not in updated_config:
        updated_config["training"] = {}
    
    # Set a lower number of epochs for hyperparameter search
    updated_config["training"]["num_epochs"] = 20
    
    # Disable early stopping during hyperparameter search to ensure consistent training length
    updated_config["training"]["early_stopping_patience"] = 999
    
    return updated_config


def objective_function(params, base_config, data_path, output_dir, device):
    """
    Objective function for Bayesian optimization.
    
    Args:
        params: Dictionary of parameters to evaluate
        base_config: Base configuration dictionary
        data_path: Path to the data
        output_dir: Directory to save outputs
        device: Device to use for training
        
    Returns:
        Validation loss (to minimize)
    """
    try:
        # Update configuration with new parameters
        config = update_config_with_params(base_config, params)
        
        # Create a unique run directory
        run_id = int(time.time())
        run_dir = os.path.join(output_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save the configuration
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create dataloaders
        train_loader, val_loader, data_info = create_dataloaders(
            data_path=data_path,
            batch_size=config["data"]["batch_size"],
            sequence_length=config["data"].get("sequence_length", 10),
            predict_steps=config["data"].get("predict_steps", 1),
            val_split=config["data"].get("val_split", 0.2),
            num_workers=min(4, os.cpu_count() or 1)  # Limit workers for parallel optimization
        )
        
        # Create model
        model = create_model(config, data_info)
        
        # Create optimizer
        optimizer = create_optimizer(config, model)
        
        # Train the model
        history = train_multi_scale_pinn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,  # No scheduler for hyperparameter search
            num_epochs=config["training"]["num_epochs"],
            save_dir=run_dir,
            device=device,
            early_stopping_patience=999,  # Disable early stopping
            log_physics=True,
            save_interval=999  # Don't save intermediate models
        )
        
        # Get the best validation loss
        val_losses = history.get("val_loss", [1e10])  # Default high value if no validation
        best_val_loss = min(val_losses) if val_losses else 1e10
        
        # Get the best combined loss (weighted sum of data and physics losses)
        val_data_losses = history.get("val_data_loss", [])
        val_physics_losses = history.get("val_physics_loss", [])
        
        if val_data_losses and val_physics_losses:
            physics_weight = config["physics"]["initial_weight"]
            combined_losses = [d + physics_weight * p for d, p in zip(val_data_losses, val_physics_losses)]
            best_combined_loss = min(combined_losses)
        else:
            best_combined_loss = best_val_loss
        
        # Save the metrics
        metrics = {
            "params": params,
            "best_val_loss": float(best_val_loss),
            "best_combined_loss": float(best_combined_loss),
            "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float(1e10),
            "train_data_loss": float(history["train_data_loss"][-1]) if history["train_data_loss"] else float(1e10),
            "train_physics_loss": float(history["train_physics_loss"][-1]) if history["train_physics_loss"] else float(1e10),
            "epochs": len(history["train_loss"])
        }
        
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Run {run_id} completed: val_loss={best_val_loss:.6f}, combined_loss={best_combined_loss:.6f}")
        
        # Return the loss to minimize
        return best_combined_loss
    
    except Exception as e:
        logger.error(f"Error in objective function: {str(e)}")
        # Return a high value on error
        return 1e10


def save_optimization_results(result, output_dir):
    """
    Save optimization results.
    
    Args:
        result: Optimization result object
        output_dir: Directory to save results
    """
    # Save the optimization results
    result_path = os.path.join(output_dir, "optimization_result.json")
    
    # Convert result to JSON-serializable format
    serializable_result = {
        "x": result.x,
        "fun": float(result.fun),
        "x_iters": [list(map(float, x)) for x in result.x_iters],
        "func_vals": [float(f) for f in result.func_vals],
        "space_dim": result.space.dimensions_names(),
        "models": [str(model) for model in result.models]
    }
    
    with open(result_path, "w") as f:
        json.dump(serializable_result, f, indent=2)
    
    # Create convergence plot
    fig_convergence = plot_convergence(result)
    plt.savefig(os.path.join(output_dir, "convergence.png"))
    plt.close(fig_convergence)
    
    # Create plots for each objective dimension
    for dim_name in result.space.dimensions_names():
        fig_objective = plot_objective(result, dimension_name=dim_name)
        plt.savefig(os.path.join(output_dir, f"objective_{dim_name}.png"))
        plt.close(fig_objective)


def create_optimal_config(base_config, optimal_params, output_dir):
    """
    Create and save optimal configuration.
    
    Args:
        base_config: Base configuration dictionary
        optimal_params: Optimal parameters dictionary
        output_dir: Directory to save the configuration
        
    Returns:
        Optimal configuration dictionary
    """
    # Update base config with optimal parameters
    optimal_config = update_config_with_params(base_config, optimal_params)
    
    # Reset training parameters to default
    if "training" in optimal_config:
        optimal_config["training"]["num_epochs"] = base_config.get("training", {}).get("num_epochs", 100)
        optimal_config["training"]["early_stopping_patience"] = base_config.get("training", {}).get("early_stopping_patience", 15)
    
    # Save the optimal configuration
    config_path = os.path.join(output_dir, "optimal_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(optimal_config, f)
    
    logger.info(f"Optimal configuration saved to {config_path}")
    
    return optimal_config


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for Multi-scale PINN")
    
    parser.add_argument(
        "--config",
        type=str,
        default="flood_warning_system/config/multi_scale_pinn_config.yaml",
        help="Path to base configuration file"
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
        default="flood_warning_system/models/optimization",
        help="Directory to save optimization results"
    )
    
    parser.add_argument(
        "--n_calls",
        type=int,
        default=20,
        help="Number of iterations for Bayesian optimization"
    )
    
    parser.add_argument(
        "--n_random_starts",
        type=int,
        default=5,
        help="Number of random initial points for Bayesian optimization"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Define search space
    space = create_search_space()
    
    # Create the objective function with fixed parameters
    objective = partial(
        objective_function,
        base_config=base_config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Wrap the objective function to use named parameters
    @use_named_args(space)
    def objective_wrapper(**params):
        return objective(params)
    
    # Run Bayesian optimization
    logger.info(f"Starting Bayesian optimization with {args.n_calls} iterations")
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts,
        random_state=42,
        verbose=True
    )
    
    # Save optimization results
    save_optimization_results(result, args.output_dir)
    
    # Get optimal parameters
    optimal_params = {dim.name: result.x[i] for i, dim in enumerate(space)}
    logger.info(f"Optimal parameters: {optimal_params}")
    
    # Create and save optimal configuration
    optimal_config = create_optimal_config(base_config, optimal_params, args.output_dir)
    
    # Print summary
    logger.info(f"Optimization completed with best loss: {result.fun:.6f}")
    logger.info(f"Optimal configuration saved to {os.path.join(args.output_dir, 'optimal_config.yaml')}")


if __name__ == "__main__":
    main() 