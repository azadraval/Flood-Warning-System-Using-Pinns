#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration script for the Multi-scale Physics-Informed Neural Network (PINN).

This script demonstrates how to use the trained multi-scale PINN model
for flood forecasting as part of the flood warning system.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from pathlib import Path
import logging
import time
import json

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MultiScalePINN model
from flood_warning_system.models.multi_scale_pinn import MultiScalePINN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flood_warning_system/logs/multi_scale_pinn_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a trained multi-scale PINN model.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model hyperparameters from the checkpoint
        model_params = checkpoint.get("model_params", {})
        
        # Create a new model with the same hyperparameters
        model = MultiScalePINN(
            input_channels=model_params.get("input_channels", 3),
            output_channels=model_params.get("output_channels", 3),
            hidden_channels=model_params.get("hidden_channels", 64),
            fno_modes=model_params.get("fno_modes", (12, 12)),
            num_scales=model_params.get("num_scales", 3),
            use_gnn=model_params.get("use_gnn", True),
            grid_size=model_params.get("grid_size", (64, 64)),
            dx=model_params.get("dx", 1.0),
            dy=model_params.get("dy", 1.0),
            dt=model_params.get("dt", 0.1),
            gravity=model_params.get("gravity", 9.81),
            manning_coef=model_params.get("manning_coef", 0.035),
            physics_weight=model_params.get("physics_weight", 0.1),
            dropout=model_params.get("dropout", 0.0),
            # Add all physics parameters with appropriate defaults
            adaptive_weighting=model_params.get("adaptive_weighting", True),
            min_physics_weight=model_params.get("min_physics_weight", 0.01),
            max_physics_weight=model_params.get("max_physics_weight", 1.0),
            adaptation_rate=model_params.get("adaptation_rate", 0.05),
            continuity_weight=model_params.get("continuity_weight", 1.0),
            x_momentum_weight=model_params.get("x_momentum_weight", 0.5),
            y_momentum_weight=model_params.get("y_momentum_weight", 0.5),
            boundary_weight=model_params.get("boundary_weight", 0.5),
            enforce_positivity=model_params.get("enforce_positivity", True)
        )
        
        # Load the model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        logger.info(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_initial_conditions(data_path, variables=None):
    """
    Load initial conditions from a netCDF file.
    
    Args:
        data_path: Path to the netCDF file
        variables: List of variables to load (default: stage, xmomentum, ymomentum)
        
    Returns:
        Dictionary containing the initial conditions
    """
    if variables is None:
        variables = ["stage", "xmomentum", "ymomentum", "elevation"]
    
    logger.info(f"Loading initial conditions from {data_path}")
    
    # Load the data
    ds = xr.open_dataset(data_path)
    
    # Extract the variables
    data = {}
    
    for var in variables:
        if var in ds:
            # Extract data for the variable
            var_data = ds[var].values
            
            # Add time dimension if it doesn't exist
            if var == "elevation" and len(var_data.shape) == 2:
                var_data = var_data[np.newaxis, ...]
            
            # Convert to torch tensor
            data[var] = torch.tensor(var_data, dtype=torch.float32)
        else:
            logger.warning(f"Variable {var} not found in dataset")
    
    # Get spatial dimensions
    if "stage" in data:
        height, width = data["stage"].shape[-2:]
    elif "elevation" in data:
        height, width = data["elevation"].shape[-2:]
    else:
        raise ValueError("Could not determine spatial dimensions")
    
    # Get coordinates if available
    x = ds.x.values if "x" in ds else np.arange(width)
    y = ds.y.values if "y" in ds else np.arange(height)
    
    # Create a dictionary with the initial conditions
    initial_conditions = {
        "data": data,
        "spatial_dims": (height, width),
        "coordinates": {
            "x": x,
            "y": y
        }
    }
    
    return initial_conditions


def prepare_model_input(initial_conditions, time_idx=0, device=None):
    """
    Prepare input for the model from initial conditions.
    
    Args:
        initial_conditions: Dictionary containing initial conditions
        time_idx: Time index to use
        device: Device to load the data to
        
    Returns:
        Dictionary containing model inputs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract data
    data = initial_conditions["data"]
    
    # Stack input variables
    input_vars = ["stage", "xmomentum", "ymomentum"]
    inputs = []
    
    for var in input_vars:
        if var in data:
            # Extract data for the specific time index
            if data[var].shape[0] > time_idx:
                inputs.append(data[var][time_idx])
            else:
                logger.warning(f"Time index {time_idx} out of bounds for {var}")
                inputs.append(data[var][0])
        else:
            logger.warning(f"Variable {var} not found in initial conditions")
            # Add zeros if variable not found
            inputs.append(torch.zeros(initial_conditions["spatial_dims"]))
    
    # Stack along channel dimension
    model_input = torch.stack(inputs, dim=0).unsqueeze(0)  # Add batch dimension
    
    # Add elevation if available
    if "elevation" in data:
        elevation = data["elevation"][0].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elevation = elevation.to(device)
    else:
        elevation = None
    
    # Move to device
    model_input = model_input.to(device)
    
    return {
        "inputs": model_input,
        "elevation": elevation
    }


def run_flood_forecast(model, initial_conditions, num_steps=24, device=None):
    """
    Run a flood forecast using the model.
    
    Args:
        model: Trained model
        initial_conditions: Dictionary containing initial conditions
        num_steps: Number of time steps to forecast
        device: Device to run the model on
        
    Returns:
        Forecast results as an xarray Dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Running flood forecast for {num_steps} steps")
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare initial input
    model_input = prepare_model_input(initial_conditions, device=device)
    
    # Initialize storage for predictions
    predictions = {
        "stage": [],
        "xmomentum": [],
        "ymomentum": []
    }
    
    # Get spatial dimensions
    height, width = initial_conditions["spatial_dims"]
    
    # Run forecast
    with torch.no_grad():
        # Initial input
        current_input = model_input["inputs"]
        elevation = model_input["elevation"]
        
        # Track execution time
        start_time = time.time()
        
        # Run prediction for each time step
        for step in range(num_steps):
            try:
                # If using elevation, concatenate with input
                if elevation is not None:
                    combined_input = torch.cat([current_input, elevation], dim=1)
                else:
                    combined_input = current_input
                
                # Forward pass
                prediction = model(combined_input)
                
                # Check for NaN values
                if torch.isnan(prediction).any():
                    logger.warning(f"NaN values detected in prediction at step {step}")
                    # Replace NaN values with zeros
                    prediction = torch.where(torch.isnan(prediction), 
                                           torch.zeros_like(prediction), 
                                           prediction)
                
                # Enforce physical constraints - water depth must be non-negative
                prediction[:, 0:1, :, :] = torch.clamp(prediction[:, 0:1, :, :], min=0.0)
                
                # Extract variables and store
                stage = prediction[0, 0].cpu().numpy()
                xmomentum = prediction[0, 1].cpu().numpy()
                ymomentum = prediction[0, 2].cpu().numpy()
                
                predictions["stage"].append(stage)
                predictions["xmomentum"].append(xmomentum)
                predictions["ymomentum"].append(ymomentum)
                
                # Update input for next step
                current_input = prediction
            
            except Exception as e:
                logger.error(f"Error during prediction at step {step}: {str(e)}")
                # If an error occurs, pad the remaining steps with the last prediction
                if len(predictions["stage"]) > 0:
                    for _ in range(step, num_steps):
                        predictions["stage"].append(predictions["stage"][-1])
                        predictions["xmomentum"].append(predictions["xmomentum"][-1])
                        predictions["ymomentum"].append(predictions["ymomentum"][-1])
                else:
                    # If no predictions were made yet, fill with zeros
                    zero_stage = np.zeros((height, width))
                    for _ in range(num_steps):
                        predictions["stage"].append(zero_stage.copy())
                        predictions["xmomentum"].append(zero_stage.copy())
                        predictions["ymomentum"].append(zero_stage.copy())
                break
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Forecast completed in {execution_time:.2f} seconds")
    
    # Convert to numpy arrays
    for var in predictions:
        predictions[var] = np.array(predictions[var])
    
    # Create xarray Dataset
    coords = {
        "time": np.arange(num_steps),
        "y": initial_conditions["coordinates"]["y"],
        "x": initial_conditions["coordinates"]["x"]
    }
    
    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "stage": (["time", "y", "x"], predictions["stage"]),
            "xmomentum": (["time", "y", "x"], predictions["xmomentum"]),
            "ymomentum": (["time", "y", "x"], predictions["ymomentum"])
        },
        coords=coords
    )
    
    # Add velocity components with safety against division by zero
    # Use a minimum water depth threshold for velocity calculations
    min_depth = 0.01  # 1 cm minimum depth for velocity calculation
    
    # Calculate velocities where depth is above threshold
    depth_mask = ds["stage"] > min_depth
    
    # Initialize velocity variables with zeros
    ds["xvelocity"] = xr.zeros_like(ds["stage"])
    ds["yvelocity"] = xr.zeros_like(ds["stage"])
    
    # Calculate velocities only where depth is sufficient
    ds["xvelocity"] = xr.where(depth_mask, ds["xmomentum"] / ds["stage"], 0.0)
    ds["yvelocity"] = xr.where(depth_mask, ds["ymomentum"] / ds["stage"], 0.0)
    
    # Add elevation if available
    if "elevation" in initial_conditions["data"]:
        elevation_np = initial_conditions["data"]["elevation"][0].cpu().numpy()
        ds["elevation"] = (["y", "x"], elevation_np)
    
    # Add metadata
    ds.attrs["forecast_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["execution_time"] = execution_time
    
    return ds


def create_forecast_animation(forecast_ds, output_path=None, fps=5, dpi=100):
    """
    Create an animation of the flood forecast.
    
    Args:
        forecast_ds: Forecast results as an xarray Dataset
        output_path: Path to save the animation (if None, only display)
        fps: Frames per second for the animation
        dpi: DPI for the animation
        
    Returns:
        Animation object
    """
    logger.info("Creating forecast animation")
    
    # Get data
    stage = forecast_ds["stage"].values
    
    # Calculate velocity magnitude for color coding
    xvel = forecast_ds["xvelocity"].values
    yvel = forecast_ds["yvelocity"].values
    vel_magnitude = np.sqrt(xvel**2 + yvel**2)
    
    # Get coordinates
    y = forecast_ds.y.values
    x = forecast_ds.x.values
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the first frame
    stage_plot = ax.imshow(
        stage[0],
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        cmap='Blues',
        norm=Normalize(vmin=0, vmax=np.max(stage) * 0.8)
    )
    
    # Add colorbar
    cbar = fig.colorbar(stage_plot, ax=ax, label='Water Depth (m)')
    
    # If velocity is available, add quiver plot
    quiver = None
    if not np.isnan(vel_magnitude).all():
        # Downsample for quiver plot
        skip = max(1, min(stage.shape[1], stage.shape[2]) // 20)
        quiver = ax.quiver(
            x[::skip], y[::skip],
            xvel[0, ::skip, ::skip],
            yvel[0, ::skip, ::skip],
            vel_magnitude[0, ::skip, ::skip],
            cmap='hot',
            scale=20,
            width=0.002
        )
    
    # Add title
    title = ax.set_title(f'Water Depth - Time: 0')
    
    # Add labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Function to update the animation
    def update(frame):
        # Update water depth plot
        stage_plot.set_array(stage[frame])
        
        # Update quiver plot if available
        if quiver is not None:
            quiver.set_UVC(
                xvel[frame, ::skip, ::skip],
                yvel[frame, ::skip, ::skip],
                vel_magnitude[frame, ::skip, ::skip]
            )
        
        # Update title
        title.set_text(f'Water Depth - Time: {frame}')
        
        return [stage_plot, title] + ([quiver] if quiver is not None else [])
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(stage),
        interval=1000/fps,
        blit=True
    )
    
    # Save if path is provided
    if output_path is not None:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        logger.info(f"Animation saved to {output_path}")
    
    plt.close()
    
    return anim


def compute_flood_metrics(forecast_ds, threshold=0.1):
    """
    Compute flood metrics from the forecast.
    
    Args:
        forecast_ds: Forecast results as an xarray Dataset
        threshold: Water depth threshold for flooding (m)
        
    Returns:
        Dictionary of flood metrics
    """
    logger.info("Computing flood metrics")
    
    # Get water depth
    stage = forecast_ds["stage"].values
    
    # Calculate flooded area at each time step
    pixel_count = (stage > threshold).sum(axis=(1, 2))
    
    # Calculate average water depth over flooded areas
    flooded_depth = np.array([
        np.mean(stage[t][stage[t] > threshold]) if np.any(stage[t] > threshold) else 0
        for t in range(len(stage))
    ])
    
    # Calculate maximum water depth
    max_depth = np.max(stage, axis=(1, 2))
    
    # Calculate time to peak
    time_to_peak = np.argmax(max_depth)
    
    # Create metrics dictionary
    metrics = {
        "flooded_pixel_count": pixel_count.tolist(),
        "flooded_depth_avg": flooded_depth.tolist(),
        "max_depth": max_depth.tolist(),
        "time_to_peak": int(time_to_peak),
        "threshold": threshold
    }
    
    return metrics


def generate_warning_levels(forecast_ds, thresholds=None):
    """
    Generate warning levels based on flood forecast.
    
    Args:
        forecast_ds: Forecast results as an xarray Dataset
        thresholds: Dictionary of thresholds for warning levels
            
    Returns:
        Dictionary of warning levels for different areas
    """
    if thresholds is None:
        thresholds = {
            "low": 0.1,    # 10 cm
            "medium": 0.3,  # 30 cm
            "high": 0.5,    # 50 cm
            "severe": 1.0   # 1 m
        }
    
    logger.info("Generating warning levels")
    
    # Get water depth
    stage = forecast_ds["stage"].values
    
    # Calculate warning levels for each time step
    warnings = []
    
    for t in range(len(stage)):
        # Calculate percentage of area in each warning level
        low = np.mean((stage[t] >= thresholds["low"]) & (stage[t] < thresholds["medium"]))
        medium = np.mean((stage[t] >= thresholds["medium"]) & (stage[t] < thresholds["high"]))
        high = np.mean((stage[t] >= thresholds["high"]) & (stage[t] < thresholds["severe"]))
        severe = np.mean(stage[t] >= thresholds["severe"])
        
        # Determine overall warning level
        if severe > 0.05:  # If more than 5% of area has severe flooding
            overall = "severe"
        elif high > 0.1:   # If more than 10% of area has high flooding
            overall = "high"
        elif medium > 0.2: # If more than 20% of area has medium flooding
            overall = "medium"
        elif low > 0.3:    # If more than 30% of area has low flooding
            overall = "low"
        else:
            overall = "none"
        
        warnings.append({
            "time": t,
            "levels": {
                "low": float(low),
                "medium": float(medium),
                "high": float(high),
                "severe": float(severe)
            },
            "overall": overall
        })
    
    # Create warning dictionary
    warning_dict = {
        "thresholds": thresholds,
        "warnings": warnings
    }
    
    return warning_dict


def save_forecast_results(forecast_ds, metrics, warnings, output_dir):
    """
    Save forecast results to disk.
    
    Args:
        forecast_ds: Forecast results as an xarray Dataset
        metrics: Dictionary of flood metrics
        warnings: Dictionary of warning levels
        output_dir: Directory to save results
        
    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save forecast data
    forecast_path = os.path.join(output_dir, f"forecast_{timestamp}.nc")
    forecast_ds.to_netcdf(forecast_path)
    logger.info(f"Forecast data saved to {forecast_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save warnings
    warnings_path = os.path.join(output_dir, f"warnings_{timestamp}.json")
    with open(warnings_path, 'w') as f:
        json.dump(warnings, f, indent=2)
    logger.info(f"Warnings saved to {warnings_path}")
    
    # Create animation
    animation_path = os.path.join(output_dir, f"animation_{timestamp}.gif")
    create_forecast_animation(forecast_ds, output_path=animation_path)
    
    # Return paths
    return {
        "forecast": forecast_path,
        "metrics": metrics_path,
        "warnings": warnings_path,
        "animation": animation_path
    }


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate flood forecasts using a trained multi-scale PINN model"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="flood_warning_system/models/saved/final_model.pt",
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--initial_conditions",
        type=str,
        default="flood_warning_system/data/processed/initial_conditions.nc",
        help="Path to the initial conditions data"
    )
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=24,
        help="Number of time steps to forecast"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flood_warning_system/output/forecasts",
        help="Directory to save forecast results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load the model
    model = load_model(args.model_path, device)
    
    # Load initial conditions
    initial_conditions = load_initial_conditions(args.initial_conditions)
    
    # Run forecast
    forecast_ds = run_flood_forecast(
        model=model,
        initial_conditions=initial_conditions,
        num_steps=args.num_steps,
        device=device
    )
    
    # Compute metrics
    metrics = compute_flood_metrics(forecast_ds)
    
    # Generate warnings
    warnings = generate_warning_levels(forecast_ds)
    
    # Save results
    save_forecast_results(
        forecast_ds=forecast_ds,
        metrics=metrics,
        warnings=warnings,
        output_dir=args.output_dir
    )
    
    logger.info("Forecast generation completed")


if __name__ == "__main__":
    main() 