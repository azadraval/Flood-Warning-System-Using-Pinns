#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare ANUGA simulation outputs for use with Physics-Informed Neural Networks.
This script processes the simulation results and creates properly formatted
training data for PINNs that enforce the shallow water equations.
"""

import os
import sys
import argparse
import logging
import json
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add the script directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Setup logging
log_file = f"pinn_data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing required scientific libraries with error handling
try:
    import xarray as xr
    import matplotlib.pyplot as plt
    import h5py
except ImportError as e:
    logger.error(f"Missing required dependency: {str(e)}")
    logger.error("Please install required dependencies: xarray, matplotlib, h5py")
    logger.error("You can install them with: pip install xarray matplotlib h5py")
    sys.exit(1)

def extract_shallow_water_variables(dataset_path, output_path, downsample_factor=1):
    """
    Extract variables relevant for shallow water equations from ANUGA output.
    
    Args:
        dataset_path: Path to ANUGA netCDF output file
        output_path: Path to save processed data
        downsample_factor: Factor to downsample the spatial and temporal dimensions
        
    Returns:
        Path to the saved processed data
    """
    logger.info(f"Extracting shallow water variables from {dataset_path}")
    
    # Ensure paths are Path objects
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load dataset
        ds = xr.open_dataset(dataset_path)
        
        # Extract scenario information
        scenario_name = ds.attrs.get('scenario_name', Path(dataset_path).stem)
        scenario_type = ds.attrs.get('scenario_type', 'unknown')
        
        # Downsample if requested
        if downsample_factor > 1:
            # Temporal downsampling
            time_slice = slice(0, len(ds.time), downsample_factor)
            ds = ds.isel(time=time_slice)
            
            # Spatial downsampling
            x_slice = slice(0, len(ds.x), downsample_factor)
            y_slice = slice(0, len(ds.y), downsample_factor)
            ds = ds.isel(x=x_slice, y=y_slice)
        
        # Calculate additional variables needed for SWE
        # Get the time step in seconds
        time_diff = np.diff(ds.time.values).astype('timedelta64[s]').astype(float)
        dt = float(np.mean(time_diff))
        
        # Create meshgrid of x and y coordinates
        X, Y = np.meshgrid(ds.x.values, ds.y.values)
        dx = float(np.mean(np.diff(ds.x.values)))
        dy = float(np.mean(np.diff(ds.y.values)))
        
        # SWE requires:
        # - h (water depth)
        # - u, v (velocity components)
        # - elevation (bed elevation)
        # - g (gravity constant)
        
        # Extract variables
        h = ds.water_depth.values  # (time, y, x)
        
        # Extract or calculate velocity components
        if 'velocity_x' in ds and 'velocity_y' in ds:
            u = ds.velocity_x.values  # (time, y, x)
            v = ds.velocity_y.values  # (time, y, x)
        else:
            # Calculate from scalar velocity
            # This is a simplification - in reality we would need the direction
            # We'll assume flow is primarily in the x-direction for demonstration
            vel = ds.flow_velocity.values  # (time, y, x)
            # Placeholder - in reality would need proper decomposition
            u = vel * 0.8  # Arbitrary split of velocity
            v = vel * 0.2
        
        # Get elevation
        z = ds.elevation.values  # (y, x)
        
        # Compute some terms used in shallow water equations
        # Mass conservation: dh/dt + d(hu)/dx + d(hv)/dy = 0
        
        # Calculate derivatives
        # Time derivatives
        dhdt = np.zeros_like(h)
        dhdt[1:, :, :] = (h[1:, :, :] - h[:-1, :, :]) / dt
        
        # Spatial derivatives (central difference)
        dhudx = np.zeros_like(h)
        dhvdy = np.zeros_like(h)
        
        for t in range(h.shape[0]):
            # Calculate hu and hv
            hu = h[t] * u[t]
            hv = h[t] * v[t]
            
            # x-derivatives (central difference)
            dhudx_t = np.zeros_like(hu)
            dhudx_t[:, 1:-1] = (hu[:, 2:] - hu[:, :-2]) / (2 * dx)
            # Forward/backward differences at boundaries
            dhudx_t[:, 0] = (hu[:, 1] - hu[:, 0]) / dx
            dhudx_t[:, -1] = (hu[:, -1] - hu[:, -2]) / dx
            
            # y-derivatives (central difference)
            dhvdy_t = np.zeros_like(hv)
            dhvdy_t[1:-1, :] = (hv[2:, :] - hv[:-2, :]) / (2 * dy)
            # Forward/backward differences at boundaries
            dhvdy_t[0, :] = (hv[1, :] - hv[0, :]) / dy
            dhvdy_t[-1, :] = (hv[-1, :] - hv[-2, :]) / dy
            
            dhudx[t] = dhudx_t
            dhvdy[t] = dhvdy_t
        
        # Calculate mass conservation residual
        mass_conservation = dhdt + dhudx + dhvdy
        
        # Create output dictionary
        data = {
            'water_depth': h,
            'velocity_x': u,
            'velocity_y': v,
            'elevation': z,
            'dhdt': dhdt,
            'dhudx': dhudx,
            'dhvdy': dhvdy,
            'mass_conservation': mass_conservation,
            'x': ds.x.values,
            'y': ds.y.values,
            'time': ds.time.values,
            'dx': dx,
            'dy': dy,
            'dt': dt
        }
        
        # Additional metadata
        metadata = {
            'scenario_name': scenario_name,
            'scenario_type': scenario_type,
            'description': ds.attrs.get('description', ''),
            'creation_date': str(datetime.now()),
            'source_file': dataset_path,
            'gravity': 9.81,  # m/s^2
            'manning_n': 0.03  # Manning's roughness coefficient (assumed)
        }
        
        # Close dataset
        ds.close()
        
        # Save to HDF5 format
        with h5py.File(output_path, 'w') as f:
            # Create groups
            data_group = f.create_group('data')
            meta_group = f.create_group('metadata')
            
            # Store data
            for key, value in data.items():
                data_group.create_dataset(key, data=value)
            
            # Store metadata
            for key, value in metadata.items():
                meta_group.attrs[key] = value
        
        logger.info(f"Processed data saved to: {output_path}")
        
        # Create a visualization of SWE terms
        create_swe_visualization(data, metadata, output_path.replace('.h5', '_swe_visualization.png'))
        
        return output_path
        
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_path}: {str(e)}")
        return None

def create_swe_visualization(data, metadata, output_path):
    """
    Create visualization of shallow water equation terms.
    
    Args:
        data: Dictionary of processed data
        metadata: Dictionary of metadata
        output_path: Path to save visualization
    """
    # Select a middle timestep
    t_idx = len(data['time']) // 2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot water depth
    im0 = axes[0].pcolormesh(data['x'], data['y'], data['water_depth'][t_idx], cmap='Blues')
    axes[0].set_title('Water Depth')
    fig.colorbar(im0, ax=axes[0])
    
    # Plot x-velocity
    im1 = axes[1].pcolormesh(data['x'], data['y'], data['velocity_x'][t_idx], cmap='RdBu_r')
    axes[1].set_title('Velocity (x)')
    fig.colorbar(im1, ax=axes[1])
    
    # Plot y-velocity
    im2 = axes[2].pcolormesh(data['x'], data['y'], data['velocity_y'][t_idx], cmap='RdBu_r')
    axes[2].set_title('Velocity (y)')
    fig.colorbar(im2, ax=axes[2])
    
    # Plot dh/dt
    im3 = axes[3].pcolormesh(data['x'], data['y'], data['dhdt'][t_idx], cmap='RdBu_r')
    axes[3].set_title('dh/dt')
    fig.colorbar(im3, ax=axes[3])
    
    # Plot d(hu)/dx
    im4 = axes[4].pcolormesh(data['x'], data['y'], data['dhudx'][t_idx], cmap='RdBu_r')
    axes[4].set_title('d(hu)/dx')
    fig.colorbar(im4, ax=axes[4])
    
    # Plot mass conservation residual
    im5 = axes[5].pcolormesh(data['x'], data['y'], data['mass_conservation'][t_idx], cmap='RdBu_r')
    axes[5].set_title('Mass Conservation Residual')
    fig.colorbar(im5, ax=axes[5])
    
    # Set common aspects
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    
    # Add metadata as text
    plt.figtext(0.5, 0.01, f"Scenario: {metadata['scenario_name']} (Type: {metadata['scenario_type']})", 
                ha='center', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"SWE visualization saved to: {output_path}")

def extract_boundary_conditions(dataset_path, output_path):
    """
    Extract boundary conditions from ANUGA output for PINNs.
    
    Args:
        dataset_path: Path to ANUGA netCDF output file
        output_path: Path to save boundary conditions
        
    Returns:
        Path to boundary conditions file
    """
    logger.info(f"Extracting boundary conditions from: {dataset_path}")
    
    # Load dataset
    ds = xr.open_dataset(dataset_path)
    
    # Extracting boundary data
    # Get domain boundaries
    x_min, x_max = float(ds.x.min()), float(ds.x.max())
    y_min, y_max = float(ds.y.min()), float(ds.y.max())
    
    # Extract data at boundaries
    # Left boundary (x=x_min)
    left_boundary = ds.isel(x=0)
    
    # Right boundary (x=x_max)
    right_boundary = ds.isel(x=-1)
    
    # Bottom boundary (y=y_min)
    bottom_boundary = ds.isel(y=0)
    
    # Top boundary (y=y_max)
    top_boundary = ds.isel(y=-1)
    
    # Save boundary conditions to netCDF
    boundaries = xr.Dataset(
        data_vars={
            # Left boundary
            'left_depth': (['time', 'y'], left_boundary.water_depth.values),
            'left_velocity': (['time', 'y'], left_boundary.flow_velocity.values),
            # Right boundary
            'right_depth': (['time', 'y'], right_boundary.water_depth.values),
            'right_velocity': (['time', 'y'], right_boundary.flow_velocity.values),
            # Bottom boundary
            'bottom_depth': (['time', 'x'], bottom_boundary.water_depth.values),
            'bottom_velocity': (['time', 'x'], bottom_boundary.flow_velocity.values),
            # Top boundary
            'top_depth': (['time', 'x'], top_boundary.water_depth.values),
            'top_velocity': (['time', 'x'], top_boundary.flow_velocity.values),
            # Elevation at boundaries
            'left_elevation': (['y'], left_boundary.elevation.values),
            'right_elevation': (['y'], right_boundary.elevation.values),
            'bottom_elevation': (['x'], bottom_boundary.elevation.values),
            'top_elevation': (['x'], top_boundary.elevation.values)
        },
        coords={
            'time': ds.time.values,
            'x': ds.x.values,
            'y': ds.y.values
        },
        attrs={
            'description': 'Boundary conditions for Physics-Informed Neural Networks',
            'source_file': dataset_path,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'creation_date': str(datetime.now())
        }
    )
    
    # Save to netCDF
    boundaries.to_netcdf(output_path)
    
    # Close datasets
    ds.close()
    boundaries.close()
    
    logger.info(f"Boundary conditions saved to: {output_path}")
    return output_path

def create_training_points(dataset_path, output_path, boundary_points=100, interior_points=1000):
    """
    Create training points for PINN by sampling from the dataset.
    
    Args:
        dataset_path: Path to processed HDF5 data file
        output_path: Path to save training points
        boundary_points: Number of points to sample on each boundary
        interior_points: Number of points to sample in the interior
        
    Returns:
        Path to training points file
    """
    logger.info(f"Creating training points from: {dataset_path}")
    
    # Load processed data
    with h5py.File(dataset_path, 'r') as f:
        data = {}
        for key in f['data'].keys():
            data[key] = f['data'][key][:]
        
        # Get metadata
        metadata = dict(f['metadata'].attrs)
    
    # Get domain dimensions
    x = data['x']
    y = data['y']
    t = data['time'].astype('datetime64[s]').astype(float)  # Convert to seconds
    
    # Normalize t to [0, 1] for better training
    t_norm = (t - t[0]) / (t[-1] - t[0])
    
    # Rescale x and y to [0, 1] as well
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[0]) / (y[-1] - y[0])
    
    # Create meshgrid for normalized coordinates
    X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
    
    # Create training points
    training_points = {
        'interior': [],   # Random points inside domain
        'boundary': [],   # Points on the domain boundary
        'initial': [],    # Points at initial time
        'data': []        # Sampled data points for supervised learning
    }
    
    # Generate interior points
    np.random.seed(42)  # For reproducibility
    
    # Sample random interior space-time points
    t_samples = np.random.uniform(0, 1, interior_points)
    x_samples = np.random.uniform(0, 1, interior_points)
    y_samples = np.random.uniform(0, 1, interior_points)
    
    for i in range(interior_points):
        training_points['interior'].append({
            't': float(t_samples[i]),
            'x': float(x_samples[i]),
            'y': float(y_samples[i])
        })
    
    # Sample points on boundaries
    # Left boundary (x=0)
    for i in range(boundary_points):
        t_val = np.random.uniform(0, 1)
        y_val = np.random.uniform(0, 1)
        training_points['boundary'].append({
            't': float(t_val),
            'x': 0.0,
            'y': float(y_val),
            'boundary': 'left'
        })
    
    # Right boundary (x=1)
    for i in range(boundary_points):
        t_val = np.random.uniform(0, 1)
        y_val = np.random.uniform(0, 1)
        training_points['boundary'].append({
            't': float(t_val),
            'x': 1.0,
            'y': float(y_val),
            'boundary': 'right'
        })
    
    # Bottom boundary (y=0)
    for i in range(boundary_points):
        t_val = np.random.uniform(0, 1)
        x_val = np.random.uniform(0, 1)
        training_points['boundary'].append({
            't': float(t_val),
            'x': float(x_val),
            'y': 0.0,
            'boundary': 'bottom'
        })
    
    # Top boundary (y=1)
    for i in range(boundary_points):
        t_val = np.random.uniform(0, 1)
        x_val = np.random.uniform(0, 1)
        training_points['boundary'].append({
            't': float(t_val),
            'x': float(x_val),
            'y': 1.0,
            'boundary': 'top'
        })
    
    # Sample points at initial time (t=0)
    for i in range(boundary_points * 2):
        x_val = np.random.uniform(0, 1)
        y_val = np.random.uniform(0, 1)
        training_points['initial'].append({
            't': 0.0,
            'x': float(x_val),
            'y': float(y_val)
        })
    
    # Sample data points for supervised learning
    # Select a subset of the actual data for supervision
    # This helps guide the physics-based learning
    data_points = boundary_points * 10
    
    # Random sampling from the dataset
    t_indices = np.random.randint(0, len(t), data_points)
    y_indices = np.random.randint(0, len(y), data_points)
    x_indices = np.random.randint(0, len(x), data_points)
    
    for i in range(data_points):
        t_idx = t_indices[i]
        y_idx = y_indices[i]
        x_idx = x_indices[i]
        
        training_points['data'].append({
            't': float(t_norm[t_idx]),
            'x': float(x_norm[x_idx]),
            'y': float(y_norm[y_idx]),
            'h': float(data['water_depth'][t_idx, y_idx, x_idx]),
            'u': float(data['velocity_x'][t_idx, y_idx, x_idx]),
            'v': float(data['velocity_y'][t_idx, y_idx, x_idx])
        })
    
    # Save training points
    training_data = {
        'points': training_points,
        'normalization': {
            't': {'min': float(t[0]), 'max': float(t[-1])},
            'x': {'min': float(x[0]), 'max': float(x[-1])},
            'y': {'min': float(y[0]), 'max': float(y[-1])}
        },
        'metadata': metadata
    }
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Training points saved to: {output_path}")
    
    # Create visualization of sampled points
    create_points_visualization(training_points, output_path.replace('.json', '_visualization.png'))
    
    return output_path

def create_points_visualization(training_points, output_path):
    """
    Create visualization of sampled training points.
    
    Args:
        training_points: Dictionary of training points
        output_path: Path to save visualization
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot interior points
    interior_t = [p['t'] for p in training_points['interior']]
    interior_x = [p['x'] for p in training_points['interior']]
    interior_y = [p['y'] for p in training_points['interior']]
    
    axes[0].scatter(interior_x, interior_y, alpha=0.5, s=5)
    axes[0].set_title('Interior Points (x-y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    
    # Plot boundary points
    boundary_t = [p['t'] for p in training_points['boundary']]
    boundary_x = [p['x'] for p in training_points['boundary']]
    boundary_y = [p['y'] for p in training_points['boundary']]
    
    # Color by boundary
    boundary_colors = []
    for p in training_points['boundary']:
        if p['boundary'] == 'left':
            boundary_colors.append('red')
        elif p['boundary'] == 'right':
            boundary_colors.append('blue')
        elif p['boundary'] == 'bottom':
            boundary_colors.append('green')
        elif p['boundary'] == 'top':
            boundary_colors.append('purple')
    
    axes[1].scatter(boundary_x, boundary_y, c=boundary_colors, alpha=0.5, s=5)
    axes[1].set_title('Boundary Points (x-y)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    
    # Plot initial points
    initial_x = [p['x'] for p in training_points['initial']]
    initial_y = [p['y'] for p in training_points['initial']]
    
    axes[2].scatter(initial_x, initial_y, alpha=0.5, s=5, c='orange')
    axes[2].set_title('Initial Condition Points (x-y at t=0)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    
    # Plot data points
    data_x = [p['x'] for p in training_points['data']]
    data_y = [p['y'] for p in training_points['data']]
    data_t = [p['t'] for p in training_points['data']]
    
    scatter = axes[3].scatter(data_x, data_y, c=data_t, alpha=0.5, s=5, cmap='viridis')
    axes[3].set_title('Data Points (x-y, colored by t)')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    axes[3].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[3], label='Time')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Points visualization saved to: {output_path}")

def process_simulation_for_pinn(dataset_path, output_dir):
    """
    Process a simulation dataset for PINN training.
    
    Args:
        dataset_path: Path to ANUGA netCDF output file
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with paths to processed files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name for output files
    base_name = Path(dataset_path).stem
    
    # Paths for output files
    swe_data_path = os.path.join(output_dir, f"{base_name}_swe_data.h5")
    boundary_path = os.path.join(output_dir, f"{base_name}_boundaries.nc")
    training_points_path = os.path.join(output_dir, f"{base_name}_training_points.json")
    
    # Step 1: Extract SWE variables
    extract_shallow_water_variables(dataset_path, swe_data_path)
    
    # Step 2: Extract boundary conditions
    extract_boundary_conditions(dataset_path, boundary_path)
    
    # Step 3: Create training points
    create_training_points(swe_data_path, training_points_path)
    
    return {
        'swe_data': swe_data_path,
        'boundaries': boundary_path,
        'training_points': training_points_path
    }

def process_multiple_simulations(simulation_dir, output_dir):
    """
    Process multiple simulation datasets for PINN training.
    
    Args:
        simulation_dir: Directory containing simulation datasets
        output_dir: Directory to save processed data
        
    Returns:
        List of dictionaries with paths to processed files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all netCDF files
    simulation_files = glob.glob(os.path.join(simulation_dir, "**", "*.nc"), recursive=True)
    
    if not simulation_files:
        logger.warning(f"No simulation files found in {simulation_dir}")
        return []
    
    logger.info(f"Found {len(simulation_files)} simulation files")
    
    # Process each simulation
    results = []
    for sim_file in tqdm(simulation_files, desc="Processing simulations"):
        # Create a subdirectory for this simulation
        sim_name = Path(sim_file).stem
        sim_output_dir = os.path.join(output_dir, sim_name)
        
        try:
            result = process_simulation_for_pinn(sim_file, sim_output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {sim_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate a summary file
    summary = {
        'total_simulations': len(simulation_files),
        'processed_simulations': len(results),
        'processing_date': str(datetime.now()),
        'simulations': []
    }
    
    for i, result in enumerate(results):
        summary['simulations'].append({
            'index': i,
            'swe_data': result['swe_data'],
            'boundaries': result['boundaries'],
            'training_points': result['training_points']
        })
    
    # Save summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing summary saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Prepare ANUGA outputs for PINN training')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to ANUGA netCDF file or directory containing simulation results')
    parser.add_argument('--output_dir', type=str, default='pinn_data',
                        help='Directory to save processed data')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Factor to downsample the spatial and temporal dimensions')
    
    args = parser.parse_args()
    
    # Determine if input is a file or directory
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    
    if os.path.isfile(input_path):
        # Process a single file
        logger.info(f"Processing single file: {input_path}")
        sim_output_dir = os.path.join(output_dir, Path(input_path).stem)
        process_simulation_for_pinn(input_path, sim_output_dir)
    elif os.path.isdir(input_path):
        # Process all files in directory
        logger.info(f"Processing all simulations in: {input_path}")
        process_multiple_simulations(input_path, output_dir)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    logger.info("Processing completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 