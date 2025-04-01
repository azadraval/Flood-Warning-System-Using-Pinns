#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ANUGA-based hydrodynamic simulator for flood scenarios.
This module provides functions to set up, run, and analyze ANUGA simulations
for generating training data for the flood warning system.
"""

import os
import numpy as np
import anuga
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logger
logger = logging.getLogger(__name__)

def load_terrain(dem_path=None, x_size=1000, y_size=1000, resolution=10):
    """
    Load terrain data from DEM or create synthetic terrain.
    
    Args:
        dem_path: Path to Digital Elevation Model file (optional)
        x_size: Width of domain in meters (default: 1000)
        y_size: Length of domain in meters (default: 1000)
        resolution: Spatial resolution in meters (default: 10)
        
    Returns:
        x, y: Coordinate arrays
        elevation: Elevation array
    """
    if dem_path and os.path.exists(dem_path):
        # Load DEM data based on file type
        if dem_path.endswith('.npy'):
            # Load from NumPy array
            dem_data = np.load(dem_path)
            return dem_data['x'], dem_data['y'], dem_data['elevation']
        
        elif dem_path.endswith('.tif') or dem_path.endswith('.tiff'):
            # Load from GeoTIFF using rasterio
            import rasterio
            with rasterio.open(dem_path) as src:
                elevation = src.read(1)
                # Create coordinate arrays
                height, width = elevation.shape
                x = np.linspace(src.bounds.left, src.bounds.right, width)
                y = np.linspace(src.bounds.bottom, src.bounds.top, height)
                return x, y, elevation
                
        else:
            logger.warning(f"Unsupported DEM format: {dem_path}")
    
    # Generate synthetic topography with a river channel
    logger.info("Creating synthetic terrain with river channel")
    x = np.linspace(0, x_size, int(x_size / resolution))
    y = np.linspace(0, y_size, int(y_size / resolution))
    X, Y = np.meshgrid(x, y)
    
    # Create a sloping terrain with a river channel
    elevation = 20 - 0.01 * Y  # General slope
    
    # Add a river channel
    channel_width = 50
    channel_depth = 2
    channel_center = x_size / 2
    
    # Create a meandering river
    river_x = channel_center + 30 * np.sin(Y * 0.01)
    
    # Apply river channel to the elevation
    for i in range(len(y)):
        river_mask = np.abs(X[i, :] - river_x[i]) < channel_width / 2
        elevation[i, river_mask] -= channel_depth
    
    return x, y, elevation

def setup_anuga_domain(x, y, elevation, mesh_resolution=10.0, domain_name="flood_domain"):
    """
    Set up the ANUGA computational domain.
    
    Args:
        x, y: Coordinate arrays
        elevation: Elevation array
        mesh_resolution: Triangle mesh size (default: 10.0)
        domain_name: Name for the domain (default: "flood_domain")
        
    Returns:
        domain: ANUGA domain object
    """
    logger.info(f"Setting up ANUGA domain with mesh resolution {mesh_resolution}m")
    
    # Create bounding polygon for domain
    boundary_polygon = np.array([
        [x.min(), y.min()], 
        [x.max(), y.min()], 
        [x.max(), y.max()], 
        [x.min(), y.max()]
    ])
    
    # Create the domain with boundary
    domain = anuga.create_domain_from_regions(
        boundary_polygon,
        boundary_tags={'bottom': [0], 'right': [1], 'top': [2], 'left': [3]},
        maximum_triangle_area=mesh_resolution**2,
        mesh_filename=f'{domain_name}.msh',
        use_cache=False
    )
    
    # Set the topography using elevation function
    def elevation_function(x_in, y_in):
        """Return elevation at (x_in, y_in) interpolated from grid."""
        # Simple linear interpolation
        i = np.searchsorted(x, x_in) - 1
        j = np.searchsorted(y, y_in) - 1
        
        # Handle edge cases
        i = np.clip(i, 0, len(x) - 2)
        j = np.clip(j, 0, len(y) - 2)
        
        # Calculate interpolation parameters
        alpha = (x_in - x[i]) / (x[i+1] - x[i])
        beta = (y_in - y[j]) / (y[j+1] - y[j])
        
        # Bilinear interpolation
        z = (1-alpha)*(1-beta)*elevation[j, i] + \
            alpha*(1-beta)*elevation[j, i+1] + \
            (1-alpha)*beta*elevation[j+1, i] + \
            alpha*beta*elevation[j+1, i+1]
        
        return z
    
    # Set quantities
    domain.set_quantity('elevation', elevation_function)
    domain.set_quantity('friction', 0.03)  # Manning's friction coefficient (constant)
    domain.set_quantity('stage', elevation_function)  # Initial water level = elevation
    
    return domain

def define_boundary_conditions(domain, scenario_config):
    """
    Define boundary conditions based on the scenario configuration.
    
    Args:
        domain: ANUGA domain object
        scenario_config: Dictionary with scenario configuration
        
    Returns:
        domain: Updated ANUGA domain with boundary conditions
    """
    scenario_type = scenario_config['type']
    logger.info(f"Setting boundary conditions for {scenario_type} scenario")
    
    if scenario_type == 'river_flooding':
        # Extract parameters
        params = scenario_config.get('parameters', {})
        base_level = params.get('base_level', 22.0)
        peak_height = params.get('peak_height', 2.0)
        peak_time = params.get('peak_time', 10.0)
        rising_duration = params.get('rising_duration', 5.0)
        falling_rate = params.get('falling_rate', 0.2)
        
        # Inflow from upstream with a flood hydrograph
        def inflow_stage(t):
            """Time-varying upstream water level."""
            if t < rising_duration:
                # Rising limb
                return base_level + peak_height * t / rising_duration
            elif t < peak_time:
                # Peak flood
                return base_level + peak_height
            else:
                # Falling limb
                return max(base_level, base_level + peak_height - falling_rate * (t - peak_time))
        
        # Set boundary conditions
        Bi = anuga.Dirichlet_boundary([inflow_stage, 0, 0])  # Stage, x-momentum, y-momentum
        Br = anuga.Reflective_boundary()  # Solid walls
        Bo = anuga.Dirichlet_boundary([-9999, 0, 0])  # Outflow condition
        
        domain.set_boundary({'left': Bi, 'right': Bo, 'top': Br, 'bottom': Br})
    
    elif scenario_type == 'urban_flooding':
        # Extract parameters
        params = scenario_config.get('parameters', {})
        rain_start = params.get('rain_start', 2.0)
        rain_duration = params.get('rain_duration', 6.0)
        rain_intensity = params.get('rain_intensity', 30.0)  # mm/hr
        
        # Rainfall-induced flooding
        def rainfall(t):
            """Time-varying rainfall intensity (mm/hr converted to m/s)."""
            if rain_start < t < (rain_start + rain_duration):
                return rain_intensity / 1000 / 3600  # Convert to m/s
            else:
                return 0.0
        
        domain.set_rainfall(rainfall)
        
        # Open boundary conditions
        Br = anuga.Reflective_boundary()  # Solid walls
        Bo = anuga.Dirichlet_boundary([-9999, 0, 0])  # Outflow condition
        
        domain.set_boundary({'left': Br, 'right': Bo, 'top': Br, 'bottom': Bo})
        
    elif scenario_type == 'coastal_surge':
        # Extract parameters
        params = scenario_config.get('parameters', {})
        base_level = params.get('base_level', 0.0)
        surge_height = params.get('surge_height', 3.0)
        surge_rise_time = params.get('surge_rise_time', 10.0)
        surge_duration = params.get('surge_duration', 5.0)
        surge_fall_rate = params.get('surge_fall_rate', 0.3)
        
        # Storm surge from the ocean
        def surge_stage(t):
            """Time-varying storm surge."""
            if t < surge_rise_time:
                # Rising surge
                return base_level + surge_height * t / surge_rise_time
            elif t < (surge_rise_time + surge_duration):
                # Peak surge
                return base_level + surge_height
            else:
                # Falling surge
                return max(base_level, base_level + surge_height - 
                          surge_fall_rate * (t - (surge_rise_time + surge_duration)))
        
        Bs = anuga.Dirichlet_boundary([surge_stage, 0, 0])  # Surge from ocean
        Br = anuga.Reflective_boundary()  # Solid walls
        
        domain.set_boundary({'bottom': Bs, 'right': Br, 'top': Br, 'left': Br})
    
    elif scenario_type == 'dam_break':
        # Extract parameters
        params = scenario_config.get('parameters', {})
        breach_time = params.get('breach_time', 5.0)
        water_level = params.get('water_level', 5.0)
        
        # Create initial water level representing the dam
        polygons = params.get('polygons', [])
        
        if polygons:
            for polygon in polygons:
                # Add water behind dam
                dam_poly = np.array(polygon['coords'])
                dam_stage = polygon.get('stage', water_level)
                
                # Set water level in the defined polygon
                domain.set_quantity('stage', dam_stage, polygon=dam_poly)
        else:
            # Default dam location (if no specific polygons provided)
            dam_poly = np.array([
                [x.min() + 0.2 * (x.max() - x.min()), y.min()],
                [x.min() + 0.2 * (x.max() - x.min()), y.max()],
                [x.min() + 0.4 * (x.max() - x.min()), y.max()],
                [x.min() + 0.4 * (x.max() - x.min()), y.min()]
            ])
            domain.set_quantity('stage', water_level, polygon=dam_poly)
        
        # Set reflective boundaries on all sides
        Br = anuga.Reflective_boundary()
        Bo = anuga.Dirichlet_boundary([-9999, 0, 0])  # Outflow condition
        
        # Set breaching of the dam by changing the outflow boundary
        def dam_break_operator(domain):
            t = domain.get_time()
            if t < breach_time:
                pass  # Dam intact
            else:
                pass  # Dam breached
        
        domain.set_boundary({'left': Br, 'right': Bo, 'top': Br, 'bottom': Br})
        domain.set_operator(dam_break_operator)
    
    else:
        logger.warning(f"Unknown scenario type: {scenario_type}. Using default reflective boundaries.")
        # Default to reflective boundaries
        Br = anuga.Reflective_boundary()
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    
    return domain

def run_anuga_simulation(domain, duration_hours=24.0, output_timestep_mins=30, 
                         output_quantities=None, output_dir=None):
    """
    Run the ANUGA simulation and save results.
    
    Args:
        domain: ANUGA domain object
        duration_hours: Simulation duration in hours (default: 24.0)
        output_timestep_mins: Time between outputs in minutes (default: 30)
        output_quantities: List of quantities to store (default: None, which means use standard set)
        output_dir: Directory to save output files (default: None)
        
    Returns:
        xarray.Dataset with simulation results
    """
    # Convert time units for ANUGA
    duration_seconds = duration_hours * 3600
    output_timestep_seconds = output_timestep_mins * 60
    
    # Set output quantities
    if output_quantities is None:
        output_quantities = {
            'stage': 2,         # Water surface elevation
            'xmomentum': 2,     # Momentum in x-direction
            'ymomentum': 2,     # Momentum in y-direction
            'elevation': 1      # Bed elevation (stored only once)
        }
    
    # Set up output
    domain.set_name('anuga_simulation')
    domain.set_quantities_to_be_stored(output_quantities)
    
    if output_dir:
        domain.set_datadir(output_dir)
    
    # Run the simulation
    logger.info(f"Starting ANUGA simulation for {duration_hours:.1f} hours")
    progress = tqdm(desc="Simulation progress", total=duration_seconds, unit="s")
    
    last_update = 0
    for t in domain.evolve(yieldstep=output_timestep_seconds, duration=duration_seconds):
        # Update progress bar
        progress.update(t - last_update)
        last_update = t
        logger.debug(f"Simulation time: {t/3600:.2f} hours")
    
    progress.close()
    logger.info("ANUGA simulation completed")
    
    # Process results into xarray format
    return extract_simulation_results(domain)

def extract_simulation_results(domain, grid_size=100):
    """
    Extract simulation results from ANUGA domain.
    
    Args:
        domain: ANUGA domain object
        grid_size: Number of points in x and y directions for regular grid
        
    Returns:
        xarray.Dataset with simulation results
    """
    logger.info("Extracting simulation results to regular grid")
    
    # Get time history
    times = domain.get_time_history()
    
    # Define regular grid for output
    x_min, y_min = domain.geo_reference.get_origin()
    x_coords = np.linspace(x_min, x_min + domain.width, grid_size)
    y_coords = np.linspace(y_min, y_min + domain.height, grid_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Initialize data arrays
    water_depths = []
    velocities = []
    water_levels = []
    
    # Extract data for each time step
    for time_idx, t in enumerate(times):
        # Get quantities
        stage = domain.get_quantity('stage').get_values(interpolation_points=points)
        elevation = domain.get_quantity('elevation').get_values(interpolation_points=points)
        xmom = domain.get_quantity('xmomentum').get_values(interpolation_points=points)
        ymom = domain.get_quantity('ymomentum').get_values(interpolation_points=points)
        
        # Calculate depth and velocity
        depth = np.maximum(stage - elevation, 0.0).reshape(grid_size, grid_size)
        
        # Calculate velocity (avoid division by zero)
        vel_x = np.zeros_like(depth)
        vel_y = np.zeros_like(depth)
        mask = depth > 0.01  # Only calculate velocity where depth is significant
        
        if np.any(mask):
            vel_x[mask] = xmom.reshape(grid_size, grid_size)[mask] / depth[mask]
            vel_y[mask] = ymom.reshape(grid_size, grid_size)[mask] / depth[mask]
        
        velocity = np.sqrt(vel_x**2 + vel_y**2)
        
        # Store results
        water_depths.append(depth)
        velocities.append(velocity)
        water_levels.append(stage.reshape(grid_size, grid_size))
    
    # Convert to datetime format for xarray
    time_coords = [datetime.datetime.now() + datetime.timedelta(seconds=t) for t in times]
    
    # Create xarray dataset
    data_vars = {
        'water_depth': (["time", "y", "x"], np.array(water_depths)),
        'flow_velocity': (["time", "y", "x"], np.array(velocities)),
        'water_level': (["time", "y", "x"], np.array(water_levels)),
        'elevation': (["y", "x"], elevation.reshape(grid_size, grid_size))
    }
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time_coords,
            'x': x_coords,
            'y': y_coords
        },
        attrs={
            'description': "ANUGA hydrodynamic simulation results",
            'creation_date': str(datetime.datetime.now())
        }
    )
    
    return ds

def save_simulation_results(ds, output_path, create_visualization=True):
    """
    Save simulation results to NetCDF file and create visualization.
    
    Args:
        ds: xarray Dataset with simulation results
        output_path: Path to save output file
        create_visualization: Whether to create visualization (default: True)
        
    Returns:
        Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to netCDF
    ds.to_netcdf(output_path)
    logger.info(f"Saved simulation results to {output_path}")
    
    # Create visualization
    if create_visualization:
        viz_path = output_path.replace('.nc', '_visualization.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot maximum water depth
        max_depth = ds.water_depth.max(dim="time")
        max_depth.plot(ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Maximum Water Depth (m)')
        
        # Plot maximum velocity
        max_vel = ds.flow_velocity.max(dim="time")
        max_vel.plot(ax=axes[0, 1], cmap='Reds')
        axes[0, 1].set_title('Maximum Flow Velocity (m/s)')
        
        # Plot time series for center point
        center_x = float(ds.x.mean())
        center_y = float(ds.y.mean())
        
        ds.water_depth.sel(x=center_x, y=center_y, method='nearest').plot(ax=axes[1, 0])
        axes[1, 0].set_title(f'Water Depth Time Series at Center Point')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Depth (m)')
        
        # Time at maximum depth
        max_time_idx = ds.water_depth.argmax(dim="time")
        max_time = ds.time[max_time_idx]
        
        # Plot water depth at maximum time
        ds.water_depth.isel(time=max_time_idx.sel(x=center_x, y=center_y, method='nearest').item()).plot(
            ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title(f'Water Depth at Peak Time')
        
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150)
        plt.close()
        
        logger.info(f"Created visualization at {viz_path}")
    
    return output_path

def run_scenario_simulation(scenario_config, output_dir, dem_path=None):
    """
    Run a full simulation for a specific scenario.
    
    Args:
        scenario_config: Dictionary with scenario configuration
        output_dir: Directory to save output files
        dem_path: Path to DEM file (optional)
        
    Returns:
        Path to saved results
    """
    scenario_name = scenario_config.get('name', f"{scenario_config['type']}_scenario")
    logger.info(f"Running simulation for scenario: {scenario_name}")
    
    # Load terrain
    x_size = scenario_config.get('domain', {}).get('x_size', 1000)
    y_size = scenario_config.get('domain', {}).get('y_size', 1000)
    resolution = scenario_config.get('domain', {}).get('resolution', 10)
    
    x, y, elevation = load_terrain(
        dem_path=dem_path, 
        x_size=x_size, 
        y_size=y_size, 
        resolution=resolution
    )
    
    # Setup domain
    mesh_resolution = scenario_config.get('domain', {}).get('mesh_resolution', 20.0)
    domain = setup_anuga_domain(
        x, y, elevation, 
        mesh_resolution=mesh_resolution,
        domain_name=scenario_name
    )
    
    # Define boundary conditions
    domain = define_boundary_conditions(domain, scenario_config)
    
    # Run simulation
    duration = scenario_config.get('duration', 24.0)
    timestep = scenario_config.get('output_timestep', 30)
    
    # Run the simulation
    results = run_anuga_simulation(
        domain, 
        duration_hours=duration,
        output_timestep_mins=timestep,
        output_dir=os.path.join(output_dir, 'anuga_sww')
    )
    
    # Save results
    output_path = os.path.join(output_dir, f"{scenario_name}.nc")
    save_simulation_results(results, output_path)
    
    # Save configuration alongside results
    config_path = os.path.join(output_dir, f"{scenario_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(scenario_config, f, indent=2)
    
    return output_path

def batch_run_scenarios(scenarios_file, output_dir, dem_dir=None):
    """
    Run multiple scenarios defined in a configuration file.
    
    Args:
        scenarios_file: Path to scenarios configuration file (YAML)
        output_dir: Directory to save output files
        dem_dir: Directory containing DEM files (optional)
        
    Returns:
        List of paths to saved results
    """
    # Load scenarios
    with open(scenarios_file, 'r') as f:
        scenarios = yaml.safe_load(f)
    
    logger.info(f"Running batch of {len(scenarios)} scenarios")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Run each scenario
    for i, scenario in enumerate(scenarios):
        logger.info(f"Running scenario {i+1}/{len(scenarios)}: {scenario.get('name', 'unnamed')}")
        
        # Check for specific DEM
        dem_path = None
        if dem_dir and 'dem_file' in scenario:
            dem_path = os.path.join(dem_dir, scenario['dem_file'])
            if not os.path.exists(dem_path):
                logger.warning(f"DEM file not found: {dem_path}")
                dem_path = None
        
        # Create scenario-specific output directory
        scenario_dir = os.path.join(output_dir, f"scenario_{i+1}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Run scenario
        try:
            result_path = run_scenario_simulation(
                scenario,
                scenario_dir,
                dem_path=dem_path
            )
            results.append(result_path)
        except Exception as e:
            logger.error(f"Error running scenario {i+1}: {e}")
    
    logger.info(f"Completed {len(results)}/{len(scenarios)} scenarios successfully")
    return results

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ANUGA hydrodynamic simulator')
    
    parser.add_argument('--scenario', type=str, default=None,
                        help='Path to scenario configuration file')
    parser.add_argument('--batch', type=str, default=None,
                        help='Path to batch scenarios configuration file')
    parser.add_argument('--output_dir', type=str, default='../simulations',
                        help='Directory to save output files')
    parser.add_argument('--dem', type=str, default=None,
                        help='Path to DEM file')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='Directory containing DEM files')
    
    args = parser.parse_args()
    
    if args.batch:
        # Run batch of scenarios
        batch_run_scenarios(args.batch, args.output_dir, dem_dir=args.dem_dir)
    elif args.scenario:
        # Run single scenario
        with open(args.scenario, 'r') as f:
            scenario_config = yaml.safe_load(f)
            
        run_scenario_simulation(scenario_config, args.output_dir, dem_path=args.dem)
    else:
        logger.error("Either --scenario or --batch must be specified") 