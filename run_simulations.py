#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run multiple ANUGA simulations with varying parameters to generate a dataset
for training Physics-Informed Neural Networks (PINNs).

This script:
1. Creates a set of different flood scenarios by varying boundary conditions and other parameters
2. Runs ANUGA simulations for each scenario
3. Collects results in a structured format for further processing
"""

import os
import sys
import yaml
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import shutil

# Add the script directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Setup logging
log_file = f"anuga_simulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing ANUGA with error handling
try:
    import anuga
except ImportError as e:
    logger.error(f"Failed to import ANUGA: {str(e)}")
    logger.error("Please ensure ANUGA is installed. You can install it using:")
    logger.error("pip install anuga==3.1.9")
    logger.error("Note: If you encounter dependency parsing errors with ANUGA, you may need to install it with:")
    logger.error("pip install anuga==3.1.9 --no-deps")
    logger.error("And then manually install its dependencies.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error importing ANUGA: {str(e)}")
    sys.exit(1)

def load_config(config_path):
    """
    Load simulation configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config

def create_output_directory(output_dir, clean=False):
    """
    Create output directory structure for simulation results.
    
    Args:
        output_dir: Base directory for outputs
        clean: If True, remove existing directory first
        
    Returns:
        Path to created directory
    """
    output_path = Path(output_dir)
    
    # Remove existing directory if requested
    if clean and output_path.exists():
        logger.info(f"Cleaning output directory: {output_path}")
        shutil.rmtree(output_path)
    
    # Create directories
    output_path.mkdir(exist_ok=True, parents=True)
    (output_path / "configs").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    (output_path / "visualizations").mkdir(exist_ok=True)
    
    logger.info(f"Created output directory structure at: {output_path}")
    return output_path

def generate_simulation_scenarios(base_config, num_scenarios):
    """
    Generate multiple simulation scenarios by varying parameters.
    
    Args:
        base_config: Base configuration dictionary
        num_scenarios: Number of scenarios to generate
        
    Returns:
        List of scenario configurations
    """
    scenarios = []
    
    # Extract scenario type from base config
    scenario_type = base_config.get("flood_scenario", {}).get("type", "river_flooding")
    
    # Parameter ranges for different scenario types
    param_ranges = {
        "river_flooding": {
            "inflow_depth": (2.0, 5.0),  # Range for river depth (m)
            "inflow_velocity": (2.0, 4.0),  # Range for river velocity (m/s)
            "rainfall_rate": (0.0, 20.0),  # Range for rainfall rate (mm/hr)
            "manning_n": (0.025, 0.05)  # Range for Manning's n
        },
        "urban_flooding": {
            "rainfall_rate": (20.0, 100.0),  # Range for heavy rainfall (mm/hr)
            "drain_capacity": (10.0, 30.0),  # Range for drainage capacity (mm/hr)
            "impervious_fraction": (0.6, 0.9),  # Range for impervious surface fraction
            "manning_n": (0.01, 0.03)  # Range for Manning's n
        },
        "coastal_surge": {
            "surge_height": (1.0, 3.0),  # Range for surge height (m)
            "wave_period": (5.0, 15.0),  # Range for wave period (s)
            "tide_level": (0.0, 1.0),  # Range for tide level (m)
            "manning_n": (0.02, 0.04)  # Range for Manning's n
        }
    }
    
    # Default to river_flooding if scenario type is not recognized
    if scenario_type not in param_ranges:
        logger.warning(f"Unknown scenario type: {scenario_type}. Using river_flooding parameters.")
        scenario_type = "river_flooding"
    
    # Get parameter ranges for this scenario type
    ranges = param_ranges[scenario_type]
    
    # Generate scenarios
    for i in range(num_scenarios):
        # Create deep copy of base config
        scenario_config = dict(base_config)
        
        # Vary parameters based on scenario type
        params = {}
        for param, (min_val, max_val) in ranges.items():
            # Use Latin Hypercube Sampling for better coverage
            # For simplicity, we're using random values here
            params[param] = min_val + (max_val - min_val) * np.random.random()
        
        # Add variation to size and resolution
        variation_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
        
        # Update scenario-specific parameters
        if scenario_type == "river_flooding":
            scenario_config["boundary_conditions"] = {
                "upstream": {
                    "type": "flow",
                    "depth": params["inflow_depth"],
                    "velocity": params["inflow_velocity"]
                },
                "downstream": {
                    "type": "stage",
                    "value": params["inflow_depth"] * 0.5  # Lower stage at downstream
                },
                "rainfall": {
                    "rate": params["rainfall_rate"]
                }
            }
            scenario_config["domain"] = {
                "x_size": 1000 * variation_factor,
                "y_size": 400 * variation_factor,
                "resolution": 10 * (1.5 - variation_factor)  # Inverse relationship with size
            }
        
        elif scenario_type == "urban_flooding":
            scenario_config["boundary_conditions"] = {
                "rainfall": {
                    "rate": params["rainfall_rate"]
                },
                "drainage": {
                    "capacity": params["drain_capacity"]
                }
            }
            scenario_config["domain"] = {
                "x_size": 500 * variation_factor,
                "y_size": 500 * variation_factor,
                "resolution": 5 * (1.5 - variation_factor)
            }
            scenario_config["urban_parameters"] = {
                "impervious_fraction": params["impervious_fraction"]
            }
        
        elif scenario_type == "coastal_surge":
            scenario_config["boundary_conditions"] = {
                "ocean_boundary": {
                    "type": "tide_plus_surge",
                    "surge_height": params["surge_height"],
                    "wave_period": params["wave_period"],
                    "tide_level": params["tide_level"]
                }
            }
            scenario_config["domain"] = {
                "x_size": 2000 * variation_factor,
                "y_size": 1000 * variation_factor,
                "resolution": 20 * (1.5 - variation_factor)
            }
        
        # Common parameters
        scenario_config["friction"] = {
            "manning_n": params["manning_n"]
        }
        
        # Add duration variation
        scenario_config["simulation"] = {
            "duration": 3600 * (0.5 + np.random.random()),  # 0.5 to 1.5 hours
            "output_interval": 60  # 1 minute
        }
        
        # Add unique ID and description
        scenario_config["id"] = f"{scenario_type}_{i+1:03d}"
        scenario_config["description"] = f"{scenario_type.replace('_', ' ').title()} scenario {i+1} with {list(params.keys())[0]} = {params[list(params.keys())[0]]:.2f}"
        
        scenarios.append(scenario_config)
    
    logger.info(f"Generated {len(scenarios)} {scenario_type} scenarios")
    return scenarios

def create_dem(config, output_file=None):
    """
    Create a Digital Elevation Model (DEM) for the simulation.
    
    Args:
        config: Configuration dictionary with domain information
        output_file: Path to save the DEM (optional)
        
    Returns:
        Tuple of (x, y, elevation) arrays
    """
    # Extract domain parameters
    x_size = config["domain"].get("x_size", 1000)
    y_size = config["domain"].get("y_size", 400)
    resolution = config["domain"].get("resolution", 10)
    scenario_type = config["flood_scenario"].get("type", "river_flooding")
    
    # Create coordinate arrays
    x = np.arange(0, x_size + resolution, resolution)
    y = np.arange(0, y_size + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation based on scenario type
    if scenario_type == "river_flooding":
        # Create a valley with a river channel
        # Base elevation that rises from 0 at x=0 to a small value at x=x_max
        base_elevation = 0.001 * X
        
        # Add valley walls
        valley_width = 0.5 * y_size
        valley_center = 0.5 * y_size
        valley_slope = 0.1
        valley = valley_slope * np.abs(Y - valley_center)
        valley = np.where(valley > 20, 20, valley)  # Limit max height
        
        # Create river channel
        channel_width = 0.1 * y_size
        channel_center = valley_center
        channel_depth = 2.0
        channel = channel_depth * np.exp(-0.5 * ((Y - channel_center) / (0.25 * channel_width))**2)
        
        # Combine components
        elevation = base_elevation + valley - channel
        
    elif scenario_type == "urban_flooding":
        # Create urban terrain with gentle slope and buildings represented as obstacles
        # Base elevation that slopes from corner to corner
        base_elevation = 0.001 * (X + Y)
        
        # Add random variations to represent urban terrain
        np.random.seed(42)  # For reproducibility
        urban_roughness = 0.5 * np.random.rand(X.shape[0], X.shape[1])
        
        # Add some buildings as raised areas
        buildings = np.zeros_like(X)
        n_buildings = 20
        for _ in range(n_buildings):
            bldg_x = np.random.randint(0, X.shape[1])
            bldg_y = np.random.randint(0, X.shape[0])
            bldg_size_x = np.random.randint(2, 8)
            bldg_size_y = np.random.randint(2, 8)
            bldg_height = np.random.uniform(3, 10)
            
            x_min = max(0, bldg_x - bldg_size_x)
            x_max = min(X.shape[1], bldg_x + bldg_size_x)
            y_min = max(0, bldg_y - bldg_size_y)
            y_max = min(X.shape[0], bldg_y + bldg_size_y)
            
            buildings[y_min:y_max, x_min:x_max] = bldg_height
        
        # Combine components
        elevation = base_elevation + urban_roughness + buildings
        
    elif scenario_type == "coastal_surge":
        # Create coastal terrain with beach and inland areas
        # Distance from coast (assumed to be at x=0)
        coastal_distance = X
        
        # Beach profile: sharp rise near coast, then gradual slope inland
        beach_profile = 0.05 * coastal_distance
        beach_profile = np.where(coastal_distance < 100, 0.2 * coastal_distance, beach_profile)
        
        # Add some dunes near the coast
        dune_position = 150
        dune_width = 50
        dune_height = 3.0
        dunes = dune_height * np.exp(-0.5 * ((X - dune_position) / (0.5 * dune_width))**2)
        
        # Add random variations
        np.random.seed(42)  # For reproducibility
        terrain_roughness = 0.2 * np.random.rand(X.shape[0], X.shape[1])
        
        # Combine components
        elevation = beach_profile + dunes + terrain_roughness
        
    else:
        # Default: simple sloping terrain
        logger.warning(f"Unknown scenario type: {scenario_type}. Creating simple sloping terrain.")
        elevation = 0.001 * (X + Y)
    
    # Ensure minimum elevation is zero
    elevation = np.maximum(elevation, 0)
    
    # Save DEM if output file is provided
    if output_file:
        dem_data = {
            'x': x,
            'y': y,
            'elevation': elevation
        }
        np.savez(output_file, **dem_data)
        logger.info(f"Saved DEM to: {output_file}")
    
    return x, y, elevation

def setup_anuga_domain(x, y, elevation, config):
    """
    Set up the ANUGA domain for simulation.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        elevation: 2D array of elevation values
        config: Configuration dictionary
        
    Returns:
        ANUGA domain object
    """
    # Extract parameters
    resolution = config["domain"].get("resolution", 10)
    scenario_id = config.get("id", "scenario")
    
    # Create bounding polygon from extent
    bounding_polygon = [[x.min(), y.min()], 
                         [x.max(), y.min()], 
                         [x.max(), y.max()], 
                         [x.min(), y.max()]]
    
    # Create domain
    domain = anuga.Domain(bounding_polygon, 
                          mesh_filename=f"{scenario_id}_mesh.msh",
                          use_cache=False,
                          verbose=False)
    
    # Set mesh resolution
    domain.set_mesh_resolution(resolution)
    
    # Set quantities
    domain.set_quantity('elevation', lambda x, y: anuga.interpolate.Interpolate(x, y, elevation))
    domain.set_quantity('friction', config["friction"].get("manning_n", 0.03))
    domain.set_quantity('stage', 0.0)  # Initialize stage as dry
    
    logger.info(f"Created ANUGA domain with resolution {resolution}m")
    return domain

def set_boundary_conditions(domain, config):
    """
    Set boundary conditions for the ANUGA domain.
    
    Args:
        domain: ANUGA domain object
        config: Configuration dictionary
        
    Returns:
        Updated domain with boundary conditions
    """
    # Extract parameters
    scenario_type = config["flood_scenario"].get("type", "river_flooding")
    boundary_config = config.get("boundary_conditions", {})
    
    # Set boundary conditions based on scenario type
    if scenario_type == "river_flooding":
        # Set upstream inflow
        upstream_depth = boundary_config.get("upstream", {}).get("depth", 2.0)
        upstream_velocity = boundary_config.get("upstream", {}).get("velocity", 2.0)
        downstream_stage = boundary_config.get("downstream", {}).get("value", 1.0)
        
        Br = anuga.Reflective_boundary(domain)
        Bu = anuga.Dirichlet_boundary([upstream_depth, upstream_depth * upstream_velocity, 0])
        Bd = anuga.Dirichlet_boundary([downstream_stage, 0, 0])
        
        # Apply boundaries (assuming rectangular domain with upstream at x=0, downstream at x=L)
        domain.set_boundary({'left': Bu, 'right': Bd, 'top': Br, 'bottom': Br})
        
        # Add rainfall if specified
        rainfall_rate = boundary_config.get("rainfall", {}).get("rate", 0.0)
        if rainfall_rate > 0:
            # Convert mm/hr to m/s
            rate_m_s = rainfall_rate / 1000 / 3600
            rain = anuga.Rate_operator(domain, rate=rate_m_s)
            
            logger.info(f"Added rainfall at rate {rainfall_rate} mm/hr")
        
    elif scenario_type == "urban_flooding":
        # Reflective boundaries for urban domain edges
        Br = anuga.Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
        
        # Add rainfall (main driver for urban flooding)
        rainfall_rate = boundary_config.get("rainfall", {}).get("rate", 50.0)
        # Convert mm/hr to m/s
        rate_m_s = rainfall_rate / 1000 / 3600
        rain = anuga.Rate_operator(domain, rate=rate_m_s)
        
        # Add drainage (negative rate) if specified
        drain_capacity = boundary_config.get("drainage", {}).get("capacity", 0.0)
        if drain_capacity > 0:
            # Convert mm/hr to m/s and make negative for drainage
            drain_rate_m_s = -drain_capacity / 1000 / 3600
            # Apply drainage only in low-lying areas
            def drainage_function(x, y, t):
                return drain_rate_m_s
            
            # This is a simplified approximation - real drainage would be more complex
            drain = anuga.Rate_operator(domain, rate=drainage_function)
            
            logger.info(f"Added drainage at capacity {drain_capacity} mm/hr")
        
    elif scenario_type == "coastal_surge":
        # Parameters for coastal boundary
        surge_config = boundary_config.get("ocean_boundary", {})
        surge_height = surge_config.get("surge_height", 2.0)
        wave_period = surge_config.get("wave_period", 10.0)
        tide_level = surge_config.get("tide_level", 0.5)
        
        # Time-varying ocean boundary to simulate tide + surge
        def ocean_stage(t):
            # Base tide level
            tide = tide_level
            # Add storm surge (simplified as a sine wave)
            surge = surge_height * np.sin(2 * np.pi * t / wave_period)
            return max(0, tide + surge)
        
        # Ocean at the left boundary (x=0), reflective on other boundaries
        Bs = anuga.Time_boundary(domain=domain, 
                                 function=lambda t: [ocean_stage(t), 0, 0])
        Br = anuga.Reflective_boundary(domain)
        
        # Apply boundaries
        domain.set_boundary({'left': Bs, 'right': Br, 'top': Br, 'bottom': Br})
    
    else:
        logger.warning(f"Unknown scenario type: {scenario_type}. Setting all boundaries as reflective.")
        Br = anuga.Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    
    logger.info(f"Set boundary conditions for {scenario_type} scenario")
    return domain

def run_simulation(domain, config, output_file):
    """
    Run the ANUGA simulation.
    
    Args:
        domain: ANUGA domain object
        config: Configuration dictionary
        output_file: Path to save the simulation results
        
    Returns:
        Path to the results file
    """
    # Extract parameters
    duration = config["simulation"].get("duration", 3600)
    output_interval = config["simulation"].get("output_interval", 60)
    
    # Set up output
    domain.set_name(output_file)
    domain.format = 'netcdf'  # Use netCDF format for output
    
    # Run simulation
    logger.info(f"Starting simulation for duration {duration}s with output every {output_interval}s")
    
    for t in domain.evolve(yieldstep=output_interval, duration=duration):
        logger.info(f"Simulation time: {t:.1f}s")
    
    logger.info(f"Simulation completed, results saved to {output_file}.nc")
    return f"{output_file}.nc"

def create_visualization(result_file, output_dir):
    """
    Create visualizations from simulation results.
    
    Args:
        result_file: Path to the simulation results (netCDF)
        output_dir: Directory to save visualizations
        
    Returns:
        List of created visualization files
    """
    import xarray as xr
    
    logger.info(f"Creating visualizations for {result_file}")
    
    # Load results
    ds = xr.open_dataset(result_file)
    scenario_id = Path(result_file).stem
    
    # Create output directory
    vis_dir = Path(output_dir) / "visualizations" / scenario_id
    vis_dir.mkdir(exist_ok=True)
    
    vis_files = []
    
    # Plot water depth at different times
    times = ds.time.values
    num_plots = min(6, len(times))
    plot_times = np.linspace(0, len(times) - 1, num_plots).astype(int)
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, time_idx in enumerate(plot_times):
        if i >= len(axes):
            break
            
        t = times[time_idx]
        ax = axes[i]
        
        # Plot water depth
        im = ax.pcolormesh(ds.x, ds.y, ds.water_depth.isel(time=time_idx), 
                           cmap='Blues', vmin=0, vmax=ds.water_depth.max())
        ax.set_title(f"Time: {t:.0f}s")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label='Water Depth (m)')
    
    plt.tight_layout()
    depth_file = vis_dir / f"{scenario_id}_water_depth.png"
    plt.savefig(depth_file, dpi=150)
    plt.close()
    vis_files.append(str(depth_file))
    
    # Plot maximum water depth
    plt.figure(figsize=(10, 8))
    max_depth = ds.water_depth.max(dim='time')
    im = plt.pcolormesh(ds.x, ds.y, max_depth, cmap='Blues', vmin=0)
    plt.colorbar(im, label='Maximum Water Depth (m)')
    plt.title(f"Maximum Water Depth - {scenario_id}")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    
    max_depth_file = vis_dir / f"{scenario_id}_max_depth.png"
    plt.savefig(max_depth_file, dpi=150)
    plt.close()
    vis_files.append(str(max_depth_file))
    
    # Plot time series at a sample point
    mid_x = len(ds.x) // 2
    mid_y = len(ds.y) // 2
    
    plt.figure(figsize=(10, 6))
    ds.water_depth.isel(y=mid_y, x=mid_x).plot()
    plt.title(f"Water Depth Time Series at Sample Point - {scenario_id}")
    plt.xlabel('Time (s)')
    plt.ylabel('Water Depth (m)')
    plt.grid(True)
    
    timeseries_file = vis_dir / f"{scenario_id}_timeseries.png"
    plt.savefig(timeseries_file, dpi=150)
    plt.close()
    vis_files.append(str(timeseries_file))
    
    # Close dataset
    ds.close()
    
    logger.info(f"Created {len(vis_files)} visualizations")
    return vis_files

def run_scenario(scenario_config, output_dir):
    """
    Run a single simulation scenario.
    
    Args:
        scenario_config: Configuration dictionary for this scenario
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary with scenario results and metadata
    """
    # Create scenario ID and paths
    scenario_id = scenario_config.get("id", f"scenario_{np.random.randint(10000)}")
    logger.info(f"Starting scenario {scenario_id}")
    
    # Create output directories
    scenario_dir = Path(output_dir) / "results" / scenario_id
    scenario_dir.mkdir(exist_ok=True, parents=True)
    
    # Save scenario configuration
    config_path = Path(output_dir) / "configs" / f"{scenario_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=False)
    
    # Create DEM
    dem_path = scenario_dir / f"{scenario_id}_dem.npz"
    x, y, elevation = create_dem(scenario_config, dem_path)
    
    # Setup domain
    domain = setup_anuga_domain(x, y, elevation, scenario_config)
    
    # Set boundary conditions
    domain = set_boundary_conditions(domain, scenario_config)
    
    # Run simulation
    result_path = scenario_dir / scenario_id
    result_file = run_simulation(domain, scenario_config, str(result_path))
    
    # Create visualizations
    visualization_files = create_visualization(result_file, output_dir)
    
    # Return metadata
    result = {
        "scenario_id": scenario_id,
        "config_file": str(config_path),
        "dem_file": str(dem_path),
        "result_file": result_file,
        "visualization_files": visualization_files,
        "scenario_type": scenario_config["flood_scenario"].get("type", "unknown"),
        "description": scenario_config.get("description", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Completed scenario {scenario_id}")
    return result

def run_scenarios_parallel(scenarios, output_dir, max_workers=None):
    """
    Run multiple simulation scenarios in parallel.
    
    Args:
        scenarios: List of scenario configurations
        output_dir: Base directory for outputs
        max_workers: Maximum number of worker processes
        
    Returns:
        List of results for all scenarios
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Running {len(scenarios)} scenarios with {max_workers} workers")
    
    # Create a pool of workers
    pool = mp.Pool(processes=max_workers)
    
    # Run scenarios
    results = []
    for scenario in tqdm(scenarios, desc="Running scenarios"):
        # Use apply_async to run in parallel
        result = pool.apply_async(run_scenario, args=(scenario, output_dir))
        results.append(result)
    
    # Close pool and wait for completion
    pool.close()
    pool.join()
    
    # Get actual results
    scenario_results = [r.get() for r in results]
    
    # Create summary
    summary = {
        "total_scenarios": len(scenarios),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": scenario_results
    }
    
    # Save summary
    summary_path = Path(output_dir) / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Completed all {len(scenarios)} scenarios")
    logger.info(f"Summary saved to {summary_path}")
    
    return scenario_results

def main():
    parser = argparse.ArgumentParser(description='Run ANUGA simulations for flood scenarios')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--output_dir', type=str, default='simulations',
                        help='Directory to save simulation results')
    parser.add_argument('--num_scenarios', type=int, default=5,
                        help='Number of simulation scenarios to generate')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directory before running')
    
    args = parser.parse_args()
    
    try:
        # Load base configuration
        base_config = load_config(args.config)
        
        # Create output directory
        output_dir = create_output_directory(args.output_dir, args.clean)
        
        # Generate scenarios
        scenarios = generate_simulation_scenarios(base_config, args.num_scenarios)
        
        # Run scenarios
        results = run_scenarios_parallel(scenarios, output_dir, args.max_workers)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running simulations: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 