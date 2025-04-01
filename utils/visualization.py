"""
Visualization Utilities for Flood Early Warning System

This module provides utilities for generating maps, plots, and visualizations
from flood forecast data.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib.figure import Figure
import xarray as xr
from typing import Dict, List, Union, Optional, Any, Tuple
import cmocean
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_forecast_maps(
    dataset: xr.Dataset,
    variable: str = "stage",
    time_steps: Optional[List[int]] = None,
    flood_threshold: float = 0.1,
    output_path: str = "forecast_map.png",
    title: Optional[str] = None,
    cmap: str = "cmocean.cm.deep",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> str:
    """
    Create visualization maps from forecast data
    
    Args:
        dataset: xarray Dataset containing model outputs
        variable: Variable to plot (default: "stage" for water depth)
        time_steps: List of time steps to plot (None for all)
        flood_threshold: Threshold value for flood classification (m)
        output_path: Path to save the plot
        title: Title for the plot (default: generated based on data)
        cmap: Colormap name or colormap object
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the figure
        
    Returns:
        Path to the created plot file
    """
    try:
        # Extract coordinates
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            x_coords = dataset.coords["lon"].values
            y_coords = dataset.coords["y"].values
        else:
            raise ValueError("Could not find spatial coordinates in dataset")
        
        # Check if variable exists
        if variable not in dataset.variables:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Check if time dimension exists
        has_time = "time" in dataset.coords
        
        if has_time:
            time_values = dataset.coords["time"].values
            if time_steps is None:
                # Select a subset of time steps if there are many
                if len(time_values) > 9:
                    # Select first, last, and evenly spaced time steps
                    time_steps = [0, len(time_values)//4, len(time_values)//2, 3*len(time_values)//4, -1]
                else:
                    time_steps = list(range(len(time_values)))
            
            # Validate time steps
            time_steps = [t if t >= 0 else len(time_values) + t for t in time_steps]
            time_steps = [t for t in time_steps if 0 <= t < len(time_values)]
            
            if not time_steps:
                logger.warning("No valid time steps provided, using last time step")
                time_steps = [-1]
            
            n_plots = len(time_steps)
        else:
            n_plots = 1
            time_steps = [0]  # Dummy value
        
        # Get colormap
        if isinstance(cmap, str):
            if cmap.startswith("cmocean.cm."):
                cmap_name = cmap.split(".")[-1]
                try:
                    cm = getattr(cmocean.cm, cmap_name)
                except AttributeError:
                    logger.warning(f"Colormap {cmap} not found in cmocean, using viridis")
                    cm = plt.cm.viridis
            else:
                cm = plt.get_cmap(cmap)
        else:
            cm = cmap
        
        # Create figure
        if n_plots == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            # Calculate grid layout
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                   squeeze=False, sharex=True, sharey=True)
            axes = axes.flatten()
        
        # Global min and max for consistent color scale
        if has_time:
            all_data = dataset[variable].sel(time=time_values[time_steps]).values
        else:
            all_data = dataset[variable].values
        
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
        
        # Apply minimum threshold for better visualization
        if vmin >= 0 and vmax > flood_threshold:
            vmin = 0
        
        # Create norm with special emphasis on flood threshold
        if variable == "stage" or variable == "water_depth":
            # Create a colormap with a sharp transition at the flood threshold
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=flood_threshold, vmax=vmax)
        else:
            norm = None
        
        # Plot each time step
        for i, time_idx in enumerate(time_steps):
            if i < len(axes):
                ax = axes[i]
                
                if has_time:
                    time_value = time_values[time_idx]
                    data = dataset[variable].sel(time=time_value).values
                    
                    # Format time for title
                    if isinstance(time_value, np.datetime64):
                        time_str = str(time_value)
                        try:
                            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d %H:%M")
                        except:
                            pass
                    else:
                        time_str = f"t={time_str}"
                else:
                    data = dataset[variable].values
                    time_str = ""
                
                # Plot the data
                im = ax.pcolormesh(x_coords, y_coords, data, cmap=cm, norm=norm)
                
                # Add contour for flood threshold
                if np.nanmax(data) > flood_threshold:
                    ax.contour(x_coords, y_coords, data, levels=[flood_threshold], 
                              colors='red', linewidths=0.5)
                
                # Set title and labels
                if has_time:
                    ax.set_title(f"{time_str}")
                
                # Only add labels to bottom and leftmost plots
                if i >= n_plots - n_cols:  # Bottom row
                    ax.set_xlabel("Longitude" if "lon" in dataset.coords else "X")
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                
                if i % n_cols == 0:  # Leftmost column
                    ax.set_ylabel("Latitude" if "lat" in dataset.coords else "Y")
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
        
        # Hide unused axes
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        units = dataset[variable].attrs.get("units", "m")
        cbar.set_label(f"{variable.replace('_', ' ').title()} ({units})")
        
        # Add title
        if title is None:
            if "name" in dataset.attrs:
                title = f"Flood forecast: {dataset.attrs['name']}"
            else:
                title = f"Flood forecast: {variable.replace('_', ' ').title()}"
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created forecast map at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating forecast map: {str(e)}")
        return None

def create_forecast_animation(
    dataset: xr.Dataset,
    variable: str = "stage",
    output_path: str = "forecast_animation.gif",
    flood_threshold: float = 0.1,
    cmap: str = "cmocean.cm.deep",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    fps: int = 4
) -> str:
    """
    Create animation of forecast data over time
    
    Args:
        dataset: xarray Dataset containing model outputs
        variable: Variable to animate (default: "stage" for water depth)
        output_path: Path to save the animation
        flood_threshold: Threshold value for flood classification (m)
        cmap: Colormap name or colormap object
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the figure
        fps: Frames per second
        
    Returns:
        Path to the created animation file
    """
    try:
        # Check if time dimension exists
        if "time" not in dataset.coords:
            raise ValueError("Dataset must have a time dimension for animation")
        
        # Extract coordinates and time
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            x_coords = dataset.coords["lon"].values
            y_coords = dataset.coords["y"].values
        else:
            raise ValueError("Could not find spatial coordinates in dataset")
        
        time_values = dataset.coords["time"].values
        
        # Check if variable exists
        if variable not in dataset.variables:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Get colormap
        if isinstance(cmap, str):
            if cmap.startswith("cmocean.cm."):
                cmap_name = cmap.split(".")[-1]
                try:
                    cm = getattr(cmocean.cm, cmap_name)
                except AttributeError:
                    logger.warning(f"Colormap {cmap} not found in cmocean, using viridis")
                    cm = plt.cm.viridis
            else:
                cm = plt.get_cmap(cmap)
        else:
            cm = cmap
        
        # Initialize figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get global min and max for consistent color scale
        all_data = dataset[variable].values
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
        
        # Apply minimum threshold for better visualization
        if vmin >= 0 and vmax > flood_threshold:
            vmin = 0
        
        # Create norm with special emphasis on flood threshold
        if variable == "stage" or variable == "water_depth":
            # Create a colormap with a sharp transition at the flood threshold
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=flood_threshold, vmax=vmax)
        else:
            norm = None
        
        # Initial plot with the first time step
        data = dataset[variable].isel(time=0).values
        mesh = ax.pcolormesh(x_coords, y_coords, data, cmap=cm, norm=norm)
        
        # Add contour for flood threshold if data exceeds it
        if np.nanmax(data) > flood_threshold:
            contour = ax.contour(x_coords, y_coords, data, levels=[flood_threshold], 
                               colors='red', linewidths=0.5)
        else:
            contour = None
        
        # Add labels and title
        ax.set_xlabel("Longitude" if "lon" in dataset.coords else "X")
        ax.set_ylabel("Latitude" if "lat" in dataset.coords else "Y")
        time_str = str(time_values[0])
        title = ax.set_title(f"Time: {time_str}")
        
        # Add colorbar
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
        units = dataset[variable].attrs.get("units", "m")
        cbar.set_label(f"{variable.replace('_', ' ').title()} ({units})")
        
        plt.tight_layout()
        
        # Animation update function
        def update(frame):
            # Update data
            data = dataset[variable].isel(time=frame).values
            mesh.set_array(data.ravel())
            
            # Update contour
            if contour:
                for coll in contour.collections:
                    coll.remove()
            
            if np.nanmax(data) > flood_threshold:
                new_contour = ax.contour(x_coords, y_coords, data, levels=[flood_threshold], 
                                        colors='red', linewidths=0.5)
            
            # Update title with time
            time_str = str(time_values[frame])
            if isinstance(time_values[frame], np.datetime64):
                try:
                    dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            title.set_text(f"Time: {time_str}")
            
            return [mesh, title]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(time_values), 
            interval=1000//fps, blit=False
        )
        
        # Save animation
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        
        plt.close(fig)
        
        logger.info(f"Created forecast animation at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating forecast animation: {str(e)}")
        return None

def plot_time_series(
    dataset: xr.Dataset,
    locations: List[Tuple[float, float]],
    variable: str = "stage",
    output_path: str = "time_series.png",
    location_names: Optional[List[str]] = None,
    flood_threshold: float = 0.1,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150
) -> str:
    """
    Plot time series at specific locations
    
    Args:
        dataset: xarray Dataset containing model outputs
        locations: List of (x, y) or (lon, lat) coordinates
        variable: Variable to plot (default: "stage" for water depth)
        output_path: Path to save the plot
        location_names: Optional list of names for each location
        flood_threshold: Threshold value for flood classification (m)
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the figure
        
    Returns:
        Path to the created plot file
    """
    try:
        # Check if time dimension exists
        if "time" not in dataset.coords:
            raise ValueError("Dataset must have a time dimension for time series")
        
        # Determine coordinate variables
        if "x" in dataset.coords and "y" in dataset.coords:
            x_var, y_var = "x", "y"
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            x_var, y_var = "lon", "lat"
        else:
            raise ValueError("Could not find spatial coordinates in dataset")
        
        # Check if variable exists
        if variable not in dataset.variables:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get time values
        time_values = dataset.coords["time"].values
        
        # Convert to datetime for better plotting
        if isinstance(time_values[0], np.datetime64):
            x_dates = True
            time_values = [datetime.fromisoformat(str(t).replace('Z', '+00:00')) 
                          for t in time_values]
        else:
            x_dates = False
        
        # Default location names if not provided
        if location_names is None:
            location_names = [f"Location {i+1}" for i in range(len(locations))]
        
        # Truncate or extend location names list if necessary
        if len(location_names) < len(locations):
            location_names.extend([f"Location {i+1}" for i in range(len(location_names), len(locations))])
        elif len(location_names) > len(locations):
            location_names = location_names[:len(locations)]
        
        # Plot time series for each location
        max_value = 0
        
        for i, (x, y) in enumerate(locations):
            # Find nearest grid point
            x_idx = np.abs(dataset.coords[x_var].values - x).argmin()
            y_idx = np.abs(dataset.coords[y_var].values - y).argmin()
            
            # Extract time series
            data = dataset[variable].sel({x_var: dataset.coords[x_var][x_idx], 
                                      y_var: dataset.coords[y_var][y_idx]}).values
            
            # Plot the time series
            line, = ax.plot(time_values, data, label=location_names[i], marker='o', markersize=4)
            
            # Update max value
            max_value = max(max_value, np.nanmax(data))
        
        # Add threshold line if data exceeds it
        if max_value >= flood_threshold:
            ax.axhline(y=flood_threshold, color='r', linestyle='--', 
                     label=f"Flood threshold ({flood_threshold} m)")
        
        # Add labels and title
        if x_dates:
            fig.autofmt_xdate()
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Time step")
        
        units = dataset[variable].attrs.get("units", "m")
        ax.set_ylabel(f"{variable.replace('_', ' ').title()} ({units})")
        
        if "name" in dataset.attrs:
            title = f"Time series at selected locations: {dataset.attrs['name']}"
        else:
            title = f"Time series of {variable.replace('_', ' ').title()} at selected locations"
        
        ax.set_title(title)
        
        # Add legend
        ax.legend(loc='best')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created time series plot at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")
        return None

def plot_flood_statistics(
    dataset: xr.Dataset,
    output_path: str = "flood_statistics.png",
    flood_threshold: float = 0.1,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150
) -> str:
    """
    Plot flood statistics over time
    
    Args:
        dataset: xarray Dataset containing model outputs
        output_path: Path to save the plot
        flood_threshold: Threshold value for flood classification (m)
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the figure
        
    Returns:
        Path to the created plot file
    """
    try:
        # Check if time dimension exists
        if "time" not in dataset.coords:
            raise ValueError("Dataset must have a time dimension for time series")
        
        # Get time values
        time_values = dataset.coords["time"].values
        
        # Convert to datetime for better plotting
        if isinstance(time_values[0], np.datetime64):
            x_dates = True
            time_values = [datetime.fromisoformat(str(t).replace('Z', '+00:00')) 
                          for t in time_values]
        else:
            x_dates = False
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Calculate flood statistics over time
        flood_area = []
        max_depth = []
        mean_depth = []
        
        for i in range(len(time_values)):
            # Get stage data for this time step
            if "stage" in dataset:
                data = dataset["stage"].isel(time=i).values
            elif "water_depth" in dataset:
                data = dataset["water_depth"].isel(time=i).values
            else:
                raise ValueError("Could not find stage or water_depth variable in dataset")
            
            # Calculate flood area (percentage of grid cells above threshold)
            flood_mask = data > flood_threshold
            pct_flooded = 100 * flood_mask.sum() / flood_mask.size
            flood_area.append(pct_flooded)
            
            # Calculate max and mean depth in flooded areas
            if flood_mask.sum() > 0:
                max_depth.append(float(np.nanmax(data[flood_mask])))
                mean_depth.append(float(np.nanmean(data[flood_mask])))
            else:
                max_depth.append(0)
                mean_depth.append(0)
        
        # Plot flood area
        ax1.plot(time_values, flood_area, 'b-', linewidth=2, marker='o')
        ax1.set_ylabel("Flooded area (%)")
        ax1.set_title("Percentage of area flooded over time")
        ax1.grid(True, alpha=0.3)
        
        # Plot max and mean depth
        ax2.plot(time_values, max_depth, 'r-', linewidth=2, marker='o', label="Maximum depth")
        ax2.plot(time_values, mean_depth, 'g-', linewidth=2, marker='s', label="Mean depth")
        ax2.set_ylabel("Water depth (m)")
        ax2.set_title("Flood depth statistics over time")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Format x-axis
        if x_dates:
            fig.autofmt_xdate()
            ax2.set_xlabel("Time")
        else:
            ax2.set_xlabel("Time step")
        
        # Add overall title
        if "name" in dataset.attrs:
            fig.suptitle(f"Flood Statistics: {dataset.attrs['name']}", fontsize=16)
        else:
            fig.suptitle("Flood Statistics", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created flood statistics plot at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating flood statistics plot: {str(e)}")
        return None 