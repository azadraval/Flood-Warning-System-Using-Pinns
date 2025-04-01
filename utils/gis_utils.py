"""
GIS Utilities for Flood Early Warning System

This module provides utilities for converting model outputs to GIS formats
for integration with mapping platforms and spatial visualization.
"""

import os
import logging
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import geojson
from geojson import Feature, FeatureCollection, Point, Polygon
from shapely.geometry import MultiPolygon, mapping
import geopandas as gpd
from typing import Dict, List, Union, Optional, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_geojson(
    dataset: xr.Dataset,
    variable: str = "stage",
    time_step: int = -1,
    threshold: float = 0.01,
    simplify_tolerance: float = None
) -> Dict:
    """
    Convert xarray dataset to GeoJSON format for web mapping
    
    Args:
        dataset: xarray Dataset containing model outputs
        variable: Variable to extract (default: "stage" for water depth)
        time_step: Time step to extract (-1 for last time step)
        threshold: Threshold value to consider for flood extent (m)
        simplify_tolerance: Tolerance for geometry simplification (degrees)
        
    Returns:
        GeoJSON compatible dictionary
    """
    try:
        # Extract coordinates
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            x_coords = dataset.coords["lon"].values
            y_coords = dataset.coords["lat"].values
        else:
            raise ValueError("Could not find spatial coordinates in dataset")
        
        # Extract time step
        if "time" in dataset.coords:
            time_values = dataset.coords["time"].values
            if time_step >= len(time_values) or time_step < -len(time_values):
                logger.warning(f"Time step {time_step} out of range, using last time step")
                time_step = -1
            time_value = time_values[time_step]
            time_str = str(time_value)
        else:
            time_str = None
        
        # Extract variable data
        if variable in dataset.variables:
            if time_str is not None and "time" in dataset[variable].dims:
                data = dataset[variable].sel(time=time_value).values
            else:
                data = dataset[variable].values
                
            if time_step != -1 and len(data.shape) > 2:
                data = data[time_step]
        else:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Create grid of coordinates
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        # Option 1: Create points for each cell with value above threshold
        features = []
        
        # Find cells above threshold
        mask = data > threshold
        
        if mask.sum() == 0:
            logger.warning(f"No values above threshold {threshold} found")
            return {"type": "FeatureCollection", "features": []}
        
        # Create a binary flood extent dataset
        flood_extent = np.zeros_like(data)
        flood_extent[mask] = 1
        
        # Convert to polygons using GeoPandas
        try:
            # Create a GeoDataFrame with polygons
            shapes = []
            values = []
            
            # Use rasterio to create shapes from the binary mask
            from rasterio import features
            
            # Create transform from coordinates
            transform = from_origin(
                x_coords.min(), 
                y_coords.max(), 
                (x_coords[1] - x_coords[0]), 
                (y_coords[1] - y_coords[0])
            )
            
            # Extract shapes from raster
            for shape, value in features.shapes(flood_extent.astype('uint8'), transform=transform):
                if value == 1:  # This is a flooded area
                    shapes.append(shape)
                    values.append(value)
            
            if not shapes:
                logger.warning("No flood polygons generated")
                return {"type": "FeatureCollection", "features": []}
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {"value": values}, 
                geometry=gpd.GeoSeries.from_wkt([str(geojson.loads(str(s))) for s in shapes])
            )
            
            # Simplify geometries if requested
            if simplify_tolerance is not None:
                gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance)
            
            # Dissolve to get a single multipolygon for the flood extent
            dissolved = gdf.dissolve(by="value")
            
            # Convert to GeoJSON features
            for idx, row in dissolved.iterrows():
                properties = {
                    "variable": variable,
                    "time": time_str,
                    "flood_area_sqkm": row.geometry.area * 111 * 111,  # Approximate conversion from degrees to km²
                }
                
                # Add statistics
                properties["max_value"] = float(data[mask].max())
                properties["mean_value"] = float(data[mask].mean())
                
                # Create feature
                feature = Feature(
                    geometry=mapping(row.geometry),
                    properties=properties
                )
                features.append(feature)
            
        except Exception as e:
            logger.error(f"Failed to create polygons: {str(e)}")
            
            # Fallback: Create point features
            for i in range(len(y_coords)):
                for j in range(len(x_coords)):
                    if data[i, j] > threshold:
                        point = Point((float(grid_x[i, j]), float(grid_y[i, j])))
                        properties = {
                            "value": float(data[i, j]),
                            "variable": variable
                        }
                        if time_str:
                            properties["time"] = time_str
                        
                        feature = Feature(geometry=point, properties=properties)
                        features.append(feature)
        
        # Create feature collection
        feature_collection = FeatureCollection(features)
        
        return feature_collection
    
    except Exception as e:
        logger.error(f"Error generating GeoJSON: {str(e)}")
        return {
            "type": "FeatureCollection", 
            "features": [],
            "error": str(e)
        }

def create_raster_file(
    dataset: xr.Dataset,
    variable: str = "stage",
    time_step: int = -1,
    output_path: str = "flood_forecast.tif",
    crs: str = "EPSG:4326"
) -> str:
    """
    Create a GeoTIFF raster file from model output
    
    Args:
        dataset: xarray Dataset containing model outputs
        variable: Variable to extract (default: "stage" for water depth)
        time_step: Time step to extract (-1 for last time step)
        output_path: Path to save the GeoTIFF file
        crs: Coordinate reference system
        
    Returns:
        Path to the created raster file
    """
    try:
        # Extract coordinates
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
        elif "lon" in dataset.coords and "lat" in dataset.coords:
            x_coords = dataset.coords["lon"].values
            y_coords = dataset.coords["lat"].values
        else:
            raise ValueError("Could not find spatial coordinates in dataset")
        
        # Extract time step
        if "time" in dataset.coords:
            time_values = dataset.coords["time"].values
            if time_step >= len(time_values) or time_step < -len(time_values):
                logger.warning(f"Time step {time_step} out of range, using last time step")
                time_step = -1
            time_value = time_values[time_step]
        else:
            time_value = None
        
        # Extract variable data
        if variable in dataset.variables:
            if time_value is not None and "time" in dataset[variable].dims:
                data = dataset[variable].sel(time=time_value).values
            else:
                data = dataset[variable].values
                
            if time_step != -1 and len(data.shape) > 2:
                data = data[time_step]
        else:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Fill NaN values with nodata value
        nodata_value = -9999.0
        data = np.where(np.isnan(data), nodata_value, data)
        
        # Create transform from coordinates
        # Note: rasterio uses (west, north) as the origin
        transform = from_origin(
            x_coords.min(), 
            y_coords.max(), 
            (x_coords[1] - x_coords[0]), 
            (y_coords[1] - y_coords[0])
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write raster file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(data, 1)
            
            # Add metadata
            dst.update_tags(
                variable=variable,
                units=dataset[variable].attrs.get("units", "unknown"),
                time=str(time_value) if time_value is not None else "unknown"
            )
        
        logger.info(f"Created raster file: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating raster file: {str(e)}")
        return None

def read_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Read a shapefile and return a GeoDataFrame
    
    Args:
        shapefile_path: Path to the shapefile
        
    Returns:
        GeoDataFrame containing shapefile data
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        logger.error(f"Error reading shapefile: {str(e)}")
        return None

def overlay_with_boundaries(
    forecast_geojson: Dict,
    boundary_gdf: gpd.GeoDataFrame,
    boundary_column: str = "NAME"
) -> Dict:
    """
    Overlay flood forecast with administrative boundaries
    
    Args:
        forecast_geojson: GeoJSON dictionary with flood forecast
        boundary_gdf: GeoDataFrame with administrative boundaries
        boundary_column: Column name for boundary names
        
    Returns:
        GeoJSON with intersected data
    """
    try:
        # Convert forecast GeoJSON to GeoDataFrame
        forecast_gdf = gpd.GeoDataFrame.from_features(forecast_geojson["features"])
        
        # Ensure CRS match
        if forecast_gdf.crs is None:
            forecast_gdf.crs = "EPSG:4326"
        
        if forecast_gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(forecast_gdf.crs)
        
        # Perform spatial join
        intersections = gpd.overlay(forecast_gdf, boundary_gdf, how="intersection")
        
        # Calculate area statistics per boundary
        result_features = []
        
        for name, group in intersections.groupby(boundary_column):
            # Calculate area of flood within this boundary
            flood_area = group.geometry.area.sum()
            
            # Get original boundary area
            boundary_area = boundary_gdf.loc[boundary_gdf[boundary_column] == name, "geometry"].area.iloc[0]
            
            # Calculate percentage of boundary affected
            pct_affected = (flood_area / boundary_area) * 100
            
            # Get union of all flood geometries in this boundary
            union_geom = group.geometry.unary_union
            
            # Create feature
            properties = {
                "boundary_name": name,
                "flood_area": float(flood_area),
                "pct_affected": float(pct_affected),
                "mean_depth": float(group["value"].mean()) if "value" in group.columns else None
            }
            
            feature = Feature(
                geometry=mapping(union_geom),
                properties=properties
            )
            result_features.append(feature)
        
        return FeatureCollection(result_features)
    
    except Exception as e:
        logger.error(f"Error overlaying with boundaries: {str(e)}")
        return {
            "type": "FeatureCollection", 
            "features": [],
            "error": str(e)
        } 