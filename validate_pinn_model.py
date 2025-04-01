"""
PINN Model Validation and Performance Benchmarking (Phase 6)

This script validates the trained PINN model against:
1. ANUGA simulation results
2. Historical flood events
3. Cross-validation with real-world data

Metrics used:
- Mean Squared Error (MSE), Root Mean Square Error (RMSE)
- Structural Similarity Index (SSIM) for spatial accuracy
- F1-score, precision-recall for flood event classification
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local modules
from flood_warning_system.utils.validation_metrics import (
    calculate_rmse, 
    calculate_mse, 
    calculate_ssim, 
    calculate_classification_metrics,
    calculate_precision_recall_curve,
    create_binary_flood_mask
)
from flood_warning_system.integrate_pinn_model import load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PINNModelValidator:
    """
    Class for validating PINN model performance against different data sources
    using multiple metrics.
    """
    
    def __init__(self, model, config=None, device=None):
        """
        Initialize the validator with a trained PINN model.
        
        Args:
            model: Trained PINN model
            config: Configuration dictionary (optional)
            device: Device to run on (auto-detected if None)
        """
        self.model = model
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            "anuga": {},
            "historical": {},
            "real_world": {}
        }
        logger.info(f"Initialized PINN model validator using device: {self.device}")
    
    def validate_against_anuga(self, anuga_data_path, flood_threshold=0.1):
        """
        Validate the PINN model against ANUGA simulation results.
        
        Args:
            anuga_data_path: Path to ANUGA simulation results (NetCDF/xarray format)
            flood_threshold: Threshold for binary flood classification (meters)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Validating against ANUGA simulation data from {anuga_data_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(anuga_data_path):
                logger.error(f"ANUGA data file not found: {anuga_data_path}")
                return {"error": "File not found"}
            
            # Load ANUGA simulation data
            try:
                anuga_ds = xr.open_dataset(anuga_data_path)
            except Exception as e:
                logger.error(f"Failed to open ANUGA dataset: {str(e)}")
                return {"error": f"Failed to open dataset: {str(e)}"}
            
            # Extract relevant variables with error handling
            try:
                # Check if stage/water depth exists in the dataset
                stage_var_name = None
                for var_name in ['stage', 'water_depth', 'depth', 'h']:
                    if var_name in anuga_ds:
                        stage_var_name = var_name
                        break
                
                if not stage_var_name:
                    logger.error("No stage or water depth variable found in ANUGA dataset")
                    return {"error": "No stage or water depth variable found"}
                
                # Extract stage/depth values
                anuga_stage = anuga_ds[stage_var_name].values
                
                # Extract momentum components if available
                anuga_xmomentum = None
                for var_name in ['xmomentum', 'momentum_x', 'hu']:
                    if var_name in anuga_ds:
                        anuga_xmomentum = anuga_ds[var_name].values
                        break
                
                anuga_ymomentum = None
                for var_name in ['ymomentum', 'momentum_y', 'hv']:
                    if var_name in anuga_ds:
                        anuga_ymomentum = anuga_ds[var_name].values
                        break
            except Exception as e:
                logger.error(f"Error extracting variables from ANUGA dataset: {str(e)}")
                return {"error": f"Variable extraction failed: {str(e)}"}
            
            # Prepare inputs for PINN prediction
            pinn_inputs = self._prepare_pinn_inputs_from_anuga(anuga_ds)
            
            # Generate PINN predictions
            pinn_predictions = self._generate_pinn_predictions(pinn_inputs)
            
            # Extract prediction components with error handling
            if "stage" not in pinn_predictions:
                logger.error("No 'stage' field in PINN predictions")
                return {"error": "Missing stage in predictions"}
            
            pinn_stage = pinn_predictions["stage"]
            pinn_xmomentum = pinn_predictions.get("xmomentum")
            pinn_ymomentum = pinn_predictions.get("ymomentum")
            
            # Validate shapes for consistency
            if anuga_stage.shape != pinn_stage.shape:
                logger.warning(f"Shape mismatch between ANUGA ({anuga_stage.shape}) and PINN ({pinn_stage.shape}) stage values")
                
                # Attempt to reshape if possible
                try:
                    # If 3D temporal data, try to match the last two dimensions (spatial)
                    if len(anuga_stage.shape) == 3 and len(pinn_stage.shape) == 3:
                        # Check if temporal dimensions match
                        if anuga_stage.shape[0] != pinn_stage.shape[0]:
                            # Truncate or pad to match
                            min_time = min(anuga_stage.shape[0], pinn_stage.shape[0])
                            anuga_stage = anuga_stage[:min_time]
                            pinn_stage = pinn_stage[:min_time]
                    elif len(anuga_stage.shape) == 2 and len(pinn_stage.shape) == 2:
                        # Attempt to resize 2D spatial data
                        from skimage.transform import resize
                        pinn_stage = resize(pinn_stage, anuga_stage.shape, preserve_range=True)
                except Exception as e:
                    logger.error(f"Failed to reshape prediction data: {str(e)}")
                    return {"error": f"Shape mismatch that couldn't be resolved: {str(e)}"}
            
            # Calculate regression metrics for water depth/stage
            stage_metrics = {
                "rmse": calculate_rmse(anuga_stage, pinn_stage),
                "mse": calculate_mse(anuga_stage, pinn_stage)
            }
            
            # Calculate SSIM for spatial accuracy (for each timestep if 3D data)
            ssim_values = []
            
            try:
                if len(anuga_stage.shape) == 3:
                    # 3D data with time dimension
                    for t in range(len(anuga_stage)):
                        if t < len(pinn_stage):
                            ssim_val = calculate_ssim(anuga_stage[t], pinn_stage[t])
                            ssim_values.append(ssim_val)
                    
                    # Also store per-timestep RMSE for temporal analysis
                    temporal_rmse = []
                    for t in range(len(anuga_stage)):
                        if t < len(pinn_stage):
                            rmse_val = calculate_rmse(anuga_stage[t], pinn_stage[t])
                            temporal_rmse.append(rmse_val)
                    
                    stage_metrics["temporal_rmse"] = temporal_rmse
                else:
                    # 2D data
                    ssim_val = calculate_ssim(anuga_stage, pinn_stage)
                    ssim_values.append(ssim_val)
            except Exception as e:
                logger.error(f"Error calculating SSIM: {str(e)}")
                # Continue without SSIM if calculation fails
            
            # Add average SSIM if calculated
            stage_metrics["ssim"] = np.mean(ssim_values) if ssim_values else np.nan
            
            # Calculate classification metrics (flooded vs. non-flooded)
            try:
                anuga_binary = create_binary_flood_mask(anuga_stage, threshold=flood_threshold)
                pinn_binary = create_binary_flood_mask(pinn_stage, threshold=flood_threshold)
                
                # Flatten arrays for classification metrics
                anuga_binary_flat = anuga_binary.flatten()
                pinn_stage_flat = pinn_stage.flatten()
                
                # Calculate classification metrics
                classification_metrics = calculate_classification_metrics(
                    anuga_binary_flat, pinn_stage_flat, threshold=flood_threshold
                )
            except Exception as e:
                logger.error(f"Error calculating classification metrics: {str(e)}")
                classification_metrics = {
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan
                }
            
            # Combine metrics
            metrics = {
                "stage": stage_metrics,
                "classification": classification_metrics
            }
            
            # Add momentum metrics if available
            if anuga_xmomentum is not None and pinn_xmomentum is not None:
                try:
                    metrics["xmomentum"] = {
                        "rmse": calculate_rmse(anuga_xmomentum, pinn_xmomentum),
                        "mse": calculate_mse(anuga_xmomentum, pinn_xmomentum)
                    }
                except Exception as e:
                    logger.error(f"Error calculating xmomentum metrics: {str(e)}")
            
            if anuga_ymomentum is not None and pinn_ymomentum is not None:
                try:
                    metrics["ymomentum"] = {
                        "rmse": calculate_rmse(anuga_ymomentum, pinn_ymomentum),
                        "mse": calculate_mse(anuga_ymomentum, pinn_ymomentum)
                    }
                except Exception as e:
                    logger.error(f"Error calculating ymomentum metrics: {str(e)}")
            
            # Store results
            self.results["anuga"] = metrics
            
            # Log results with safe access
            stage_rmse = metrics.get("stage", {}).get("rmse", np.nan)
            stage_ssim = metrics.get("stage", {}).get("ssim", np.nan)
            f1_score = metrics.get("classification", {}).get("f1_score", np.nan)
            
            logger.info(f"ANUGA validation metrics: RMSE={stage_rmse:.4f}, "
                       f"SSIM={stage_ssim:.4f}, "
                       f"F1-score={f1_score:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating against ANUGA data: {str(e)}")
            self.results["anuga"] = {"error": str(e)}
            return {"error": str(e)}
    
    def validate_against_historical(self, historical_data_path, flood_threshold=0.1):
        """
        Validate the PINN model against historical flood event data.
        
        Args:
            historical_data_path: Path to historical flood data
            flood_threshold: Threshold for binary flood classification (meters)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Validating against historical flood data from {historical_data_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(historical_data_path):
                logger.error(f"Historical data file not found: {historical_data_path}")
                return {"error": "File not found"}
                
            # Determine file format based on extension
            file_extension = os.path.splitext(historical_data_path)[1].lower()
            
            try:
                # Load historical data based on file format
                if file_extension == '.nc':
                    # NetCDF format
                    historical_ds = xr.open_dataset(historical_data_path)
                    data_format = 'netcdf'
                elif file_extension in ['.csv', '.txt']:
                    # CSV format
                    historical_df = pd.read_csv(historical_data_path)
                    data_format = 'csv'
                elif file_extension == '.json':
                    # JSON format
                    with open(historical_data_path, 'r') as f:
                        historical_data = json.load(f)
                    data_format = 'json'
                else:
                    logger.error(f"Unsupported historical data format: {file_extension}")
                    return {"error": f"Unsupported file format: {file_extension}"}
            except Exception as e:
                logger.error(f"Failed to load historical data: {str(e)}")
                return {"error": f"Failed to load data: {str(e)}"}
            
            # Extract and validate data based on format
            try:
                if data_format == 'netcdf':
                    # Check if water depth/stage exists in the dataset
                    stage_var_name = None
                    for var_name in ['stage', 'water_depth', 'depth', 'h', 'flood_depth']:
                        if var_name in historical_ds:
                            stage_var_name = var_name
                            break
                    
                    if not stage_var_name:
                        logger.error("No water depth variable found in historical dataset")
                        return {"error": "No water depth variable found"}
                    
                    historical_stage = historical_ds[stage_var_name].values
                    
                elif data_format == 'csv':
                    # Check for water depth column
                    depth_col_name = None
                    for col_name in ['stage', 'water_depth', 'depth', 'flood_depth']:
                        if col_name in historical_df.columns:
                            depth_col_name = col_name
                            break
                    
                    if not depth_col_name:
                        logger.error("No water depth column found in historical CSV data")
                        return {"error": "No water depth column found"}
                    
                    historical_stage = historical_df[depth_col_name].values
                    
                    # Reshape if needed (assuming we need to convert 1D to 2D grid)
                    if len(historical_stage.shape) == 1:
                        # Check if x and y coordinates exist
                        if 'x' in historical_df.columns and 'y' in historical_df.columns:
                            # Determine grid dimensions from data
                            unique_x = historical_df['x'].unique()
                            unique_y = historical_df['y'].unique()
                            
                            if len(unique_x) * len(unique_y) == len(historical_stage):
                                # Reshape to 2D grid
                                historical_stage = historical_stage.reshape(len(unique_y), len(unique_x))
                    
                elif data_format == 'json':
                    # Attempt to extract water depth from JSON format
                    if isinstance(historical_data, dict) and 'water_depth' in historical_data:
                        # Dictionary structure
                        historical_stage = np.array(historical_data['water_depth'])
                    elif isinstance(historical_data, list) and len(historical_data) > 0:
                        # List of records
                        # Look for depth field in the first record
                        depth_field = None
                        for field in ['stage', 'water_depth', 'depth', 'flood_depth']:
                            if field in historical_data[0]:
                                depth_field = field
                                break
                        
                        if depth_field:
                            historical_stage = np.array([record[depth_field] for record in historical_data])
                        else:
                            logger.error("No water depth field found in historical JSON data")
                            return {"error": "No water depth field found"}
                    else:
                        logger.error("Unexpected JSON structure in historical data")
                        return {"error": "Unexpected JSON structure"}
            except Exception as e:
                logger.error(f"Error extracting water depth from historical data: {str(e)}")
                return {"error": f"Failed to extract water depth: {str(e)}"}
            
            # Prepare inputs for PINN prediction
            if data_format == 'netcdf':
                pinn_inputs = self._prepare_pinn_inputs_from_historical(historical_ds)
            elif data_format == 'csv':
                pinn_inputs = self._prepare_pinn_inputs_from_historical(historical_df, format='csv')
            else:  # JSON
                pinn_inputs = self._prepare_pinn_inputs_from_historical(historical_data, format='json')
            
            # Generate PINN predictions
            pinn_predictions = self._generate_pinn_predictions(pinn_inputs)
            
            # Extract predictions
            if "stage" not in pinn_predictions:
                logger.error("No 'stage' field in PINN predictions")
                return {"error": "Missing stage in predictions"}
                
            pinn_stage = pinn_predictions["stage"]
            
            # Validate shapes for consistency
            if historical_stage.shape != pinn_stage.shape:
                logger.warning(f"Shape mismatch between historical ({historical_stage.shape}) and PINN ({pinn_stage.shape}) stage values")
                
                # Attempt to reshape if possible
                try:
                    # If 3D temporal data, try to match the last two dimensions (spatial)
                    if len(historical_stage.shape) == 3 and len(pinn_stage.shape) == 3:
                        # Check if temporal dimensions match
                        if historical_stage.shape[0] != pinn_stage.shape[0]:
                            # Truncate to match
                            min_time = min(historical_stage.shape[0], pinn_stage.shape[0])
                            historical_stage = historical_stage[:min_time]
                            pinn_stage = pinn_stage[:min_time]
                    elif len(historical_stage.shape) == 2 and len(pinn_stage.shape) == 2:
                        # Attempt to resize 2D spatial data
                        from skimage.transform import resize
                        pinn_stage = resize(pinn_stage, historical_stage.shape, preserve_range=True)
                    elif len(historical_stage.shape) == 1 and len(pinn_stage.shape) > 1:
                        # Flatten PINN prediction if historical is 1D
                        pinn_stage = pinn_stage.flatten()[:len(historical_stage)]
                    elif len(pinn_stage.shape) == 1 and len(historical_stage.shape) > 1:
                        # Flatten historical if PINN is 1D
                        historical_stage = historical_stage.flatten()[:len(pinn_stage)]
                except Exception as e:
                    logger.error(f"Failed to reshape prediction data: {str(e)}")
                    return {"error": f"Shape mismatch that couldn't be resolved: {str(e)}"}
            
            # Calculate regression metrics
            try:
                stage_metrics = {
                    "rmse": calculate_rmse(historical_stage, pinn_stage),
                    "mse": calculate_mse(historical_stage, pinn_stage)
                }
            
                # Calculate SSIM if data is 2D or 3D
                ssim_values = []
                if len(historical_stage.shape) > 1:
                    try:
                        if len(historical_stage.shape) == 3:
                            # 3D data with time dimension
                            for t in range(min(len(historical_stage), len(pinn_stage))):
                                ssim_val = calculate_ssim(historical_stage[t], pinn_stage[t])
                                ssim_values.append(ssim_val)
                        else:
                            # 2D data
                            ssim_val = calculate_ssim(historical_stage, pinn_stage)
                            ssim_values.append(ssim_val)
                    except Exception as e:
                        logger.error(f"Error calculating SSIM: {str(e)}")
                
                # Add average SSIM if calculated
                stage_metrics["ssim"] = np.mean(ssim_values) if ssim_values else np.nan
            except Exception as e:
                logger.error(f"Error calculating regression metrics: {str(e)}")
                stage_metrics = {"rmse": np.nan, "mse": np.nan, "ssim": np.nan}
            
            # Calculate classification metrics
            try:
                # Create binary masks
                historical_binary = create_binary_flood_mask(historical_stage, threshold=flood_threshold)
                pinn_binary = create_binary_flood_mask(pinn_stage, threshold=flood_threshold)
                
                # Flatten arrays for classification metrics
                historical_binary_flat = historical_binary.flatten()
                pinn_stage_flat = pinn_stage.flatten()
                
                # Calculate precision-recall
                precision, recall, thresholds = calculate_precision_recall_curve(
                    historical_binary_flat, pinn_stage_flat
                )
                
                # Calculate classification metrics at the specified threshold
                classification_metrics = calculate_classification_metrics(
                    historical_binary_flat, pinn_stage_flat, threshold=flood_threshold
                )
                
                # Add precision-recall data
                classification_metrics["precision_curve"] = precision.tolist() if isinstance(precision, np.ndarray) else precision
                classification_metrics["recall_curve"] = recall.tolist() if isinstance(recall, np.ndarray) else recall
                classification_metrics["thresholds"] = thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds
            except Exception as e:
                logger.error(f"Error calculating classification metrics: {str(e)}")
                classification_metrics = {
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan
                }
            
            # Combine metrics
            metrics = {
                "stage": stage_metrics,
                "classification": classification_metrics
            }
            
            # Store results
            self.results["historical"] = metrics
            
            # Log results with safe access
            stage_rmse = metrics.get("stage", {}).get("rmse", np.nan)
            stage_ssim = metrics.get("stage", {}).get("ssim", np.nan)
            f1_score = metrics.get("classification", {}).get("f1_score", np.nan)
            
            logger.info(f"Historical validation metrics: RMSE={stage_rmse:.4f}, "
                       f"SSIM={stage_ssim:.4f}, "
                       f"F1-score={f1_score:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating against historical data: {str(e)}")
            self.results["historical"] = {"error": str(e)}
            return {"error": str(e)}
    
    def cross_validate_real_world(self, real_world_data_path, k_folds=5, flood_threshold=0.1):
        """
        Cross-validate the PINN model with real-world data.
        
        Args:
            real_world_data_path: Path to real-world data
            k_folds: Number of cross-validation folds
            flood_threshold: Threshold for binary flood classification (meters)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Cross-validating with real-world data from {real_world_data_path} using {k_folds} folds")
        
        try:
            # Load real-world data (adapt the loading based on your data format)
            if real_world_data_path.endswith('.nc'):
                real_world_ds = xr.open_dataset(real_world_data_path)
                real_world_data = real_world_ds.to_dict()
            elif real_world_data_path.endswith('.csv'):
                real_world_df = pd.read_csv(real_world_data_path)
                real_world_data = real_world_df.to_dict('list')
            else:
                with open(real_world_data_path, 'r') as f:
                    real_world_data = json.load(f)
            
            # Extract features and targets
            # NOTE: Adapt this based on your specific data structure
            features = real_world_data.get("features", [])
            targets = real_world_data.get("targets", [])
            
            # Convert to numpy arrays if needed
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if not isinstance(targets, np.ndarray):
                targets = np.array(targets)
            
            # Perform k-fold cross-validation
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            fold_metrics = []
            for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
                logger.info(f"Processing fold {fold+1}/{k_folds}")
                
                # Split data
                test_features = features[test_idx]
                test_targets = targets[test_idx]
                
                # Prepare inputs for PINN prediction
                pinn_inputs = self._prepare_pinn_inputs_from_real_world(test_features)
                
                # Generate PINN predictions
                pinn_predictions = self._generate_pinn_predictions(pinn_inputs)
                
                # Extract predicted values
                predicted_values = pinn_predictions.get("stage", [])
                
                # Calculate regression metrics
                fold_metric = {
                    "rmse": calculate_rmse(test_targets, predicted_values),
                    "mse": calculate_mse(test_targets, predicted_values)
                }
                
                # If spatial data is available, calculate SSIM
                if isinstance(test_targets, np.ndarray) and len(test_targets.shape) > 1:
                    fold_metric["ssim"] = calculate_ssim(test_targets, predicted_values)
                
                # If binary flood data is available, calculate classification metrics
                if "flood_extent" in real_world_data:
                    observed_binary = real_world_data["flood_extent"][test_idx]
                    predicted_binary = create_binary_flood_mask(predicted_values, threshold=flood_threshold)
                    
                    classification_metrics = calculate_classification_metrics(
                        observed_binary, predicted_values, threshold=flood_threshold
                    )
                    fold_metric["classification"] = classification_metrics
                
                fold_metrics.append(fold_metric)
            
            # Aggregate metrics across folds
            aggregated_metrics = {}
            for key in fold_metrics[0].keys():
                if isinstance(fold_metrics[0][key], dict):
                    # Handle nested dictionaries (e.g., classification metrics)
                    aggregated_metrics[key] = {}
                    for subkey in fold_metrics[0][key].keys():
                        values = [fold[key][subkey] for fold in fold_metrics]
                        aggregated_metrics[key][subkey] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "values": values
                        }
                else:
                    # Handle simple metrics
                    values = [fold[key] for fold in fold_metrics]
                    aggregated_metrics[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values
                    }
            
            # Store results
            self.results["real_world"] = aggregated_metrics
            
            logger.info(f"Real-world cross-validation metrics: "
                       f"RMSE={aggregated_metrics['rmse']['mean']:.4f} ± {aggregated_metrics['rmse']['std']:.4f}")
            
            if "classification" in aggregated_metrics:
                logger.info(f"Classification metrics: "
                           f"F1-score={aggregated_metrics['classification']['f1_score']['mean']:.4f} ± "
                           f"{aggregated_metrics['classification']['f1_score']['std']:.4f}")
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Error cross-validating with real-world data: {str(e)}")
            raise
    
    def _prepare_pinn_inputs_from_anuga(self, anuga_ds):
        """
        Prepare inputs for PINN prediction from ANUGA simulation data.
        
        Args:
            anuga_ds: xarray Dataset containing ANUGA simulation results
            
        Returns:
            Dictionary of input variables for PINN prediction
        """
        try:
            logger.info("Preparing PINN inputs from ANUGA data")
            
            # Extract spatial coordinates
            x_coords = None
            y_coords = None
            
            # Look for coordinate variables
            for x_var in ['x', 'X', 'longitude', 'lon', 'x_origin']:
                if x_var in anuga_ds.coords or x_var in anuga_ds.variables:
                    x_coords = anuga_ds[x_var].values
                    break
            
            for y_var in ['y', 'Y', 'latitude', 'lat', 'y_origin']:
                if y_var in anuga_ds.coords or y_var in anuga_ds.variables:
                    y_coords = anuga_ds[y_var].values
                    break
            
            # If coordinates not found as variables, try inferring from dimensions
            if x_coords is None and 'x' in anuga_ds.dims:
                x_coords = np.arange(anuga_ds.dims['x'])
            if y_coords is None and 'y' in anuga_ds.dims:
                y_coords = np.arange(anuga_ds.dims['y'])
                
            # If still not found, create default grid coordinates
            if x_coords is None or y_coords is None:
                # Try to determine dimensions from a data variable
                for var_name in ['stage', 'water_depth', 'depth', 'h', 'elevation']:
                    if var_name in anuga_ds and len(anuga_ds[var_name].shape) >= 2:
                        shape = anuga_ds[var_name].shape
                        if len(shape) == 3:  # Has time dimension
                            y_size, x_size = shape[1], shape[2]
                        else:
                            y_size, x_size = shape[0], shape[1]
                        
                        x_coords = np.linspace(0, 1, x_size)
                        y_coords = np.linspace(0, 1, y_size)
                        break
            
            if x_coords is None or y_coords is None:
                logger.error("Failed to determine spatial coordinates from ANUGA data")
                return {}
            
            # Extract time coordinates if available
            t_coords = None
            for t_var in ['time', 't', 'timestep']:
                if t_var in anuga_ds.coords or t_var in anuga_ds.variables:
                    t_values = anuga_ds[t_var].values
                    # Convert to numeric time representation if needed
                    if hasattr(t_values, 'dtype') and np.issubdtype(t_values.dtype, np.datetime64):
                        t_coords = np.array([(t - t_values[0]) / np.timedelta64(1, 's') 
                                           for t in t_values])
                    else:
                        t_coords = t_values
                    break
            
            # If time not found, create default time array
            if t_coords is None:
                for var_name in ['stage', 'water_depth', 'depth', 'h']:
                    if var_name in anuga_ds and len(anuga_ds[var_name].shape) == 3:
                        t_coords = np.arange(anuga_ds[var_name].shape[0])
                        break
            
            # Extract elevation/terrain data
            elevation = None
            for elev_var in ['elevation', 'topography', 'terrain', 'z']:
                if elev_var in anuga_ds:
                    elevation = anuga_ds[elev_var].values
                    break
            
            # If elevation not found, try to infer from other variables
            if elevation is None and 'stage' in anuga_ds and 'depth' in anuga_ds:
                try:
                    elevation = anuga_ds['stage'].values - anuga_ds['depth'].values
                except Exception as e:
                    logger.warning(f"Failed to infer elevation from stage and depth: {str(e)}")
            
            # Create spatial grid for PINN inputs
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Prepare inputs dictionary
            inputs = {
                'x': X,
                'y': Y
            }
            
            # Add time dimension if available
            if t_coords is not None:
                inputs['t'] = t_coords
            
            # Add elevation if available
            if elevation is not None:
                inputs['elevation'] = elevation
            
            # Add any other relevant variables
            for var_name in ['rainfall', 'friction', 'manning']:
                if var_name in anuga_ds:
                    inputs[var_name] = anuga_ds[var_name].values
            
            # Extract boundary conditions if available
            for bc_var in ['boundary_conditions', 'boundaries']:
                if bc_var in anuga_ds:
                    inputs[bc_var] = anuga_ds[bc_var].values
            
            # Convert inputs to appropriate format for the PINN model
            formatted_inputs = self._convert_inputs_to_tensors(inputs)
            
            return formatted_inputs
            
        except Exception as e:
            logger.error(f"Error preparing PINN inputs from ANUGA data: {str(e)}")
            return {}
    
    def _prepare_pinn_inputs_from_historical(self, historical_data, format='netcdf'):
        """
        Prepare inputs for PINN prediction from historical flood event data.
        
        Args:
            historical_data: Historical data (xarray Dataset, pandas DataFrame, or dict)
            format: Data format ('netcdf', 'csv', or 'json')
            
        Returns:
            Dictionary of input variables for PINN prediction
        """
        try:
            logger.info(f"Preparing PINN inputs from historical data (format: {format})")
            
            inputs = {}
            
            if format == 'netcdf':
                # Handle xarray Dataset
                historical_ds = historical_data
                
                # Extract coordinates
                x_coords = None
                y_coords = None
                
                for x_var in ['x', 'X', 'longitude', 'lon']:
                    if x_var in historical_ds.coords or x_var in historical_ds.variables:
                        x_coords = historical_ds[x_var].values
                        break
                
                for y_var in ['y', 'Y', 'latitude', 'lat']:
                    if y_var in historical_ds.coords or y_var in historical_ds.variables:
                        y_coords = historical_ds[y_var].values
                        break
                
                # If not found, infer from dimensions
                if x_coords is None and 'x' in historical_ds.dims:
                    x_coords = np.arange(historical_ds.dims['x'])
                if y_coords is None and 'y' in historical_ds.dims:
                    y_coords = np.arange(historical_ds.dims['y'])
                
                # Create spatial grid
                X, Y = np.meshgrid(x_coords, y_coords)
                inputs['x'] = X
                inputs['y'] = Y
                
                # Extract time coordinates if available
                for t_var in ['time', 't', 'date']:
                    if t_var in historical_ds.coords:
                        t_values = historical_ds[t_var].values
                        # Convert to numeric time representation if needed
                        if hasattr(t_values, 'dtype') and np.issubdtype(t_values.dtype, np.datetime64):
                            inputs['t'] = np.array([(t - t_values[0]) / np.timedelta64(1, 's') 
                                                 for t in t_values])
                        else:
                            inputs['t'] = t_values
                        break
                
                # Extract elevation/terrain data
                for elev_var in ['elevation', 'topography', 'terrain', 'z']:
                    if elev_var in historical_ds:
                        inputs['elevation'] = historical_ds[elev_var].values
                        break
                
                # Add any other relevant variables
                for var_name in ['rainfall', 'precipitation', 'friction']:
                    if var_name in historical_ds:
                        inputs[var_name] = historical_ds[var_name].values
                
            elif format == 'csv':
                # Handle pandas DataFrame
                df = historical_data
                
                # Check if dataframe contains coordinate columns
                has_coords = all(col in df.columns for col in ['x', 'y'])
                
                if has_coords:
                    # Extract coordinates
                    x = df['x'].values
                    y = df['y'].values
                    
                    # Check if data is already on a regular grid
                    unique_x = np.unique(x)
                    unique_y = np.unique(y)
                    
                    # If regular grid (number of points equals product of unique x and y)
                    if len(x) == len(unique_x) * len(unique_y):
                        X, Y = np.meshgrid(unique_x, unique_y)
                        inputs['x'] = X
                        inputs['y'] = Y
                        
                        # Reshape any other columns to 2D grid
                        for col in df.columns:
                            if col not in ['x', 'y']:
                                # Try to reshape to 2D grid
                                try:
                                    inputs[col] = df[col].values.reshape(len(unique_y), len(unique_x))
                                except Exception:
                                    logger.warning(f"Could not reshape column '{col}' to 2D grid")
                    else:
                        # Handle irregular or scattered data points
                        inputs['x'] = x
                        inputs['y'] = y
                        
                        # Add other columns directly
                        for col in df.columns:
                            if col not in ['x', 'y']:
                                inputs[col] = df[col].values
                else:
                    # No coordinates in dataframe
                    logger.warning("CSV data does not contain x, y coordinates")
                    
                    # Try to use columns as variables
                    for col in df.columns:
                        inputs[col] = df[col].values
                
            elif format == 'json':
                # Handle JSON data
                if isinstance(historical_data, dict):
                    # Dictionary structure - extract fields
                    for key, value in historical_data.items():
                        if isinstance(value, list):
                            inputs[key] = np.array(value)
                        else:
                            inputs[key] = value
                            
                    # If no coordinates, try to create default grid
                    if 'x' not in inputs and 'y' not in inputs:
                        # Try to infer grid dimensions from other fields
                        for key, value in inputs.items():
                            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                                y_size, x_size = value.shape
                                X, Y = np.meshgrid(np.linspace(0, 1, x_size), 
                                                np.linspace(0, 1, y_size))
                                inputs['x'] = X
                                inputs['y'] = Y
                                break
                
                elif isinstance(historical_data, list):
                    # List of records - extract x, y coordinates if available
                    if all('x' in record and 'y' in record for record in historical_data):
                        x = np.array([record['x'] for record in historical_data])
                        y = np.array([record['y'] for record in historical_data])
                        inputs['x'] = x
                        inputs['y'] = y
                        
                        # Extract other fields
                        for key in historical_data[0].keys():
                            if key not in ['x', 'y']:
                                inputs[key] = np.array([record.get(key, 0) for record in historical_data])
            
            # Convert inputs to appropriate format for PINN model
            formatted_inputs = self._convert_inputs_to_tensors(inputs)
            
            return formatted_inputs
            
        except Exception as e:
            logger.error(f"Error preparing PINN inputs from historical data: {str(e)}")
            return {}
    
    def _prepare_pinn_inputs_from_real_world(self, real_world_data):
        """
        Prepare inputs for PINN prediction from real-world observation data.
        
        Args:
            real_world_data: Real-world observation data (pandas DataFrame)
            
        Returns:
            Dictionary of input variables for PINN prediction
        """
        try:
            logger.info("Preparing PINN inputs from real-world data")
            
            df = real_world_data
            inputs = {}
            
            # Check if dataframe contains coordinate columns
            required_cols = ['x', 'y']
            
            if set(required_cols).issubset(df.columns):
                # Extract coordinates
                inputs['x'] = df['x'].values
                inputs['y'] = df['y'].values
                
                # Extract time coordinates if available
                if 'time' in df.columns or 't' in df.columns:
                    t_col = 'time' if 'time' in df.columns else 't'
                    t_values = df[t_col].values
                    
                    # Check if time is in datetime format and convert to numeric
                    if hasattr(t_values, 'dtype') and t_values.dtype.kind == 'M':
                        t0 = t_values[0]
                        inputs['t'] = np.array([(t - t0).total_seconds() for t in t_values])
                    else:
                        inputs['t'] = t_values
                
                # Add other variables from available columns
                optional_vars = [
                    'elevation', 'stage', 'water_depth', 'rainfall', 
                    'precipitation', 'velocity', 'discharge'
                ]
                
                for var in optional_vars:
                    if var in df.columns:
                        inputs[var] = df[var].values
                
                # Add any other numeric columns except coordinates and time
                for col in df.columns:
                    if (col not in inputs and col not in required_cols and 
                        col not in ['time', 't'] and 
                        np.issubdtype(df[col].dtype, np.number)):
                        inputs[col] = df[col].values
            else:
                logger.error(f"Real-world data missing required columns: {required_cols}")
                return {}
            
            # Convert inputs to appropriate format for PINN model
            formatted_inputs = self._convert_inputs_to_tensors(inputs)
            
            return formatted_inputs
            
        except Exception as e:
            logger.error(f"Error preparing PINN inputs from real-world data: {str(e)}")
            return {}
    
    def _convert_inputs_to_tensors(self, inputs_dict):
        """
        Convert input data to tensors compatible with the PINN model.
        
        Args:
            inputs_dict: Dictionary of input arrays
            
        Returns:
            Dictionary of tensors formatted for PINN prediction
        """
        try:
            logger.info("Converting inputs to tensors")
            
            tensor_inputs = {}
            
            # Check if CUDA is available
            device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
            
            for key, value in inputs_dict.items():
                if value is None:
                    continue
                    
                # Convert numpy arrays to tensors
                if isinstance(value, np.ndarray):
                    # Handle NaN values
                    if np.isnan(value).any():
                        # Replace NaNs with 0 or appropriate value
                        value = np.nan_to_num(value, nan=0.0)
                    
                    # Convert to float32 for better compatibility with neural networks
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    
                    # Convert to tensor and move to appropriate device
                    tensor = torch.from_numpy(value).to(device)
                    tensor_inputs[key] = tensor
                
                # Handle scalar values
                elif np.isscalar(value):
                    tensor_inputs[key] = torch.tensor([value], dtype=torch.float32, device=device)
                
                # Handle lists by converting to numpy first
                elif isinstance(value, list):
                    try:
                        # Convert list to numpy array
                        np_array = np.array(value, dtype=np.float32)
                        
                        # Handle NaN values
                        if np.isnan(np_array).any():
                            np_array = np.nan_to_num(np_array, nan=0.0)
                        
                        # Convert to tensor
                        tensor = torch.from_numpy(np_array).to(device)
                        tensor_inputs[key] = tensor
                    except Exception as e:
                        logger.warning(f"Could not convert list '{key}' to tensor: {str(e)}")
            
            # Check if spatial coordinates need reshaping for batched input
            # Many PINN models expect input in format [batch_size, features]
            if ('x' in tensor_inputs and 'y' in tensor_inputs and 
                len(tensor_inputs['x'].shape) == 2):
                
                # Flatten spatial coordinates for grid input
                x_flat = tensor_inputs['x'].flatten().unsqueeze(1)  # [N, 1]
                y_flat = tensor_inputs['y'].flatten().unsqueeze(1)  # [N, 1]
                
                # Combine coordinates into feature matrix
                spatial_coords = torch.cat([x_flat, y_flat], dim=1)  # [N, 2]
                
                # Add time dimension if available
                if 't' in tensor_inputs:
                    # Handle different time dimension formats
                    t = tensor_inputs['t']
                    
                    if len(t.shape) == 1:
                        # For multiple timesteps, create a grid of time and space
                        num_times = t.shape[0]
                        num_points = x_flat.shape[0]
                        
                        # Repeat spatial coordinates for each timestep
                        repeated_coords = spatial_coords.repeat(num_times, 1)  # [N*timesteps, 2]
                        
                        # Create time vector repeated for each spatial point
                        time_vec = t.unsqueeze(1).repeat_interleave(num_points, dim=0)  # [N*timesteps, 1]
                        
                        # Combine time and space
                        space_time_coords = torch.cat([repeated_coords, time_vec], dim=1)  # [N*timesteps, 3]
                        tensor_inputs['space_time_grid'] = space_time_coords
                    else:
                        # Single timestep or already formatted time
                        t_flat = t.flatten().unsqueeze(1)  # [N, 1]
                        space_time_coords = torch.cat([x_flat, y_flat, t_flat], dim=1)  # [N, 3]
                        tensor_inputs['space_time_grid'] = space_time_coords
                else:
                    # Just spatial coordinates
                    tensor_inputs['spatial_grid'] = spatial_coords
                
                # Process elevation if available
                if 'elevation' in tensor_inputs:
                    elev = tensor_inputs['elevation']
                    if len(elev.shape) == 2:
                        # Flatten to match coordinates
                        elev_flat = elev.flatten().unsqueeze(1)  # [N, 1]
                        tensor_inputs['elevation_flat'] = elev_flat
            
            return tensor_inputs
            
        except Exception as e:
            logger.error(f"Error converting inputs to tensors: {str(e)}")
            return {}
    
    def _generate_pinn_predictions(self, pinn_inputs):
        """
        Generate predictions using the PINN model.
        
        Args:
            pinn_inputs: Dictionary of tensors formatted for PINN prediction
            
        Returns:
            Dictionary of prediction results
        """
        try:
            logger.info("Generating PINN predictions")
            
            if not self.model:
                logger.error("No PINN model loaded for prediction")
                return {}
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Extract core input tensors based on what's available
            input_tensor = None
            
            # Check for pre-processed grid inputs
            if 'space_time_grid' in pinn_inputs:
                input_tensor = pinn_inputs['space_time_grid']
            elif 'spatial_grid' in pinn_inputs:
                input_tensor = pinn_inputs['spatial_grid']
            else:
                # Try to build input tensor from components
                components = []
                
                # Add spatial coordinates
                if 'x' in pinn_inputs and 'y' in pinn_inputs:
                    x = pinn_inputs['x']
                    y = pinn_inputs['y']
                    
                    # Check if reshape needed
                    if len(x.shape) == 2:
                        x = x.flatten().unsqueeze(1)
                        y = y.flatten().unsqueeze(1)
                    
                    components.append(x)
                    components.append(y)
                
                # Add time component if available
                if 't' in pinn_inputs:
                    t = pinn_inputs['t']
                    
                    # Check if reshape needed
                    if len(t.shape) == 1:
                        t = t.unsqueeze(1)
                    
                    components.append(t)
                
                # If no components found, try a fallback approach
                if not components:
                    logger.warning("No standard input components found, trying fallback method")
                    
                    # Look for any tensor that could serve as input
                    for key, tensor in pinn_inputs.items():
                        if isinstance(tensor, torch.Tensor) and len(tensor.shape) >= 1:
                            input_tensor = tensor
                            logger.info(f"Using '{key}' as fallback input tensor")
                            break
                else:
                    # Concatenate components into input tensor
                    input_tensor = torch.cat(components, dim=1)
            
            if input_tensor is None:
                logger.error("Failed to create input tensor for PINN prediction")
                return {}
            
            # Check for additional conditional inputs
            conditional_inputs = {}
            for key, tensor in pinn_inputs.items():
                if key not in ['x', 'y', 't', 'spatial_grid', 'space_time_grid'] and isinstance(tensor, torch.Tensor):
                    conditional_inputs[key] = tensor
            
            # Generate predictions with gradient computation disabled
            with torch.no_grad():
                try:
                    # Try direct prediction if model has a predict method
                    if hasattr(self.model, 'predict'):
                        if conditional_inputs:
                            raw_predictions = self.model.predict(input_tensor, **conditional_inputs)
                        else:
                            raw_predictions = self.model.predict(input_tensor)
                    # Try direct forward method
                    else:
                        if conditional_inputs:
                            raw_predictions = self.model(input_tensor, **conditional_inputs)
                        else:
                            raw_predictions = self.model(input_tensor)
                except Exception as e:
                    logger.error(f"Error during model prediction: {str(e)}")
                    
                    # Try alternative prediction approaches
                    try:
                        # Try with input_tensor as first positional argument
                        raw_predictions = self.model(input_tensor)
                    except Exception:
                        try:
                            # Try with input_tensor as named argument
                            raw_predictions = self.model(x=input_tensor)
                        except Exception:
                            logger.error("All prediction attempts failed")
                            return {}
            
            # Process raw predictions into structured output
            processed_predictions = self._process_raw_predictions(raw_predictions, pinn_inputs)
            
            return processed_predictions
            
        except Exception as e:
            logger.error(f"Error generating PINN predictions: {str(e)}")
            return {}
    
    def _process_raw_predictions(self, raw_predictions, pinn_inputs):
        """
        Process raw PINN model predictions into structured output.
        
        Args:
            raw_predictions: Raw output from PINN model
            pinn_inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of processed prediction results
        """
        try:
            logger.info("Processing raw PINN predictions")
            
            processed = {}
            
            # Handle different prediction formats
            if isinstance(raw_predictions, dict):
                # Dictionary output - use keys as variable names
                for key, value in raw_predictions.items():
                    # Convert tensor to numpy if needed
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    
                    processed[key] = value
                    
            elif isinstance(raw_predictions, torch.Tensor):
                # Single tensor output - determine variables based on output shape
                output_tensor = raw_predictions.detach().cpu()
                
                # Determine output format based on dimensions
                if len(output_tensor.shape) == 1:
                    # 1D output - single variable
                    processed['output'] = output_tensor.numpy()
                    
                elif len(output_tensor.shape) == 2:
                    # 2D output - multiple variables per point
                    output_np = output_tensor.numpy()
                    
                    # If output has 1-3 columns, assume it's stage, xmomentum, ymomentum
                    if output_np.shape[1] == 1:
                        processed['stage'] = output_np[:, 0]
                    elif output_np.shape[1] == 2:
                        processed['stage'] = output_np[:, 0]
                        processed['xmomentum'] = output_np[:, 1]
                    elif output_np.shape[1] == 3:
                        processed['stage'] = output_np[:, 0]
                        processed['xmomentum'] = output_np[:, 1]
                        processed['ymomentum'] = output_np[:, 2]
                    else:
                        # Multiple output variables - use generic naming
                        for i in range(output_np.shape[1]):
                            processed[f'var_{i}'] = output_np[:, i]
                
                else:
                    # Higher dimensional output - needs reshaping
                    output_np = output_tensor.numpy()
                    
                    # Try to reshape back to original grid if spatial inputs available
                    if 'x' in pinn_inputs and 'y' in pinn_inputs:
                        x = pinn_inputs['x']
                        y = pinn_inputs['y']
                        
                        if len(x.shape) == 2:
                            # Get original grid shape
                            grid_shape = x.shape
                            
                            # Check if output can be reshaped to grid
                            if len(output_np.shape) == 2:
                                # Multiple variables
                                n_vars = output_np.shape[1]
                                
                                # Check if first dimension matches grid size
                                if output_np.shape[0] == grid_shape[0] * grid_shape[1]:
                                    for i in range(n_vars):
                                        # Reshape each variable back to 2D grid
                                        var_data = output_np[:, i].reshape(grid_shape)
                                        
                                        # Name variables based on index
                                        if i == 0:
                                            processed['stage'] = var_data
                                        elif i == 1:
                                            processed['xmomentum'] = var_data
                                        elif i == 2:
                                            processed['ymomentum'] = var_data
                                        else:
                                            processed[f'var_{i}'] = var_data
                                else:
                                    # Can't reshape, store as-is
                                    for i in range(n_vars):
                                        processed[f'var_{i}'] = output_np[:, i]
                            
                            elif len(output_np.shape) == 1:
                                # Single variable
                                if output_np.shape[0] == grid_shape[0] * grid_shape[1]:
                                    # Reshape to 2D grid
                                    processed['stage'] = output_np.reshape(grid_shape)
                                else:
                                    # Can't reshape, store as-is
                                    processed['output'] = output_np
                        else:
                            # Inputs not in grid format, store as-is
                            if len(output_np.shape) == 2:
                                for i in range(output_np.shape[1]):
                                    var_name = ['stage', 'xmomentum', 'ymomentum'][i] if i < 3 else f'var_{i}'
                                    processed[var_name] = output_np[:, i]
                            else:
                                processed['output'] = output_np
                    else:
                        # No spatial grid info available, store as-is
                        processed['output'] = output_np
            
            else:
                # Unknown prediction format
                logger.warning(f"Unknown prediction format: {type(raw_predictions)}")
                processed['output'] = raw_predictions
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing PINN predictions: {str(e)}")
            return {}
    
    def save_results(self, output_dir):
        """
        Save validation results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall results as JSON
        with open(os.path.join(output_dir, f"validation_results_{timestamp}.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=self._json_serialize)
        
        logger.info(f"Saved validation results to {output_dir}")
        
        # Generate and save visualizations
        self._generate_visualizations()
    
    def _json_serialize(self, obj):
        """Helper method to serialize objects for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _generate_visualizations(self):
        """
        Generate and save visualization plots for validation results.
        """
        try:
            logger.info("Generating validation result visualizations")
            
            # Create visualization directory if it doesn't exist
            vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Get timestamp for filenames
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Check which data sources are available
            has_anuga = "anuga" in self.results and not self.results.get("anuga", {}).get("error", None)
            has_historical = "historical" in self.results and not self.results.get("historical", {}).get("error", None)
            has_real_world = "real_world" in self.results and not self.results.get("real_world", {}).get("error", None)
            
            # 1. RMSE comparison across data sources
            try:
                if any([has_anuga, has_historical, has_real_world]):
                    plt.figure(figsize=(10, 6))
                    
                    data_sources = []
                    rmse_values = []
                    mse_values = []
                    
                    # Safely access metrics with dictionary get method
                    if has_anuga:
                        anuga_rmse = self.results.get("anuga", {}).get("stage", {}).get("rmse")
                        anuga_mse = self.results.get("anuga", {}).get("stage", {}).get("mse")
                        if anuga_rmse is not None:
                            data_sources.append("ANUGA")
                            rmse_values.append(anuga_rmse)
                            if anuga_mse is not None:
                                mse_values.append(anuga_mse)
                    
                    if has_historical:
                        hist_rmse = self.results.get("historical", {}).get("stage", {}).get("rmse")
                        hist_mse = self.results.get("historical", {}).get("stage", {}).get("mse")
                        if hist_rmse is not None:
                            data_sources.append("Historical")
                            rmse_values.append(hist_rmse)
                            if hist_mse is not None:
                                mse_values.append(hist_mse)
                    
                    if has_real_world:
                        rw_rmse = self.results.get("real_world", {}).get("stage", {}).get("rmse")
                        rw_mse = self.results.get("real_world", {}).get("stage", {}).get("mse")
                        if rw_rmse is not None:
                            data_sources.append("Real-World")
                            rmse_values.append(rw_rmse)
                            if rw_mse is not None:
                                mse_values.append(rw_mse)
                    
                    # Create plot if we have data
                    if data_sources and rmse_values:
                        x = np.arange(len(data_sources))
                        width = 0.35
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        rmse_bars = ax.bar(x - width/2, rmse_values, width, label='RMSE')
                        
                        # Only add MSE bars if we have values
                        if mse_values and len(mse_values) == len(data_sources):
                            mse_bars = ax.bar(x + width/2, mse_values, width, label='MSE')
                        
                        ax.set_ylabel('Error Value')
                        ax.set_title('PINN Model Error Metrics by Data Source')
                        ax.set_xticks(x)
                        ax.set_xticklabels(data_sources)
                        ax.legend()
                        
                        # Add value labels on bars
                        for i, v in enumerate(rmse_values):
                            ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
                        
                        if mse_values and len(mse_values) == len(data_sources):
                            for i, v in enumerate(mse_values):
                                ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f"error_metrics_comparison_{timestamp}.png"), dpi=300)
                        plt.close()
                        
                        logger.info(f"Saved error metrics comparison plot")
            except Exception as e:
                logger.error(f"Error generating error metrics comparison plot: {str(e)}")
            
            # 2. Classification metrics comparison (precision/recall/F1)
            try:
                if any([has_anuga, has_historical, has_real_world]):
                    # Collect classification metrics from each data source
                    data_sources = []
                    precision_values = []
                    recall_values = []
                    f1_values = []
                    
                    # Safely access metrics with dictionary get method
                    if has_anuga:
                        anuga_precision = self.results.get("anuga", {}).get("classification", {}).get("precision")
                        anuga_recall = self.results.get("anuga", {}).get("classification", {}).get("recall")
                        anuga_f1 = self.results.get("anuga", {}).get("classification", {}).get("f1_score")
                        
                        if anuga_precision is not None and anuga_recall is not None and anuga_f1 is not None:
                            data_sources.append("ANUGA")
                            precision_values.append(anuga_precision)
                            recall_values.append(anuga_recall)
                            f1_values.append(anuga_f1)
                    
                    if has_historical:
                        hist_precision = self.results.get("historical", {}).get("classification", {}).get("precision")
                        hist_recall = self.results.get("historical", {}).get("classification", {}).get("recall")
                        hist_f1 = self.results.get("historical", {}).get("classification", {}).get("f1_score")
                        
                        if hist_precision is not None and hist_recall is not None and hist_f1 is not None:
                            data_sources.append("Historical")
                            precision_values.append(hist_precision)
                            recall_values.append(hist_recall)
                            f1_values.append(hist_f1)
                    
                    if has_real_world:
                        rw_precision = self.results.get("real_world", {}).get("classification", {}).get("precision")
                        rw_recall = self.results.get("real_world", {}).get("classification", {}).get("recall")
                        rw_f1 = self.results.get("real_world", {}).get("classification", {}).get("f1_score")
                        
                        if rw_precision is not None and rw_recall is not None and rw_f1 is not None:
                            data_sources.append("Real-World")
                            precision_values.append(rw_precision)
                            recall_values.append(rw_recall)
                            f1_values.append(rw_f1)
                    
                    # Create plot if we have data
                    if data_sources and precision_values and recall_values and f1_values:
                        x = np.arange(len(data_sources))
                        width = 0.25
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        precision_bars = ax.bar(x - width, precision_values, width, label='Precision')
                        recall_bars = ax.bar(x, recall_values, width, label='Recall')
                        f1_bars = ax.bar(x + width, f1_values, width, label='F1-Score')
                        
                        ax.set_ylabel('Metric Value')
                        ax.set_title('PINN Model Classification Metrics by Data Source')
                        ax.set_xticks(x)
                        ax.set_xticklabels(data_sources)
                        ax.legend()
                        
                        # Set y-axis limits to 0-1 for percentage metrics
                        ax.set_ylim(0, 1.1)
                        
                        # Add value labels on bars
                        for bars, values in zip([precision_bars, recall_bars, f1_bars], 
                                              [precision_values, recall_values, f1_values]):
                            for bar, value in zip(bars, values):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                      f'{value:.2f}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f"classification_metrics_{timestamp}.png"), dpi=300)
                        plt.close()
                        
                        logger.info(f"Saved classification metrics comparison plot")
            except Exception as e:
                logger.error(f"Error generating classification metrics plot: {str(e)}")
            
            # 3. Temporal RMSE plot (if available)
            try:
                # Check if we have temporal RMSE from ANUGA validation
                if has_anuga:
                    temporal_rmse = self.results.get("anuga", {}).get("stage", {}).get("temporal_rmse")
                    
                    if temporal_rmse is not None and isinstance(temporal_rmse, (list, np.ndarray)) and len(temporal_rmse) > 1:
                        plt.figure(figsize=(10, 6))
                        
                        # Create x-axis (timesteps)
                        timesteps = np.arange(len(temporal_rmse))
                        
                        # Plot temporal RMSE
                        plt.plot(timesteps, temporal_rmse, marker='o', linestyle='-', linewidth=2)
                        plt.xlabel('Timestep')
                        plt.ylabel('RMSE')
                        plt.title('PINN Model RMSE Over Time')
                        plt.grid(True)
                        
                        # Add trendline
                        z = np.polyfit(timesteps, temporal_rmse, 1)
                        p = np.poly1d(z)
                        plt.plot(timesteps, p(timesteps), "r--", alpha=0.8, 
                                label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
                        
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f"temporal_rmse_{timestamp}.png"), dpi=300)
                        plt.close()
                        
                        logger.info(f"Saved temporal RMSE plot")
            except Exception as e:
                logger.error(f"Error generating temporal RMSE plot: {str(e)}")
            
            # 4. Precision-Recall Curves
            try:
                # Generate precision-recall curves if available
                for source in ['anuga', 'historical', 'real_world']:
                    # Safely check if source is available
                    source_available = source in self.results and not self.results.get(source, {}).get("error", None)
                    
                    if source_available:
                        # Safely access PR curve data
                        precision_curve = self.results.get(source, {}).get("classification", {}).get("precision_curve")
                        recall_curve = self.results.get(source, {}).get("classification", {}).get("recall_curve")
                        
                        if precision_curve is not None and recall_curve is not None:
                            # Create plot
                            plt.figure(figsize=(8, 8))
                            plt.plot(recall_curve, precision_curve, lw=2)
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.title(f'Precision-Recall Curve ({source.capitalize()} Data)')
                            plt.grid(True)
                            
                            # Add AUC if available
                            try:
                                if len(precision_curve) == len(recall_curve) and len(precision_curve) > 1:
                                    # Calculate AUC manually
                                    auc = np.trapz(precision_curve, recall_curve)
                                    plt.text(0.6, 0.2, f'AUC = {auc:.4f}', bbox=dict(facecolor='white', alpha=0.8))
                            except Exception as e:
                                logger.warning(f"Could not calculate AUC for {source} PR curve: {str(e)}")
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f"pr_curve_{source}_{timestamp}.png"), dpi=300)
                            plt.close()
                            
                            logger.info(f"Saved precision-recall curve for {source} data")
            except Exception as e:
                logger.error(f"Error generating precision-recall curves: {str(e)}")
            
            # 5. K-fold cross-validation results (if available)
            try:
                if has_real_world and "cv_results" in self.results.get("real_world", {}):
                    cv_results = self.results["real_world"]["cv_results"]
                    
                    if cv_results and isinstance(cv_results, dict):
                        # Extract fold results
                        folds = sorted(int(k.split('_')[1]) for k in cv_results.keys() if k.startswith('fold_'))
                        
                        if folds:
                            # Collect metrics across folds
                            rmse_values = []
                            f1_values = []
                            
                            for fold in folds:
                                fold_key = f'fold_{fold}'
                                if fold_key in cv_results:
                                    fold_rmse = cv_results[fold_key].get('stage', {}).get('rmse')
                                    if fold_rmse is not None:
                                        rmse_values.append(fold_rmse)
                                    
                                    fold_f1 = cv_results[fold_key].get('classification', {}).get('f1_score')
                                    if fold_f1 is not None:
                                        f1_values.append(fold_f1)
                            
                            # Create cross-validation performance plot
                            if rmse_values or f1_values:
                                fig, ax1 = plt.subplots(figsize=(10, 6))
                                
                                # Plot RMSE on primary y-axis
                                if rmse_values:
                                    color = 'tab:blue'
                                    ax1.set_xlabel('Fold')
                                    ax1.set_ylabel('RMSE', color=color)
                                    ax1.plot(folds, rmse_values, marker='o', color=color, label='RMSE')
                                    ax1.tick_params(axis='y', labelcolor=color)
                                
                                # Plot F1-score on secondary y-axis if available
                                if f1_values:
                                    if rmse_values:
                                        ax2 = ax1.twinx()
                                        color = 'tab:red'
                                        ax2.set_ylabel('F1-Score', color=color)
                                        ax2.plot(folds, f1_values, marker='s', color=color, label='F1-Score')
                                        ax2.tick_params(axis='y', labelcolor=color)
                                        ax2.set_ylim(0, 1.1)  # F1-score is between 0 and 1
                                    else:
                                        # If no RMSE, just plot F1 on primary axis
                                        ax1.set_xlabel('Fold')
                                        ax1.set_ylabel('F1-Score')
                                        ax1.plot(folds, f1_values, marker='s', color='tab:red', label='F1-Score')
                                        ax1.set_ylim(0, 1.1)
                                
                                # Add mean values as horizontal lines
                                if rmse_values:
                                    mean_rmse = np.mean(rmse_values)
                                    ax1.axhline(y=mean_rmse, color='tab:blue', linestyle='--', 
                                              alpha=0.7, label=f'Mean RMSE: {mean_rmse:.4f}')
                                
                                if f1_values and rmse_values:
                                    mean_f1 = np.mean(f1_values)
                                    ax2.axhline(y=mean_f1, color='tab:red', linestyle='--', 
                                              alpha=0.7, label=f'Mean F1: {mean_f1:.4f}')
                                elif f1_values:
                                    mean_f1 = np.mean(f1_values)
                                    ax1.axhline(y=mean_f1, color='tab:red', linestyle='--', 
                                              alpha=0.7, label=f'Mean F1: {mean_f1:.4f}')
                                
                                # Set integer ticks for folds
                                plt.xticks(folds)
                                
                                # Add legend
                                lines1, labels1 = ax1.get_legend_handles_labels()
                                if rmse_values and f1_values:
                                    lines2, labels2 = ax2.get_legend_handles_labels()
                                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                                else:
                                    ax1.legend(loc='best')
                                
                                plt.title('Cross-Validation Performance Metrics')
                                plt.tight_layout()
                                plt.savefig(os.path.join(vis_dir, f"cross_validation_{timestamp}.png"), dpi=300)
                                plt.close()
                                
                                logger.info(f"Saved cross-validation performance plot")
            except Exception as e:
                logger.error(f"Error generating cross-validation plot: {str(e)}")
            
            # Log visualization completion
            logger.info(f"Validation visualizations saved to {vis_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            
        return

def main():
    parser = argparse.ArgumentParser(description="Validate PINN Flood Model")
    parser.add_argument("--model_path", required=True, help="Path to trained PINN model")
    parser.add_argument("--anuga_data", help="Path to ANUGA simulation data")
    parser.add_argument("--historical_data", help="Path to historical flood data")
    parser.add_argument("--real_world_data", help="Path to real-world data")
    parser.add_argument("--output_dir", default="validation_results", help="Directory to save validation results")
    parser.add_argument("--flood_threshold", type=float, default=0.1, help="Threshold for flood classification (meters)")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Load PINN model
        logger.info(f"Loading PINN model from {args.model_path}")
        model = load_model(args.model_path)
        
        # Initialize validator
        validator = PINNModelValidator(model, config)
        
        # Validate against different data sources
        if args.anuga_data:
            validator.validate_against_anuga(args.anuga_data, flood_threshold=args.flood_threshold)
        
        if args.historical_data:
            validator.validate_against_historical(args.historical_data, flood_threshold=args.flood_threshold)
        
        if args.real_world_data:
            validator.cross_validate_real_world(
                args.real_world_data, 
                k_folds=args.k_folds,
                flood_threshold=args.flood_threshold
            )
        
        # Save results
        validator.save_results(args.output_dir)
        
        logger.info("Validation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 