import numpy as np
import torch
from sklearn.metrics import mean_squared_error, f1_score, precision_recall_fscore_support, precision_recall_curve, auc
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import logging

logger = logging.getLogger(__name__)

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error between observed and predicted values
    
    Parameters:
    y_true: Array of actual values
    y_pred: Array of predicted values
    
    Returns:
    RMSE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Check for empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("Empty array detected in RMSE calculation")
        return np.nan
    
    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        logger.warning("NaN values detected in RMSE calculation")
        # Remove NaN values if any
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if y_true.size == 0:
            return np.nan
    
    # Check if shapes match
    if y_true.shape != y_pred.shape:
        logger.warning(f"Shape mismatch in RMSE calculation: {y_true.shape} vs {y_pred.shape}")
        # Try to reshape if possible
        try:
            y_pred = y_pred.reshape(y_true.shape)
        except ValueError:
            # If reshaping is not possible, flatten both arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            # Truncate to same length if needed
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
    
    return sqrt(mean_squared_error(y_true, y_pred))

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error between observed and predicted values
    
    Parameters:
    y_true: Array of actual values
    y_pred: Array of predicted values
    
    Returns:
    MSE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Check for empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("Empty array detected in MSE calculation")
        return np.nan
    
    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        logger.warning("NaN values detected in MSE calculation")
        # Remove NaN values if any
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if y_true.size == 0:
            return np.nan
    
    # Check if shapes match
    if y_true.shape != y_pred.shape:
        logger.warning(f"Shape mismatch in MSE calculation: {y_true.shape} vs {y_pred.shape}")
        # Try to reshape if possible
        try:
            y_pred = y_pred.reshape(y_true.shape)
        except ValueError:
            # If reshaping is not possible, flatten both arrays
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            # Truncate to same length if needed
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
    
    return mean_squared_error(y_true, y_pred)

def calculate_ssim(img_true, img_pred, data_range=None):
    """
    Calculate Structural Similarity Index for spatial accuracy comparison
    
    Parameters:
    img_true: Reference image (2D or 3D array)
    img_pred: Predicted image (2D or 3D array)
    data_range: Data range of the images (if None, will be determined from the images)
    
    Returns:
    SSIM value (-1 to 1, where 1 means perfect similarity)
    """
    if isinstance(img_true, torch.Tensor):
        img_true = img_true.detach().cpu().numpy()
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.detach().cpu().numpy()
    
    # Check for empty arrays
    if img_true.size == 0 or img_pred.size == 0:
        logger.warning("Empty array detected in SSIM calculation")
        return np.nan
    
    # Check for NaN values
    if np.isnan(img_true).any() or np.isnan(img_pred).any():
        logger.warning("NaN values detected in SSIM calculation")
        # SSIM can't handle NaN values, so we need to replace them
        img_true = np.nan_to_num(img_true)
        img_pred = np.nan_to_num(img_pred)
    
    # Check shape compatibility
    if img_true.shape != img_pred.shape:
        logger.warning(f"Shape mismatch in SSIM calculation: {img_true.shape} vs {img_pred.shape}")
        try:
            # Try to resize the predicted image to match the true image
            from skimage.transform import resize
            img_pred = resize(img_pred, img_true.shape, preserve_range=True)
        except Exception as e:
            logger.error(f"Failed to resize images for SSIM calculation: {str(e)}")
            return np.nan
    
    # If data_range is not provided, calculate it
    if data_range is None:
        data_range = max(np.nanmax(img_true) - np.nanmin(img_true), 
                          np.nanmax(img_pred) - np.nanmin(img_pred))
        if data_range == 0:
            logger.warning("Data range is 0, setting to 1 to avoid division by zero")
            data_range = 1
    
    try:
        return ssim(img_true, img_pred, data_range=data_range)
    except Exception as e:
        logger.error(f"SSIM calculation failed: {str(e)}")
        return np.nan

def calculate_classification_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate classification metrics for flood event detection
    
    Parameters:
    y_true: Binary ground truth (flooded=1, not flooded=0)
    y_pred_prob: Predicted probabilities or values
    threshold: Classification threshold
    
    Returns:
    Dictionary with precision, recall, and F1 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
    
    # Check for empty arrays
    if y_true.size == 0 or y_pred_prob.size == 0:
        logger.warning("Empty array detected in classification metrics calculation")
        return {'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}
    
    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred_prob).any():
        logger.warning("NaN values detected in classification metrics calculation")
        # Remove NaN values if any
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_prob))
        y_true = y_true[mask]
        y_pred_prob = y_pred_prob[mask]
        
        if y_true.size == 0:
            return {'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}
    
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    # If different lengths, truncate to the shortest
    if len(y_true) != len(y_pred_prob):
        logger.warning(f"Length mismatch in classification metrics: {len(y_true)} vs {len(y_pred_prob)}")
        min_len = min(len(y_true), len(y_pred_prob))
        y_true = y_true[:min_len]
        y_pred_prob = y_pred_prob[:min_len]
    
    # Convert continuous values to binary predictions using threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Check for class imbalance
    class_counts = np.bincount(y_true.astype(int))
    if len(class_counts) < 2 or 0 in class_counts:
        logger.warning(f"Single class or missing class detected: class counts = {class_counts}")
        
        # If only one class exists in the data, return default values
        if len(np.unique(y_true)) == 1:
            if np.unique(y_true)[0] == np.unique(y_pred)[0]:
                # If prediction equals the only true class, perfect match
                return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
            else:
                # If prediction doesn't match the true class, zero accuracy
                return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    try:
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        logger.error(f"Classification metrics calculation failed: {str(e)}")
        return {'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}

def calculate_precision_recall_curve(y_true, y_pred_prob):
    """
    Calculate precision-recall curve and area under the curve
    
    Parameters:
    y_true: Binary ground truth (flooded=1, not flooded=0)
    y_pred_prob: Predicted probabilities
    
    Returns:
    Dictionary with precision, recall, thresholds, and AUC
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
    
    # Check for empty arrays
    if y_true.size == 0 or y_pred_prob.size == 0:
        logger.warning("Empty array detected in precision-recall curve calculation")
        return {'precision': np.array([]), 'recall': np.array([]), 'thresholds': np.array([]), 'auc': np.nan}
    
    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred_prob).any():
        logger.warning("NaN values detected in precision-recall curve calculation")
        # Remove NaN values if any
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_prob))
        y_true = y_true[mask]
        y_pred_prob = y_pred_prob[mask]
        
        if y_true.size == 0:
            return {'precision': np.array([]), 'recall': np.array([]), 'thresholds': np.array([]), 'auc': np.nan}
    
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    # If different lengths, truncate to the shortest
    if len(y_true) != len(y_pred_prob):
        logger.warning(f"Length mismatch in PR curve: {len(y_true)} vs {len(y_pred_prob)}")
        min_len = min(len(y_true), len(y_pred_prob))
        y_true = y_true[:min_len]
        y_pred_prob = y_pred_prob[:min_len]
    
    # Check for binary labels
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(f"Single class detected: {unique_labels}. PR curve requires both positive and negative samples.")
        return {'precision': np.array([0.0]), 'recall': np.array([0.0]), 'thresholds': np.array([]), 'auc': 0.0}
    
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc': pr_auc
        }
    except Exception as e:
        logger.error(f"Precision-recall curve calculation failed: {str(e)}")
        return {'precision': np.array([]), 'recall': np.array([]), 'thresholds': np.array([]), 'auc': np.nan}

def create_binary_flood_mask(water_depth, threshold=0.1):
    """
    Create binary flood mask from water depth
    
    Parameters:
    water_depth: Water depth values
    threshold: Flood threshold in meters
    
    Returns:
    Binary mask where 1 indicates flooded areas
    """
    # Check for NaN values
    if isinstance(water_depth, np.ndarray) and np.isnan(water_depth).any():
        logger.warning("NaN values detected in water depth, replacing with zeros")
        water_depth = np.nan_to_num(water_depth, nan=0.0)
    
    if isinstance(water_depth, torch.Tensor):
        if torch.isnan(water_depth).any():
            logger.warning("NaN values detected in water depth tensor, replacing with zeros")
            water_depth = torch.nan_to_num(water_depth, nan=0.0)
        return (water_depth >= threshold).int()
    else:
        return (water_depth >= threshold).astype(int) 