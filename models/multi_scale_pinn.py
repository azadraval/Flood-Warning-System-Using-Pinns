"""
Multi-scale Physics-Informed Neural Network for flood modeling.

This module integrates neural operators (FNO, GNN) with physics-informed
constraints to create a hybrid model for flood prediction and forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from flood_warning_system.models.neural_operators import (
    FourierNeuralOperator2D,
    GraphNeuralOperator,
    MultiScaleNeuralOperator,
    PhysicsInformedNeuralOperator
)

class MultiScalePINN(nn.Module):
    """
    Multi-scale Physics-Informed Neural Network for flood modeling.
    
    This class combines neural operators with physics constraints at multiple
    scales to create a comprehensive flood prediction model.
    """
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        hidden_channels: int = 64,
        fno_modes: Tuple[int, int] = (12, 12),
        num_scales: int = 3,
        use_gnn: bool = True,
        grid_size: Tuple[int, int] = (64, 64),
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 0.1,
        gravity: float = 9.81,
        manning_coef: float = 0.035,
        physics_weight: float = 0.1,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Physics loss parameters
        adaptive_weighting: bool = True,
        min_physics_weight: float = 0.01,
        max_physics_weight: float = 1.0,
        adaptation_rate: float = 0.05,
        continuity_weight: float = 1.0,
        x_momentum_weight: float = 0.5,
        y_momentum_weight: float = 0.5,
        boundary_weight: float = 0.5,
        enforce_positivity: bool = True
    ):
        """
        Initialize the Multi-scale Physics-Informed Neural Network.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            hidden_channels: Number of hidden channels
            fno_modes: Number of Fourier modes in each dimension
            num_scales: Number of scales for the multi-scale operators
            use_gnn: Whether to use graph neural networks for irregular domains
            grid_size: Size of the grid (height, width)
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            dt: Time step
            gravity: Gravity constant
            manning_coef: Manning's roughness coefficient
            physics_weight: Weight for physics-informed loss
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            device: Device to use for computations
            adaptive_weighting: Whether to use adaptive weighting for physics loss
            min_physics_weight: Minimum weight for physics loss
            max_physics_weight: Maximum weight for physics loss
            adaptation_rate: Rate at which to adapt physics weight
            continuity_weight: Weight for continuity equation
            x_momentum_weight: Weight for x-momentum equation
            y_momentum_weight: Weight for y-momentum equation
            boundary_weight: Weight for boundary conditions
            enforce_positivity: Whether to enforce positive water depth
        """
        super(MultiScalePINN, self).__init__()
        
        # Initialize the neural operator models
        self.operator = MultiScaleNeuralOperator(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            fno_modes=fno_modes,
            num_scales=num_scales,
            use_gnn=use_gnn,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # Initialize the boundary condition handler
        self.boundary_handler = BoundaryConditionHandler(grid_size, input_channels)
        
        # Initialize the physics loss module
        self.physics_loss = PhysicsLossModule(
            dx=dx,
            dy=dy,
            dt=dt,
            gravity=gravity,
            manning_coef=manning_coef,
            initial_physics_weight=physics_weight,
            adaptive_weighting=adaptive_weighting,
            min_physics_weight=min_physics_weight,
            max_physics_weight=max_physics_weight,
            adaptation_rate=adaptation_rate,
            continuity_weight=continuity_weight,
            x_momentum_weight=x_momentum_weight,
            y_momentum_weight=y_momentum_weight,
            boundary_weight=boundary_weight,
            enforce_positivity=enforce_positivity
        )
        
        # Store parameters
        self.grid_size = grid_size
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Create spatial coordinate grid for position encoding
        self.register_buffer('coord_grid', self.create_coordinate_grid(grid_size))
        
        # Move to device
        self.to(device)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Save hyperparameters for model loading
        self.hparams = {
            'input_channels': input_channels,
            'output_channels': output_channels,
            'hidden_channels': hidden_channels,
            'fno_modes': fno_modes,
            'num_scales': num_scales,
            'use_gnn': use_gnn,
            'grid_size': grid_size,
            'dx': dx,
            'dy': dy,
            'dt': dt,
            'gravity': gravity,
            'manning_coef': manning_coef,
            'physics_weight': physics_weight,
            'dropout': dropout,
            'use_batch_norm': use_batch_norm,
            'adaptive_weighting': adaptive_weighting,
            'min_physics_weight': min_physics_weight,
            'max_physics_weight': max_physics_weight,
            'adaptation_rate': adaptation_rate,
            'continuity_weight': continuity_weight,
            'x_momentum_weight': x_momentum_weight,
            'y_momentum_weight': y_momentum_weight,
            'boundary_weight': boundary_weight,
            'enforce_positivity': enforce_positivity
        }
    
    def create_coordinate_grid(self, grid_size):
        """
        Create a normalized coordinate grid for positional encoding.
        
        Args:
            grid_size: Tuple of (height, width)
        """
        height, width = grid_size
        
        # Create meshgrid of normalized coordinates
        y_coords = torch.linspace(-1, 1, height)
        x_coords = torch.linspace(-1, 1, width)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack coordinates and add batch dimension
        self.coord_grid = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
        self.register_buffer('grid_coords', self.coord_grid)
    
    def forward(self, x, boundaries=None):
        """
        Forward pass of the multi-scale PINN.
        
        Args:
            x: Input tensor [batch, channels, height, width]
                channels typically: [h, u, v, elevation]
            boundaries: Optional boundary condition tensor
            
        Returns:
            Output tensor with predicted fields [batch, channels, height, width]
        """
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        
        # Create coordinate grid for this batch
        grid = self.grid_coords.expand(batch_size, -1, -1, -1)
        
        # Forward pass through neural operator
        prediction = self.operator(
            x, 
            grid=grid, 
            dx=self.dx, 
            dy=self.dy, 
            dt=self.dt
        )
        
        # Apply boundary conditions if provided
        if boundaries is not None:
            prediction = self.boundary_handler(prediction, boundaries)
        
        return prediction
    
    def compute_physics_loss(self, predictions, inputs, elevation=None):
        """
        Compute physics-informed loss based on shallow water equations.
        
        This implements a comprehensive physics-informed loss incorporating:
        1. Continuity equation (mass conservation)
        2. Momentum equations with friction and bed slope terms
        3. Boundary conditions enforcement
        
        Args:
            predictions: Model predictions [batch, channels, height, width]
            inputs: Input conditions [batch, channels, height, width]
            elevation: Optional elevation data [batch, 1, height, width]
            
        Returns:
            Dictionary of loss components and combined physics loss
        """
        batch_size, num_channels, height, width = predictions.shape
        
        # Extract variables from predictions
        h_pred = predictions[:, 0:1, :, :]  # Water depth
        qx_pred = predictions[:, 1:2, :, :]  # x-momentum (h*u)
        qy_pred = predictions[:, 2:3, :, :]  # y-momentum (h*v)
        
        # Extract previous state from inputs
        h_prev = inputs[:, 0:1, :, :]
        qx_prev = inputs[:, 1:2, :, :]
        qy_prev = inputs[:, 2:3, :, :]
        
        # Compute velocities for momentum terms
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        u_pred = qx_pred / (h_pred + eps)
        v_pred = qy_pred / (h_pred + eps)
        
        # Apply physical constraint - water depth cannot be negative
        h_pred = torch.clamp(h_pred, min=0.0)
        
        # Initialize loss components
        loss_dict = {}
        
        # --------------- Continuity Equation (Mass Conservation) ---------------
        # dh/dt + d(h*u)/dx + d(h*v)/dy = 0
        
        # Calculate time derivative: dh/dt
        dh_dt = (h_pred - h_prev) / self.dt
        
        # Spatial derivatives using finite differences
        # Central difference for interior points
        dqx_dx = torch.zeros_like(h_pred)
        dqy_dy = torch.zeros_like(h_pred)
        
        # x-direction derivative: d(h*u)/dx
        dqx_dx[:, :, :, 1:-1] = (qx_pred[:, :, :, 2:] - qx_pred[:, :, :, :-2]) / (2 * self.dx)
        
        # y-direction derivative: d(h*v)/dy
        dqy_dy[:, :, 1:-1, :] = (qy_pred[:, :, 2:, :] - qy_pred[:, :, :-2, :]) / (2 * self.dy)
        
        # Continuity equation residual
        mass_residual = dh_dt + dqx_dx + dqy_dy
        
        # Calculate continuity loss (squared residual)
        continuity_loss = torch.mean(mass_residual**2)
        loss_dict["continuity"] = continuity_loss
        
        # --------------- Momentum Equations ---------------
        # x-momentum: d(h*u)/dt + d(h*u²)/dx + d(h*u*v)/dy + g*h*dz/dx + g*n²*u*√(u²+v²)/h^(1/3) = 0
        # y-momentum: d(h*v)/dt + d(h*u*v)/dx + d(h*v²)/dy + g*h*dz/dy + g*n²*v*√(u²+v²)/h^(1/3) = 0
        
        if elevation is not None:
            # Calculate bed slope terms
            dz_dx = torch.zeros_like(h_pred)
            dz_dy = torch.zeros_like(h_pred)
            
            # Compute bed slopes using central difference
            dz_dx[:, :, :, 1:-1] = (elevation[:, :, :, 2:] - elevation[:, :, :, :-2]) / (2 * self.dx)
            dz_dy[:, :, 1:-1, :] = (elevation[:, :, 2:, :] - elevation[:, :, :-2, :]) / (2 * self.dy)
            
            # Time derivatives
            dqx_dt = (qx_pred - qx_prev) / self.dt
            dqy_dt = (qy_pred - qy_prev) / self.dt
            
            # Compute flux derivatives
            # For x-momentum: d(h*u²)/dx + d(h*u*v)/dy
            # For y-momentum: d(h*u*v)/dx + d(h*v²)/dy
            
            # Calculate fluxes
            flux_xx = h_pred * u_pred * u_pred  # h*u²
            flux_xy = h_pred * u_pred * v_pred  # h*u*v
            flux_yy = h_pred * v_pred * v_pred  # h*v²
            
            # Compute spatial derivatives of fluxes
            dflux_xx_dx = torch.zeros_like(h_pred)
            dflux_xy_dy = torch.zeros_like(h_pred)
            dflux_xy_dx = torch.zeros_like(h_pred)
            dflux_yy_dy = torch.zeros_like(h_pred)
            
            # x-direction derivatives
            dflux_xx_dx[:, :, :, 1:-1] = (flux_xx[:, :, :, 2:] - flux_xx[:, :, :, :-2]) / (2 * self.dx)
            dflux_xy_dx[:, :, :, 1:-1] = (flux_xy[:, :, :, 2:] - flux_xy[:, :, :, :-2]) / (2 * self.dx)
            
            # y-direction derivatives
            dflux_xy_dy[:, :, 1:-1, :] = (flux_xy[:, :, 2:, :] - flux_xy[:, :, :-2, :]) / (2 * self.dy)
            dflux_yy_dy[:, :, 1:-1, :] = (flux_yy[:, :, 2:, :] - flux_yy[:, :, :-2, :]) / (2 * self.dy)
            
            # Compute friction terms (Manning's equation)
            # Sf = n²*u*√(u²+v²)/h^(4/3)
            vel_magnitude = torch.sqrt(u_pred**2 + v_pred**2)
            h_powered = h_pred**(4/3)
            h_powered = torch.where(h_powered > eps, h_powered, torch.ones_like(h_powered) * eps)
            
            friction_x = self.g * self.n**2 * u_pred * vel_magnitude / h_powered
            friction_y = self.g * self.n**2 * v_pred * vel_magnitude / h_powered
            
            # Calculate pressure terms
            pressure_x = self.g * h_pred * dz_dx
            pressure_y = self.g * h_pred * dz_dy
            
            # Momentum residuals
            x_momentum_residual = dqx_dt + dflux_xx_dx + dflux_xy_dy + pressure_x + friction_x
            y_momentum_residual = dqy_dt + dflux_xy_dx + dflux_yy_dy + pressure_y + friction_y
            
            # Calculate momentum loss (squared residuals)
            x_momentum_loss = torch.mean(x_momentum_residual**2)
            y_momentum_loss = torch.mean(y_momentum_residual**2)
            
            loss_dict["x_momentum"] = x_momentum_loss
            loss_dict["y_momentum"] = y_momentum_loss
            
            # Apply adaptive weighting for momentum equations
            momentum_loss = x_momentum_loss + y_momentum_loss
        else:
            # Without elevation data, we only enforce continuity
            momentum_loss = torch.tensor(0.0, device=h_pred.device)
            loss_dict["momentum"] = momentum_loss
        
        # --------------- Combined Physics Loss ---------------
        # Total physics loss with current weights
        physics_loss = self.physics_weight * (continuity_loss + momentum_loss)
        loss_dict["total_physics"] = physics_loss
        
        return loss_dict
    
    def training_step(self, batch, return_predictions=False):
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing:
                - inputs: Tensor of shape [batch, channels, height, width]
                - targets: Tensor of shape [batch, channels, height, width]
                - elevation: Optional tensor of shape [batch, 1, height, width]
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing loss components and optionally predictions
        """
        # Extract data from batch
        inputs = batch["inputs"]
        targets = batch["targets"]
        elevation = batch.get("elevation", None)
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Data-driven loss (L2 loss between predictions and targets)
        data_loss = F.mse_loss(predictions, targets)
        
        # Physics-informed loss
        physics_losses = self.compute_physics_loss(predictions, inputs, elevation)
        physics_loss = physics_losses["total_physics"]
        
        # Adaptive weighting between physics and data losses
        # Implement dynamic balancing based on loss magnitudes
        with torch.no_grad():
            # Calculate the ratio of the losses
            loss_ratio = data_loss / (physics_loss + 1e-8)
            
            # Adjust weights to balance the losses
            if loss_ratio > 10.0:
                # Data loss is much larger, increase physics weight
                self.physics_weight = min(self.physics_weight * 1.05, 1.0)
            elif loss_ratio < 0.1:
                # Physics loss is much larger, decrease physics weight
                self.physics_weight = max(self.physics_weight * 0.95, 0.01)
        
        # Combined loss
        total_loss = data_loss + physics_loss
        
        # Prepare result dictionary
        result = {
            "loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "physics_weight": self.physics_weight
        }
        
        # Add detailed physics loss components
        for key, value in physics_losses.items():
            if key != "total_physics":  # Already included as physics_loss
                result[f"physics_{key}"] = value
        
        if return_predictions:
            result["predictions"] = predictions
            
        return result
    
    def validation_step(self, batch):
        """
        Perform a validation step.
        
        Args:
            batch: Dictionary containing validation data
            
        Returns:
            Dictionary with losses and predictions
        """
        # Run training step with return_predictions=True
        return self.training_step(batch, return_predictions=True)
    
    def save_model(self, save_dir, filename="multi_scale_pinn_model.pt"):
        """
        Save the model to disk.
        
        Args:
            save_dir: Directory to save the model to
            filename: Name of the file to save the model to
        """
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        # Save model state dict and hyperparameters
        save_dict = {
            "model_state_dict": self.state_dict(),
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "hidden_channels": self.operator.hidden_channels,
            "grid_size": self.grid_size,
            "use_gnn": self.operator.use_gnn,
            "dx": self.dx,
            "dy": self.dy,
            "dt": self.dt,
            "gravity": self.g,
            "manning_coef": self.n,
            "physics_weight": self.physics_weight,
            "use_batch_norm": self.operator.use_batch_norm,
            "hparams": self.hparams
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load saved dictionary
        save_dict = torch.load(filepath, map_location=device)
        
        # Create model with saved hyperparameters
        model = cls(
            input_channels=save_dict["input_channels"],
            output_channels=save_dict["output_channels"],
            hidden_channels=save_dict["hidden_channels"],
            grid_size=save_dict["grid_size"],
            use_gnn=save_dict["use_gnn"],
            dx=save_dict["dx"],
            dy=save_dict["dy"],
            dt=save_dict["dt"],
            gravity=save_dict["gravity"],
            manning_coef=save_dict["manning_coef"],
            physics_weight=save_dict["physics_weight"],
            use_batch_norm=save_dict["use_batch_norm"],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(save_dict["model_state_dict"])
        model.to(device)
        
        return model
    
    def predict(self, inputs, elevation=None, boundaries=None, num_steps=1):
        """
        Make predictions for multiple time steps.
        
        Args:
            inputs: Initial conditions [batch, channels, height, width]
            elevation: Optional elevation data [batch, 1, height, width]
            boundaries: Optional boundary conditions
            num_steps: Number of time steps to predict
            
        Returns:
            Predictions for each time step [num_steps, batch, channels, height, width]
        """
        self.eval()  # Set model to evaluation mode
        
        # Initialize predictions list
        predictions = []
        
        # Initial input
        current_state = inputs
        
        # Make predictions for each time step
        with torch.no_grad():
            for t in range(num_steps):
                # Prepare input for current step
                if elevation is not None:
                    model_input = torch.cat([current_state, elevation], dim=1)
                else:
                    model_input = current_state
                
                # Forward pass
                prediction = self.forward(model_input, boundaries)
                
                # Handle invalid predictions
                # 1. Replace NaNs with zeros
                if torch.isnan(prediction).any():
                    prediction = torch.where(torch.isnan(prediction), 
                                           torch.zeros_like(prediction), 
                                           prediction)
                    logging.warning(f"NaN values detected and replaced in prediction at step {t}")
                
                # 2. Enforce physical constraints - water depth must be non-negative
                prediction[:, 0:1, :, :] = torch.clamp(prediction[:, 0:1, :, :], min=0.0)
                
                # 3. Clip extremely large values to prevent instability
                max_depth = 20.0  # Maximum reasonable water depth (in meters)
                max_velocity = 15.0  # Maximum reasonable velocity (in m/s)
                
                prediction[:, 0:1, :, :] = torch.clamp(prediction[:, 0:1, :, :], max=max_depth)
                
                # For momentum, we need to consider water depth when clamping
                # Get the water depth for scaling
                h = prediction[:, 0:1, :, :]
                
                # Scale max momentum by water depth
                max_momentum = h * max_velocity
                
                # Clamp the momentum
                prediction[:, 1:2, :, :] = torch.clamp(prediction[:, 1:2, :, :], 
                                                     min=-max_momentum, 
                                                     max=max_momentum)
                prediction[:, 2:3, :, :] = torch.clamp(prediction[:, 2:3, :, :], 
                                                     min=-max_momentum, 
                                                     max=max_momentum)
                
                # Store prediction
                predictions.append(prediction)
                
                # Update current state for next iteration
                current_state = prediction
        
        # Stack predictions along a new time dimension
        return torch.stack(predictions)
    
    def visualize_prediction(self, inputs, targets, elevation=None, boundaries=None,
                           save_path=None, show=True):
        """
        Visualize model predictions compared to targets.
        
        Args:
            inputs: Input data [batch, channels, height, width]
            targets: Target data [batch, channels, height, width]
            elevation: Optional elevation data [batch, 1, height, width]
            boundaries: Optional boundary conditions
            save_path: Path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Prepare input
            if elevation is not None:
                model_input = torch.cat([inputs, elevation], dim=1)
            else:
                model_input = inputs
            
            # Generate prediction
            prediction = self.forward(model_input, boundaries)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Get first batch item for visualization
        h_input = inputs[0, 0].cpu().numpy()
        u_input = inputs[0, 1].cpu().numpy()
        v_input = inputs[0, 2].cpu().numpy()
        
        h_target = targets[0, 0].cpu().numpy()
        u_target = targets[0, 1].cpu().numpy()
        v_target = targets[0, 2].cpu().numpy()
        
        h_pred = prediction[0, 0].cpu().numpy()
        u_pred = prediction[0, 1].cpu().numpy()
        v_pred = prediction[0, 2].cpu().numpy()
        
        # Plot water depth
        im1 = axes[0, 0].imshow(h_input, cmap='Blues')
        axes[0, 0].set_title('Input Water Depth (h)')
        fig.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(h_target, cmap='Blues')
        axes[0, 1].set_title('Target Water Depth (h)')
        fig.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(h_pred, cmap='Blues')
        axes[0, 2].set_title('Predicted Water Depth (h)')
        fig.colorbar(im3, ax=axes[0, 2])
        
        # Plot x-velocity
        im4 = axes[1, 0].imshow(u_input, cmap='coolwarm')
        axes[1, 0].set_title('Input x-velocity (u)')
        fig.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(u_target, cmap='coolwarm')
        axes[1, 1].set_title('Target x-velocity (u)')
        fig.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(u_pred, cmap='coolwarm')
        axes[1, 2].set_title('Predicted x-velocity (u)')
        fig.colorbar(im6, ax=axes[1, 2])
        
        # Plot y-velocity
        im7 = axes[2, 0].imshow(v_input, cmap='coolwarm')
        axes[2, 0].set_title('Input y-velocity (v)')
        fig.colorbar(im7, ax=axes[2, 0])
        
        im8 = axes[2, 1].imshow(v_target, cmap='coolwarm')
        axes[2, 1].set_title('Target y-velocity (v)')
        fig.colorbar(im8, ax=axes[2, 1])
        
        im9 = axes[2, 2].imshow(v_pred, cmap='coolwarm')
        axes[2, 2].set_title('Predicted y-velocity (v)')
        fig.colorbar(im9, ax=axes[2, 2])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path is not None:
            plt.savefig(save_path)
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig

    def visualize_physics_losses(self, inputs, targets, elevation=None,
                               save_path=None, show=True):
        """
        Visualize the physics-informed loss components.
        
        Args:
            inputs: Input data [batch, channels, height, width]
            targets: Target data [batch, channels, height, width]
            elevation: Optional elevation data [batch, 1, height, width]
            save_path: Path to save the visualization (optional)
            show: Whether to display the visualization
            
        Returns:
            Matplotlib figure
        """
        # Ensure inputs and targets are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        if elevation is not None:
            elevation = elevation.to(self.device)
        
        # Forward pass to get predictions
        with torch.no_grad():
            predictions = self.forward(inputs)
        
        # Extract variables from predictions and inputs
        h_pred = predictions[0, 0].cpu().numpy()  # Water depth
        qx_pred = predictions[0, 1].cpu().numpy()  # x-momentum
        qy_pred = predictions[0, 2].cpu().numpy()  # y-momentum
        
        h_prev = inputs[0, 0].cpu().numpy()
        qx_prev = inputs[0, 1].cpu().numpy()
        qy_prev = inputs[0, 2].cpu().numpy()
        
        # Compute velocities
        eps = 1e-6
        u_pred = qx_pred / (h_pred + eps)
        v_pred = qy_pred / (h_pred + eps)
        
        # Get physics loss components
        if hasattr(self, 'physics_loss'):
            loss_dict = self.physics_loss(predictions, inputs, targets, elevation)
        else:
            loss_dict = self.compute_physics_loss(predictions, inputs, elevation)
        
        # Create figure for visualization
        fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        
        # Plot water depth and velocities
        im0 = axs[0, 0].imshow(h_pred, cmap='Blues')
        axs[0, 0].set_title('Water Depth (h)')
        plt.colorbar(im0, ax=axs[0, 0])
        
        im1 = axs[0, 1].imshow(u_pred, cmap='RdBu_r')
        axs[0, 1].set_title('x-velocity (u)')
        plt.colorbar(im1, ax=axs[0, 1])
        
        im2 = axs[0, 2].imshow(v_pred, cmap='RdBu_r')
        axs[0, 2].set_title('y-velocity (v)')
        plt.colorbar(im2, ax=axs[0, 2])
        
        # Plot mass conservation terms
        # Compute the terms for visualization (using numpy for simplicity)
        dh_dt = (h_pred - h_prev) / self.dt
        
        dqx_dx = np.zeros_like(h_pred)
        dqy_dy = np.zeros_like(h_pred)
        
        # Interior points only
        dqx_dx[:, 1:-1] = (qx_pred[:, 2:] - qx_pred[:, :-2]) / (2 * self.dx)
        dqy_dy[1:-1, :] = (qy_pred[2:, :] - qy_pred[:-2, :]) / (2 * self.dy)
        
        # Compute residual
        mass_residual = dh_dt + dqx_dx + dqy_dy
        
        im3 = axs[1, 0].imshow(dh_dt, cmap='RdBu_r')
        axs[1, 0].set_title('dh/dt')
        plt.colorbar(im3, ax=axs[1, 0])
        
        im4 = axs[1, 1].imshow(dqx_dx, cmap='RdBu_r')
        axs[1, 1].set_title('d(hu)/dx')
        plt.colorbar(im4, ax=axs[1, 1])
        
        im5 = axs[1, 2].imshow(dqy_dy, cmap='RdBu_r')
        axs[1, 2].set_title('d(hv)/dy')
        plt.colorbar(im5, ax=axs[1, 2])
        
        # Plot residuals
        im6 = axs[2, 0].imshow(mass_residual, cmap='RdBu_r')
        axs[2, 0].set_title('Mass Conservation Residual')
        plt.colorbar(im6, ax=axs[2, 0])
        
        # Plot momentum residuals if available
        if elevation is not None:
            # Extract elevation for plotting
            elev = elevation[0, 0].cpu().numpy()
            
            # Show elevation
            im7 = axs[2, 1].imshow(elev, cmap='terrain')
            axs[2, 1].set_title('Bed Elevation')
            plt.colorbar(im7, ax=axs[2, 1])
            
            # Show physics loss weights
            loss_data = []
            labels = []
            
            for key, value in loss_dict.items():
                if key == "physics_components":
                    for comp_key, comp_value in loss_dict["physics_components"].items():
                        loss_data.append(comp_value.item())
                        labels.append(f"physics_{comp_key}")
                elif isinstance(value, torch.Tensor):
                    loss_data.append(value.item())
                    labels.append(key)
            
            axs[2, 2].bar(range(len(loss_data)), loss_data)
            axs[2, 2].set_xticks(range(len(loss_data)))
            axs[2, 2].set_xticklabels(labels, rotation=45, ha="right")
            axs[2, 2].set_title('Loss Components')
            axs[2, 2].set_yscale('log')
        else:
            axs[2, 1].axis('off')
            axs[2, 2].axis('off')
        
        # Add overall title with loss information
        if isinstance(loss_dict, dict) and "total" in loss_dict:
            title = f"Physics-Informed Loss Analysis: Total={loss_dict['total']:.6f}, "
            title += f"Data={loss_dict['data']:.6f}, Physics={loss_dict['physics']:.6f}"
        else:
            title = "Physics-Informed Loss Analysis"
            
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved physics loss visualization to {save_path}")
        
        # Show if requested
        if not show:
            plt.close(fig)
            
        return fig


class BoundaryConditionHandler(nn.Module):
    """
    Module to handle boundary conditions for the shallow water equations.
    """
    def __init__(self, grid_size, input_channels=3):
        super(BoundaryConditionHandler, self).__init__()
        self.grid_size = grid_size
        self.input_channels = input_channels
    
    def forward(self, x, boundaries):
        """
        Apply boundary conditions to the prediction.
        
        Args:
            x: Prediction tensor [batch, channels, height, width]
            boundaries: Dictionary of boundary conditions:
                - 'north', 'south', 'east', 'west': Dirichlet boundary values
                - 'type': Type of boundary condition ('dirichlet', 'neumann', 'periodic')
                
        Returns:
            Tensor with boundary conditions applied
        """
        if boundaries is None:
            return x
        
        batch_size, channels, height, width = x.shape
        boundary_type = boundaries.get('type', 'dirichlet')
        
        # Apply different boundary conditions based on type
        if boundary_type == 'dirichlet':
            # Apply Dirichlet boundary conditions (fixed values)
            for direction, values in boundaries.items():
                if direction == 'type':
                    continue
                    
                if direction == 'north' and values is not None:
                    x[:, :, 0, :] = values
                elif direction == 'south' and values is not None:
                    x[:, :, -1, :] = values
                elif direction == 'east' and values is not None:
                    x[:, :, :, -1] = values
                elif direction == 'west' and values is not None:
                    x[:, :, :, 0] = values
        
        elif boundary_type == 'neumann':
            # Apply Neumann boundary conditions (fixed derivatives)
            for direction, values in boundaries.items():
                if direction == 'type':
                    continue
                    
                if direction == 'north' and values is not None:
                    x[:, :, 0, :] = x[:, :, 1, :] + values
                elif direction == 'south' and values is not None:
                    x[:, :, -1, :] = x[:, :, -2, :] + values
                elif direction == 'east' and values is not None:
                    x[:, :, :, -1] = x[:, :, :, -2] + values
                elif direction == 'west' and values is not None:
                    x[:, :, :, 0] = x[:, :, :, 1] + values
        
        elif boundary_type == 'periodic':
            # Apply periodic boundary conditions
            # For north-south periodicity
            if boundaries.get('periodic_y', False):
                x[:, :, 0, :] = x[:, :, -2, :]
                x[:, :, -1, :] = x[:, :, 1, :]
            
            # For east-west periodicity
            if boundaries.get('periodic_x', False):
                x[:, :, :, 0] = x[:, :, :, -2]
                x[:, :, :, -1] = x[:, :, :, 1]
        
        return x


def train_multi_scale_pinn(
    model,
    train_loader,
    val_loader=None,
    optimizer=None,
    scheduler=None,
    num_epochs=100,
    save_dir="models/saved",
    save_interval=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the multi-scale PINN model.
    
    Args:
        model: MultiScalePINN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        optimizer: Optimizer (default: Adam)
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train for
        save_dir: Directory to save checkpoints
        save_interval: Interval (in epochs) to save checkpoints
        device: Device to train on
        
    Returns:
        Dictionary with training history
    """
    # Move model to device
    model = model.to(device)
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize training history
    history = {
        "train_total_loss": [],
        "train_data_loss": [],
        "train_physics_loss": [],
        "val_total_loss": [],
        "val_data_loss": [],
        "val_physics_loss": []
    }
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_total_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            results = model.training_step(batch)
            
            # Backward pass
            results["loss"].backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            train_total_loss += results["loss"].item()
            train_data_loss += results["data_loss"].item()
            train_physics_loss += results["physics_loss"].item()
            
        # Average losses
        train_total_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)
        
        # Update history
        history["train_total_loss"].append(train_total_loss)
        history["train_data_loss"].append(train_data_loss)
        history["train_physics_loss"].append(train_physics_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_data_loss = 0.0
            val_physics_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Move batch to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    
                    # Forward pass
                    results = model.validation_step(batch)
                    
                    # Accumulate losses
                    val_total_loss += results["loss"].item()
                    val_data_loss += results["data_loss"].item()
                    val_physics_loss += results["physics_loss"].item()
                
                # Average losses
                val_total_loss /= len(val_loader)
                val_data_loss /= len(val_loader)
                val_physics_loss /= len(val_loader)
                
                # Update history
                history["val_total_loss"].append(val_total_loss)
                history["val_data_loss"].append(val_data_loss)
                history["val_physics_loss"].append(val_physics_loss)
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_total_loss:.6f} "
                  f"(Data: {train_data_loss:.6f}, Physics: {train_physics_loss:.6f}) - "
                  f"Val Loss: {val_total_loss:.6f} "
                  f"(Data: {val_data_loss:.6f}, Physics: {val_physics_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_total_loss:.6f} "
                  f"(Data: {train_data_loss:.6f}, Physics: {train_physics_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            model.save_model(save_dir, filename=f"model_epoch_{epoch+1}.pt")
    
    # Save final model
    model.save_model(save_dir, filename="model_final.pt")
    
    return history


def plot_training_history(history, save_path=None, show=True):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot total loss
    ax1.plot(history["train_total_loss"], label="Training")
    if "val_total_loss" in history and len(history["val_total_loss"]) > 0:
        ax1.plot(history["val_total_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss vs. Epoch")
    ax1.legend()
    
    # Plot component losses
    ax2.plot(history["train_data_loss"], label="Training Data Loss")
    ax2.plot(history["train_physics_loss"], label="Training Physics Loss")
    if "val_data_loss" in history and len(history["val_data_loss"]) > 0:
        ax2.plot(history["val_data_loss"], label="Validation Data Loss")
        ax2.plot(history["val_physics_loss"], label="Validation Physics Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Component Losses vs. Epoch")
    ax2.legend()
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


class PhysicsLossModule(nn.Module):
    """
    Physics-Informed Loss Module for Shallow Water Equations.
    
    This module computes the physics-informed loss components for the shallow water equations
    and provides an adaptive weighting mechanism between data and physics losses.
    """
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 0.1,
        gravity: float = 9.81,
        manning_coef: float = 0.035,
        initial_physics_weight: float = 0.1,
        adaptive_weighting: bool = True,
        min_physics_weight: float = 0.01,
        max_physics_weight: float = 1.0,
        adaptation_rate: float = 0.05,
        continuity_weight: float = 1.0,
        x_momentum_weight: float = 0.5,
        y_momentum_weight: float = 0.5,
        boundary_weight: float = 0.5,
        enforce_positivity: bool = True
    ):
        super(PhysicsLossModule, self).__init__()
        
        # Physical parameters
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.g = gravity
        self.n = manning_coef
        
        # Loss weighting parameters
        self.physics_weight = initial_physics_weight
        self.adaptive_weighting = adaptive_weighting
        self.min_physics_weight = min_physics_weight
        self.max_physics_weight = max_physics_weight
        self.adaptation_rate = adaptation_rate
        
        # Equation-specific weights
        self.continuity_weight = continuity_weight
        self.x_momentum_weight = x_momentum_weight
        self.y_momentum_weight = y_momentum_weight
        self.boundary_weight = boundary_weight
        self.enforce_positivity = enforce_positivity
        
        # Register parameters as buffers (non-trainable)
        self.register_buffer('g_tensor', torch.tensor(gravity, dtype=torch.float))
        self.register_buffer('n_tensor', torch.tensor(manning_coef, dtype=torch.float))
    
    def compute_continuity_residual(self, h_pred, h_prev, qx_pred, qy_pred):
        """
        Compute the continuity equation residual.
        
        Continuity equation: dh/dt + d(h*u)/dx + d(h*v)/dy = 0
        
        Args:
            h_pred: Predicted water depth [batch, 1, height, width]
            h_prev: Previous water depth [batch, 1, height, width]
            qx_pred: Predicted x-momentum [batch, 1, height, width]
            qy_pred: Predicted y-momentum [batch, 1, height, width]
            
        Returns:
            Continuity residual [batch, 1, height, width]
        """
        # Calculate time derivative: dh/dt
        dh_dt = (h_pred - h_prev) / self.dt
        
        # Spatial derivatives using finite differences
        dqx_dx = torch.zeros_like(h_pred)
        dqy_dy = torch.zeros_like(h_pred)
        
        # x-direction derivative: d(h*u)/dx
        dqx_dx[:, :, :, 1:-1] = (qx_pred[:, :, :, 2:] - qx_pred[:, :, :, :-2]) / (2 * self.dx)
        
        # y-direction derivative: d(h*v)/dy
        dqy_dy[:, :, 1:-1, :] = (qy_pred[:, :, 2:, :] - qy_pred[:, :, :-2, :]) / (2 * self.dy)
        
        # Continuity equation residual
        return dh_dt + dqx_dx + dqy_dy
    
    def compute_momentum_residuals(self, h_pred, h_prev, qx_pred, qx_prev, qy_pred, qy_prev, elevation=None):
        """
        Compute the x and y momentum equation residuals.
        
        x-momentum: d(h*u)/dt + d(h*u²)/dx + d(h*u*v)/dy + g*h*dz/dx + g*n²*u*√(u²+v²)/h^(1/3) = 0
        y-momentum: d(h*v)/dt + d(h*u*v)/dx + d(h*v²)/dy + g*h*dz/dy + g*n²*v*√(u²+v²)/h^(1/3) = 0
        
        Args:
            h_pred: Predicted water depth [batch, 1, height, width]
            h_prev: Previous water depth [batch, 1, height, width]
            qx_pred: Predicted x-momentum [batch, 1, height, width]
            qx_prev: Previous x-momentum [batch, 1, height, width]
            qy_pred: Predicted y-momentum [batch, 1, height, width]
            qy_prev: Previous y-momentum [batch, 1, height, width]
            elevation: Bed elevation [batch, 1, height, width] or None
            
        Returns:
            Tuple of (x_momentum_residual, y_momentum_residual)
        """
        # Compute velocities with epsilon for stability
        eps = 1e-6
        u_pred = qx_pred / (h_pred + eps)
        v_pred = qy_pred / (h_pred + eps)
        
        # Time derivatives
        dqx_dt = (qx_pred - qx_prev) / self.dt
        dqy_dt = (qy_pred - qy_prev) / self.dt
        
        # Calculate fluxes
        flux_xx = h_pred * u_pred * u_pred  # h*u²
        flux_xy = h_pred * u_pred * v_pred  # h*u*v
        flux_yy = h_pred * v_pred * v_pred  # h*v²
        
        # Initialize derivatives
        dflux_xx_dx = torch.zeros_like(h_pred)
        dflux_xy_dy = torch.zeros_like(h_pred)
        dflux_xy_dx = torch.zeros_like(h_pred)
        dflux_yy_dy = torch.zeros_like(h_pred)
        
        # Compute flux derivatives
        dflux_xx_dx[:, :, :, 1:-1] = (flux_xx[:, :, :, 2:] - flux_xx[:, :, :, :-2]) / (2 * self.dx)
        dflux_xy_dx[:, :, :, 1:-1] = (flux_xy[:, :, :, 2:] - flux_xy[:, :, :, :-2]) / (2 * self.dx)
        dflux_xy_dy[:, :, 1:-1, :] = (flux_xy[:, :, 2:, :] - flux_xy[:, :, :-2, :]) / (2 * self.dy)
        dflux_yy_dy[:, :, 1:-1, :] = (flux_yy[:, :, 2:, :] - flux_yy[:, :, :-2, :]) / (2 * self.dy)
        
        # Initialize pressure and friction terms
        pressure_x = torch.zeros_like(h_pred)
        pressure_y = torch.zeros_like(h_pred)
        friction_x = torch.zeros_like(h_pred)
        friction_y = torch.zeros_like(h_pred)
        
        # Compute pressure and friction terms if elevation data is available
        if elevation is not None:
            # Compute bed slopes
            dz_dx = torch.zeros_like(h_pred)
            dz_dy = torch.zeros_like(h_pred)
            
            dz_dx[:, :, :, 1:-1] = (elevation[:, :, :, 2:] - elevation[:, :, :, :-2]) / (2 * self.dx)
            dz_dy[:, :, 1:-1, :] = (elevation[:, :, 2:, :] - elevation[:, :, :-2, :]) / (2 * self.dy)
            
            # Pressure terms (bed slope)
            pressure_x = self.g_tensor * h_pred * dz_dx
            pressure_y = self.g_tensor * h_pred * dz_dy
            
            # Friction terms (Manning's equation)
            vel_magnitude = torch.sqrt(u_pred**2 + v_pred**2)
            h_powered = h_pred**(4/3)
            h_powered = torch.where(h_powered > eps, h_powered, torch.ones_like(h_powered) * eps)
            
            friction_x = self.g_tensor * self.n_tensor**2 * u_pred * vel_magnitude / h_powered
            friction_y = self.g_tensor * self.n_tensor**2 * v_pred * vel_magnitude / h_powered
        
        # Momentum residuals
        x_momentum_residual = dqx_dt + dflux_xx_dx + dflux_xy_dy + pressure_x + friction_x
        y_momentum_residual = dqy_dt + dflux_xy_dx + dflux_yy_dy + pressure_y + friction_y
        
        return x_momentum_residual, y_momentum_residual
    
    def forward(self, predictions, inputs, targets, elevation=None):
        """
        Compute the physics-informed loss.
        
        Args:
            predictions: Model predictions [batch, channels, height, width]
            inputs: Input conditions [batch, channels, height, width]
            targets: Target values [batch, channels, height, width]
            elevation: Elevation data [batch, 1, height, width] or None
            
        Returns:
            Dictionary containing loss components and weights
        """
        # Extract variables from predictions and inputs
        h_pred = predictions[:, 0:1, :, :]  # Water depth
        qx_pred = predictions[:, 1:2, :, :]  # x-momentum
        qy_pred = predictions[:, 2:3, :, :]  # y-momentum
        
        h_prev = inputs[:, 0:1, :, :]  # Previous water depth
        qx_prev = inputs[:, 1:2, :, :]  # Previous x-momentum
        qy_prev = inputs[:, 2:3, :, :]  # Previous y-momentum
        
        # Compute data loss (MSE between predictions and targets)
        data_loss = F.mse_loss(predictions, targets)
        
        # Initialize loss components dictionary
        losses = {
            "data": data_loss,
            "physics_components": {}
        }
        
        # Compute continuity residual
        continuity_residual = self.compute_continuity_residual(h_pred, h_prev, qx_pred, qy_pred)
        
        # Apply continuity weight (for mass conservation)
        continuity_loss = torch.mean(continuity_residual**2) * self.continuity_weight
        losses["physics_components"]["continuity"] = continuity_loss
        
        # Initialize the physics loss with continuity loss
        physics_loss = continuity_loss
        
        # Compute momentum residuals if elevation is available
        if elevation is not None:
            x_momentum_residual, y_momentum_residual = self.compute_momentum_residuals(
                h_pred, h_prev, qx_pred, qx_prev, qy_pred, qy_prev, elevation
            )
            
            # Apply momentum weights
            x_momentum_loss = torch.mean(x_momentum_residual**2) * self.x_momentum_weight
            y_momentum_loss = torch.mean(y_momentum_residual**2) * self.y_momentum_weight
            
            losses["physics_components"]["x_momentum"] = x_momentum_loss
            losses["physics_components"]["y_momentum"] = y_momentum_loss
            
            # Combined momentum loss
            momentum_loss = x_momentum_loss + y_momentum_loss
            
            # Total physics loss (adding momentum to continuity)
            physics_loss = continuity_loss + momentum_loss
        
        losses["physics"] = physics_loss
        
        # Apply adaptive weighting
        if self.adaptive_weighting and self.training:
            # Calculate ratio of losses
            loss_ratio = data_loss / (physics_loss + 1e-8)
            
            # Adjust weights based on loss ratio
            if loss_ratio > 10.0:
                # Data loss is much larger, increase physics weight
                new_weight = min(
                    self.physics_weight * (1 + self.adaptation_rate),
                    self.max_physics_weight
                )
            elif loss_ratio < 0.1:
                # Physics loss is much larger, decrease physics weight
                new_weight = max(
                    self.physics_weight * (1 - self.adaptation_rate),
                    self.min_physics_weight
                )
            else:
                # Loss ratio is balanced, maintain current weight
                new_weight = self.physics_weight
            
            # Update physics weight
            self.physics_weight = new_weight
        
        # Apply weighting to physics loss
        weighted_physics_loss = self.physics_weight * physics_loss
        
        # Calculate total loss
        total_loss = data_loss + weighted_physics_loss
        
        # Store weights and weighted losses
        losses["physics_weight"] = self.physics_weight
        losses["weighted_physics"] = weighted_physics_loss
        losses["total"] = total_loss
        
        return losses


def validation_step(self, batch):
        """
        Perform a validation step.
        
        Args:
            batch: Dictionary containing:
                - inputs: Tensor of shape [batch, channels, height, width]
                - targets: Tensor of shape [batch, channels, height, width]
                - elevation: Optional tensor of shape [batch, 1, height, width]
                
        Returns:
            Dictionary containing validation losses
        """
        # Extract data from batch
        inputs = batch["inputs"]
        targets = batch["targets"]
        elevation = batch.get("elevation", None)
        
        # Forward pass (no gradient computation needed)
        with torch.no_grad():
            predictions = self.forward(inputs)
            
            # Compute detailed losses
            if hasattr(self, 'physics_loss'):
                # Use the specialized physics loss module if available
                loss_dict = self.physics_loss(predictions, inputs, targets, elevation)
                
                # Extract components
                data_loss = loss_dict["data"]
                physics_loss = loss_dict["physics"]
                total_loss = loss_dict["total"]
                
                # Create result dictionary
                result = {
                    "loss": total_loss,
                    "data_loss": data_loss,
                    "physics_loss": physics_loss,
                    "physics_weight": loss_dict["physics_weight"]
                }
                
                # Add physics components to result
                for key, value in loss_dict["physics_components"].items():
                    result[f"physics_{key}"] = value
            else:
                # Use the regular compute_physics_loss method
                data_loss = F.mse_loss(predictions, targets)
                physics_losses = self.compute_physics_loss(predictions, inputs, elevation)
                physics_loss = physics_losses["total_physics"]
                total_loss = data_loss + physics_loss
                
                # Create result dictionary
                result = {
                    "loss": total_loss,
                    "data_loss": data_loss,
                    "physics_loss": physics_loss,
                    "physics_weight": self.physics_weight
                }
                
                # Add detailed physics components
                for key, value in physics_losses.items():
                    if key != "total_physics":
                        result[f"physics_{key}"] = value
        
        return result 