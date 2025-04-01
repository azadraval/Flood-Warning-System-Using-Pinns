import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

from ..config.config import PINNConfig, SystemConfig

class FourierFeatureMapping(nn.Module):
    """
    Fourier feature mapping layer for enhancing the representation capacity of the network.
    
    References:
    - "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
      (Tancik et al., 2020)
    """
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        
        # Initialize random Fourier features
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input onto random basis
        x_proj = torch.matmul(x, self.B)
        
        # Apply periodic activation
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GeometryEncoder(nn.Module):
    """
    Encodes complex geometries for the PINN to better handle irregular domains.
    
    Implements the geometry-adaptive approach from GeoPINS methodology.
    """
    def __init__(self, method: str = "distance_field", resolution: int = 64):
        super().__init__()
        self.method = method
        self.resolution = resolution
        
        # Parameters for learned geometry mapping (if applicable)
        if method == "learnable":
            self.mapping_network = nn.Sequential(
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 2)
            )
    
    def compute_distance_field(self, boundary_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """Compute the distance field from query points to the closest boundary points."""
        # Efficient vectorized distance computation
        diffs = query_points.unsqueeze(1) - boundary_points.unsqueeze(0)
        dist = torch.norm(diffs, dim=2)
        return torch.min(dist, dim=1)[0].unsqueeze(-1)
    
    def forward(self, x: torch.Tensor, boundary_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.method == "identity":
            return x
        
        elif self.method == "distance_field":
            if boundary_points is None:
                raise ValueError("Boundary points must be provided for distance field encoding")
            distance = self.compute_distance_field(boundary_points, x)
            return torch.cat([x, distance], dim=-1)
        
        elif self.method == "learnable":
            return self.mapping_network(x)
        
        else:
            raise ValueError(f"Unknown geometry encoding method: {self.method}")

class PhysicsInformedNeuralNetwork(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for flood modeling.
    
    This model solves the shallow water equations using neural networks
    and physics-based constraints.
    """
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        
        # Parameters for the shallow water equations
        self.g = nn.Parameter(torch.tensor(config.gravity), requires_grad=False)
        self.n = nn.Parameter(torch.tensor(config.manning_coefficient), requires_grad=False)
        self.theta = nn.Parameter(torch.tensor(config.theta), requires_grad=False)
        
        # Input dimensions: (x, y, t) = 3
        input_dim = 3
        
        # Fourier feature layer if enabled
        if config.use_fourier_features:
            self.fourier_layer = FourierFeatureMapping(
                input_dim=input_dim,
                mapping_size=config.num_fourier_features,
                scale=config.fourier_sigma
            )
            # New input dim: original dim + 2 * mapping_size (sin and cos components)
            mapped_dim = 2 * config.num_fourier_features
        else:
            self.fourier_layer = None
            mapped_dim = input_dim
        
        # Geometry adaptation if enabled
        if config.use_geometry_adaptation:
            self.geometry_encoder = GeometryEncoder(
                method=config.boundary_encoding_method
            )
            # Add 1 for distance field encoding
            if config.boundary_encoding_method == "distance_field":
                mapped_dim += 1
        else:
            self.geometry_encoder = None
        
        # Construct the fully connected layers
        layer_dims = [mapped_dim] + (config.hidden_layers or [128, 128, 128]) + [3]  # Output: (h, u, v)
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
        # Initialize weights using specified method
        self._initialize_weights()
            
    def _initialize_weights(self):
        """Initialize the weights of the network based on the configuration."""
        for layer in self.layers:
            if self.config.initialization == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif self.config.initialization == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.config.initialization == "kaiming_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh')
            elif self.config.initialization == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            else:
                # Default initialization
                pass
            
            # Initialize bias to zeros
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor, boundary_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the PINN.
        
        Args:
            x: Input tensor with shape (batch_size, 3), where the dimensions are (x, y, t)
            boundary_points: Optional tensor of boundary points for geometry encoding
            
        Returns:
            Output tensor with shape (batch_size, 3), representing (h, u, v)
            h: water depth
            u: x-velocity
            v: y-velocity
        """
        # Apply Fourier feature mapping if enabled
        if self.fourier_layer is not None:
            x = self.fourier_layer(x)
        
        # Apply geometry encoding if enabled
        if self.geometry_encoder is not None:
            x = self.geometry_encoder(x, boundary_points)
        
        # Forward pass through the network
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            # Apply activation based on configuration
            if self.config.activation == "tanh":
                x = torch.tanh(x)
            elif self.config.activation == "relu":
                x = F.relu(x)
            elif self.config.activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif self.config.activation == "gelu":
                x = F.gelu(x)
            elif self.config.activation == "swish":
                x = x * torch.sigmoid(x)
            else:
                # Default to tanh if not specified
                x = torch.tanh(x)
                
            # Apply dropout if specified
            if self.config.dropout_rate > 0.0:
                x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        
        # Final layer without activation
        x = self.layers[-1](x)
        
        # Post-process outputs to ensure physical constraints
        h, u, v = torch.split(x, 1, dim=-1)
        
        # Water depth must be non-negative
        h = F.softplus(h)
        
        return torch.cat([h, u, v], dim=-1)
    
    def compute_pde_residuals(self, x: torch.Tensor, outputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the residuals of the shallow water equations.
        
        Args:
            x: Input tensor with shape (batch_size, 3), representing (x, y, t)
            outputs: Optional pre-computed outputs from the model
            
        Returns:
            Dictionary containing the residuals for continuity and momentum equations
        """
        # Ensure inputs require gradient for automatic differentiation
        x.requires_grad_(True)
        
        # Get the predictions if not provided
        if outputs is None:
            outputs = self.forward(x)
        
        h, u, v = torch.split(outputs, 1, dim=-1)
        
        # Extract the coordinates
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        t = x[:, 2:3]
        
        # Compute gradients for PDE residuals
        h_t = torch.autograd.grad(
            h, t, grad_outputs=torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]
        
        h_x = torch.autograd.grad(
            h, x_coord, grad_outputs=torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]
        
        h_y = torch.autograd.grad(
            h, y_coord, grad_outputs=torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]
        
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x_coord, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u, y_coord, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        v_t = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        
        v_x = torch.autograd.grad(
            v, x_coord, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        
        v_y = torch.autograd.grad(
            v, y_coord, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        
        # Continuity equation: dh/dt + d(uh)/dx + d(vh)/dy = 0
        # We expand the derivatives:
        # dh/dt + u*dh/dx + h*du/dx + v*dh/dy + h*dv/dy = 0
        continuity_residual = h_t + u * h_x + h * u_x + v * h_y + h * v_y
        
        # Momentum equation in x: du/dt + u*du/dx + v*du/dy + g*dh/dx + g*n^2*u*sqrt(u^2+v^2)/h^(4/3) = 0
        friction_term_x = self.g * self.n**2 * u * torch.sqrt(u**2 + v**2) / (h**(4/3) + 1e-6)
        momentum_x_residual = u_t + u * u_x + v * u_y + self.g * h_x + friction_term_x
        
        # Momentum equation in y: dv/dt + u*dv/dx + v*dv/dy + g*dh/dy + g*n^2*v*sqrt(u^2+v^2)/h^(4/3) = 0
        friction_term_y = self.g * self.n**2 * v * torch.sqrt(u**2 + v**2) / (h**(4/3) + 1e-6)
        momentum_y_residual = v_t + u * v_x + v * v_y + self.g * h_y + friction_term_y
        
        return {
            "continuity": continuity_residual,
            "momentum_x": momentum_x_residual,
            "momentum_y": momentum_y_residual
        }
    
    def compute_total_pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the total PDE loss as a weighted sum of all residuals."""
        residuals = self.compute_pde_residuals(x)
        
        # Squared L2 norm of each residual
        continuity_loss = torch.mean(residuals["continuity"]**2)
        momentum_x_loss = torch.mean(residuals["momentum_x"]**2)
        momentum_y_loss = torch.mean(residuals["momentum_y"]**2)
        
        # Total loss with weights
        total_pde_loss = (
            continuity_loss + 
            momentum_x_loss + 
            momentum_y_loss
        )
        
        return total_pde_loss
    
    def compute_boundary_condition_loss(
        self, 
        bc_points: torch.Tensor, 
        bc_values: torch.Tensor, 
        bc_type: str = "dirichlet"
    ) -> torch.Tensor:
        """
        Compute the loss for boundary conditions.
        
        Args:
            bc_points: Points on the boundary (x, y, t)
            bc_values: Target values at boundary points (h, u, v)
            bc_type: Type of boundary condition ("dirichlet", "neumann", or "periodic")
            
        Returns:
            Loss for the boundary conditions
        """
        # Predict at boundary points
        predictions = self.forward(bc_points)
        
        if bc_type == "dirichlet":
            # Direct value matching
            bc_loss = F.mse_loss(predictions, bc_values)
            
        elif bc_type == "neumann":
            # Gradient matching
            # We need to compute gradients with respect to spatial coordinates
            bc_points.requires_grad_(True)
            predictions = self.forward(bc_points)
            
            # Extract components
            h_pred, u_pred, v_pred = torch.split(predictions, 1, dim=-1)
            h_target, u_target, v_target = torch.split(bc_values, 1, dim=-1)
            
            # Compute gradients
            h_x = torch.autograd.grad(
                h_pred, bc_points, grad_outputs=torch.ones_like(h_pred),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]
            
            # Compare with target gradients
            bc_loss = F.mse_loss(h_x, h_target)
            
        elif bc_type == "periodic":
            # For periodic BCs, bc_points should contain pairs of points
            # that should have the same values
            n_pairs = bc_points.shape[0] // 2
            points_a = bc_points[:n_pairs]
            points_b = bc_points[n_pairs:]
            
            pred_a = self.forward(points_a)
            pred_b = self.forward(points_b)
            
            bc_loss = F.mse_loss(pred_a, pred_b)
            
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
            
        return bc_loss
    
    def compute_initial_condition_loss(
        self, 
        ic_points: torch.Tensor, 
        ic_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for initial conditions.
        
        Args:
            ic_points: Points at initial time (x, y, t=0)
            ic_values: Target values at initial points (h, u, v)
            
        Returns:
            Loss for the initial conditions
        """
        # Predict at initial condition points
        predictions = self.forward(ic_points)
        
        # MSE loss for initial conditions
        ic_loss = F.mse_loss(predictions, ic_values)
        
        return ic_loss
    
    def compute_data_loss(
        self, 
        data_points: torch.Tensor, 
        data_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss against observed data.
        
        Args:
            data_points: Observation points (x, y, t)
            data_values: Observed values at data points (h, u, v)
            
        Returns:
            Loss for the data fitting
        """
        # Predict at data points
        predictions = self.forward(data_points)
        
        # MSE loss for data
        data_loss = F.mse_loss(predictions, data_values)
        
        return data_loss
    
    def compute_total_loss(
        self,
        pde_points: torch.Tensor,
        bc_points: Optional[torch.Tensor] = None,
        bc_values: Optional[torch.Tensor] = None,
        bc_type: str = "dirichlet",
        ic_points: Optional[torch.Tensor] = None,
        ic_values: Optional[torch.Tensor] = None,
        data_points: Optional[torch.Tensor] = None,
        data_values: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including PDE residuals, boundary conditions,
        initial conditions, and data fitting.
        
        Returns:
            Dictionary containing individual losses and the total loss
        """
        # Initialize loss components
        pde_loss = self.compute_total_pde_loss(pde_points)
        losses = {"pde_loss": pde_loss}
        
        # Boundary condition loss
        bc_loss = torch.tensor(0.0, device=pde_points.device)
        if bc_points is not None and bc_values is not None:
            bc_loss = self.compute_boundary_condition_loss(bc_points, bc_values, bc_type)
            losses["bc_loss"] = bc_loss
        
        # Initial condition loss
        ic_loss = torch.tensor(0.0, device=pde_points.device)
        if ic_points is not None and ic_values is not None:
            ic_loss = self.compute_initial_condition_loss(ic_points, ic_values)
            losses["ic_loss"] = ic_loss
        
        # Data loss
        data_loss = torch.tensor(0.0, device=pde_points.device)
        if data_points is not None and data_values is not None:
            data_loss = self.compute_data_loss(data_points, data_values)
            losses["data_loss"] = data_loss
        
        # Total loss with weights
        total_loss = (
            self.config.pde_weight * pde_loss +
            self.config.bc_weight * bc_loss +
            self.config.ic_weight * ic_loss +
            self.config.data_weight * data_loss
        )
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def predict_flood_probability(
        self, 
        x_tensor: torch.Tensor, 
        threshold: float = 0.1,  # Water depth threshold (in meters) for flood classification
        ensemble_size: int = 1,
        dropout_enabled: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the probability of flooding at given points.
        
        Args:
            x_tensor: Input tensor with shape (batch_size, 3) for (x, y, t)
            threshold: Water depth threshold for flood classification (in meters)
            ensemble_size: Number of stochastic forward passes for uncertainty estimation
            dropout_enabled: Whether to enable dropout during inference
            
        Returns:
            Tuple of (flood_probability, uncertainty)
        """
        if ensemble_size == 1:
            # Single prediction without uncertainty
            with torch.no_grad():
                predictions = self.forward(x_tensor)
                
            h = predictions[:, 0]  # Water depth
            flood_prob = torch.sigmoid(10.0 * (h - threshold))  # Sigmoid-based probability
            uncertainty = torch.zeros_like(flood_prob)
            
        else:
            # Multiple stochastic forward passes for uncertainty estimation
            previous_training = self.training
            if dropout_enabled:
                self.train()  # Enable dropout
            
            flood_preds = []
            with torch.no_grad():
                for _ in range(ensemble_size):
                    predictions = self.forward(x_tensor)
                    h = predictions[:, 0]  # Water depth
                    flood_pred = torch.sigmoid(10.0 * (h - threshold))
                    flood_preds.append(flood_pred)
            
            # Reset training mode
            self.train(previous_training)
            
            # Stack predictions
            flood_preds = torch.stack(flood_preds, dim=0)
            
            # Compute mean and standard deviation
            flood_prob = torch.mean(flood_preds, dim=0)
            uncertainty = torch.std(flood_preds, dim=0)
        
        return flood_prob, uncertainty

class SequenceToSequencePINN(nn.Module):
    """
    Sequence-to-sequence model using PINN for long-term flood forecasting.
    
    This model builds on the PINN model to handle temporal sequences and predict
    future states in an autoregressive manner.
    """
    def __init__(self, 
                 config: PINNConfig, 
                 sequence_length: int = 24, 
                 prediction_horizon: int = 72):
        super().__init__()
        
        self.config = config
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Base PINN model
        self.pinn = PhysicsInformedNeuralNetwork(config)
        
        # Encoder for processing input sequences
        self.encoder = nn.GRU(
            input_size=3,  # (h, u, v)
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1 if config.dropout_rate > 0 else 0
        )
        
        # Decoder for predicting output sequences
        self.decoder = nn.GRU(
            input_size=3,  # (h, u, v)
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1 if config.dropout_rate > 0 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(128, 3)  # Project to (h, u, v)
        
    def encode_sequence(self, 
                       input_seq: torch.Tensor, 
                       spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence.
        
        Args:
            input_seq: Sequence of (h, u, v) values with shape (batch_size, sequence_length, 3)
            spatial_coords: Spatial coordinates (x, y) with shape (batch_size, 2)
            
        Returns:
            Encoder hidden state
        """
        # Run encoder
        _, hidden = self.encoder(input_seq)
        return hidden
        
    def forward(self, 
                input_seq: torch.Tensor, 
                spatial_coords: torch.Tensor,
                target_times: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass of the sequence-to-sequence model.
        
        Args:
            input_seq: Input sequence with shape (batch_size, sequence_length, 3) for (h, u, v)
            spatial_coords: Spatial coordinates with shape (batch_size, 2) for (x, y)
            target_times: Target time steps with shape (batch_size, prediction_horizon) for t
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Predicted sequence with shape (batch_size, prediction_horizon, 3) for (h, u, v)
        """
        batch_size = input_seq.shape[0]
        device = input_seq.device
        
        # Encode input sequence
        hidden = self.encode_sequence(input_seq, spatial_coords)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.prediction_horizon, 3, device=device)
        
        # Initialize the first input to the decoder
        decoder_input = input_seq[:, -1, :].unsqueeze(1)  # Last step of input
        
        # Decode step by step
        for t in range(self.prediction_horizon):
            # Current time
            current_time = target_times[:, t].unsqueeze(1)
            
            # Current model input for PINN
            x_t = torch.cat([spatial_coords, current_time], dim=1)
            
            # Run decoder for one step
            output_t, hidden = self.decoder(decoder_input, hidden)
            output_t = self.output_proj(output_t)
            
            # Run PINN to refine the prediction with physics constraints
            pinn_input = x_t
            pinn_output = self.pinn(pinn_input)
            
            # Combine decoder and PINN outputs (weighted average)
            alpha = 0.7  # Weight for PINN output
            combined_output = alpha * pinn_output + (1.0 - alpha) * output_t.squeeze(1)
            
            # Store output
            outputs[:, t, :] = combined_output
            
            # Next decoder input (use ground truth with teacher forcing probability)
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            if t < self.prediction_horizon - 1 and teacher_force and self.training:
                decoder_input = target_seq[:, t, :].unsqueeze(1)
            else:
                decoder_input = combined_output.unsqueeze(1)
        
        return outputs
    
    def predict(self, 
                input_seq: torch.Tensor, 
                spatial_coords: torch.Tensor,
                target_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction without teacher forcing.
        
        Args:
            input_seq: Input sequence with shape (batch_size, sequence_length, 3) for (h, u, v)
            spatial_coords: Spatial coordinates with shape (batch_size, 2) for (x, y)
            target_times: Target time steps with shape (batch_size, prediction_horizon) for t
            
        Returns:
            Tuple of (predictions, uncertainties) with shape (batch_size, prediction_horizon, 3)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(
                input_seq=input_seq,
                spatial_coords=spatial_coords,
                target_times=target_times,
                teacher_forcing_ratio=0.0  # No teacher forcing during inference
            )
            
            # Compute uncertainties (simplified)
            uncertainties = torch.zeros_like(predictions)
            # In a real system, you would use ensemble methods or dropout to estimate uncertainty
        
        return predictions, uncertainties 