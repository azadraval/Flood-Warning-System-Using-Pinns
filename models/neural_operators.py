"""
Neural operator implementations for flood modeling.

This module provides neural operator architectures for efficiently modeling complex
fluid dynamics, specifically designed for flood simulations.

Key components:
1. Fourier Neural Operator (FNO) - For modeling global dependencies in fluid flow
2. Graph Neural Network (GNN) - For handling irregular geometries and mesh structures
3. Multi-scale operator networks - For capturing physics at different spatial scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer for spectral convolutions.
    
    This implements the core of the Fourier Neural Operator (FNO) which
    performs convolutions in the frequency domain.
    
    References:
    - "Fourier Neural Operator for Parametric Partial Differential Equations"
      (Li et al., 2020)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1  # modes in the x-direction
        self.modes2 = modes2  # modes in the y-direction
        
        # Complex weights for each mode
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
        
    def forward(self, x):
        """
        Forward pass of the spectral convolution layer.
        
        Args:
            x: Input tensor (batch, channels, x, y)
            
        Returns:
            Output tensor after spectral convolution
        """
        batch_size = x.shape[0]
        
        # Get grid dimensions
        x_dim = x.shape[2]
        y_dim = x.shape[3]
        
        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output tensor in frequency domain
        out_ft = torch.zeros(batch_size, self.out_channels, x_dim, y_dim//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        # Process the modes we want to keep
        # Lower frequencies - always present
        modes1_to_use = min(self.modes1, x_dim)
        modes2_to_use = min(self.modes2, y_dim//2 + 1)
        
        out_ft[:, :, :modes1_to_use, :modes2_to_use] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes1_to_use, :modes2_to_use],
            torch.view_as_complex(self.weights1[:, :, :modes1_to_use, :modes2_to_use])
        )
        
        # Higher frequencies - only if grid is large enough
        if x_dim > self.modes1:
            higher_modes1 = min(self.modes1, x_dim - self.modes1)
            out_ft[:, :, -higher_modes1:, :modes2_to_use] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -higher_modes1:, :modes2_to_use],
                torch.view_as_complex(self.weights2[:, :, :higher_modes1, :modes2_to_use])
            )
        
        # Convert back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(x_dim, y_dim))
        return x

class FourierNeuralOperator2D(nn.Module):
    """
    2D Fourier Neural Operator for learning mappings between function spaces.
    
    This implements a series of Fourier layers with fully connected layers.
    """
    def __init__(
        self, 
        modes1: int, 
        modes2: int,
        width: int = 64, 
        in_channels: int = 3, 
        out_channels: int = 3,
        num_layers: int = 4,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        super(FourierNeuralOperator2D, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Projection layer
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for spatial coordinates
        
        # Spectral convolution layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.conv_layers.append(SpectralConv2d(width, width, modes1, modes2))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(width))
            self.dropout_layers.append(nn.Dropout(dropout))
            
        # Output layer
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x, grid):
        # x: (batch_size, channels, height, width)
        # grid: (batch_size, 2, height, width) spatial coordinates
        
        batch_size, _, height, width = x.shape
        
        # Concatenate inputs with grid for positional encoding
        grid = grid.expand(batch_size, 2, height, width)
        x = torch.cat([x, grid], dim=1)
        
        # Reshape for linear layer
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels+2)
        x = self.fc0(x)
        x = F.gelu(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, height, width)
        
        # Apply spectral convolution layers
        for i in range(self.num_layers - 1):
            x1 = self.conv_layers[i](x)
            x2 = x1
            
            if self.use_batch_norm:
                x2 = self.bn_layers[i](x2)
                
            x2 = F.gelu(x2)
            x2 = self.dropout_layers[i](x2)
            x = x + x2  # Residual connection
            
        # Output layers
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, height, width)
        
        return x

class MeshGraphConv(nn.Module):
    """
    Graph convolution for irregular meshes.
    
    This allows the model to handle irregular geometries like river networks
    and complex terrain.
    """
    def __init__(self, in_features, out_features):
        super(MeshGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass of graph convolution.
        
        Args:
            x: Node features (batch, nodes, features)
            adj: Adjacency matrix (batch, nodes, nodes) or (nodes, nodes)
            
        Returns:
            Updated node features
        """
        # Apply adjacency matrix (graph structure)
        support = torch.matmul(x, self.weight)
        
        # If the adjacency matrix doesn't have batch dimension, add it
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
            
        # Normalize adjacency matrix
        rowsum = adj.sum(dim=2, keepdim=True) + 1e-6
        norm_adj = adj / rowsum
        
        # Message passing
        output = torch.matmul(norm_adj, support)
        output = output + self.bias
        
        return output

class GraphNeuralOperator(nn.Module):
    """
    Graph Neural Operator for learning on mesh-based representations.
    
    This implements a series of graph convolution layers for processing
    irregular spatial domains.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int = 64, 
        out_features: int = 3,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        super(GraphNeuralOperator, self).__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.input_layer = nn.Linear(in_features, hidden_features)
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.graph_layers.append(MeshGraphConv(hidden_features, hidden_features))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_features))
            self.dropout_layers.append(nn.Dropout(dropout))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, adj):
        # x: (batch_size, nodes, features)
        # adj: (batch_size, nodes, nodes) or (nodes, nodes)
        
        # Input layer
        x = self.input_layer(x)
        x = F.gelu(x)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x_res = self.graph_layers[i](x, adj)
            
            if self.use_batch_norm:
                # Reshape for BatchNorm1d
                batch_size, nodes, features = x_res.shape
                x_res = x_res.reshape(-1, features)
                x_res = self.bn_layers[i](x_res)
                x_res = x_res.reshape(batch_size, nodes, features)
            
            x_res = F.gelu(x_res)
            x_res = self.dropout_layers[i](x_res)
            x = x + x_res  # Residual connection
            
        # Output layer
        x = self.output_layer(x)
        
        return x

class MultiScaleNeuralOperator(nn.Module):
    """
    Multi-scale neural operator that combines FNO and GNN components.
    
    This model operates at multiple spatial scales to capture both global and local
    fluid dynamics patterns in flood simulations.
    """
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        hidden_channels: int = 64,
        fno_modes: Tuple[int, int] = (12, 12),
        num_scales: int = 3,
        use_gnn: bool = True,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        super(MultiScaleNeuralOperator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.fno_modes = fno_modes
        self.num_scales = num_scales
        self.use_gnn = use_gnn
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Create FNO components at different scales
        self.fno_components = nn.ModuleList()
        
        for i in range(num_scales):
            # Scale parameters by scale factor
            scale_factor = 2 ** i
            modes_scale = (max(2, fno_modes[0] // scale_factor), 
                          max(2, fno_modes[1] // scale_factor))
            
            self.fno_components.append(
                FourierNeuralOperator2D(
                    modes1=modes_scale[0],
                    modes2=modes_scale[1],
                    width=hidden_channels,
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=4,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm
                )
            )
        
        # Create GNN component if requested
        self.gnn_component = None
        if use_gnn:
            self.gnn_component = GraphNeuralOperator(
                in_features=input_channels + 2,  # +2 for coordinates
                hidden_features=hidden_channels,
                out_features=output_channels,
                num_layers=3,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
        
        # Fusion layer to combine outputs from different scales
        fusion_inputs = num_scales * output_channels
        if use_gnn:
            fusion_inputs += output_channels
            
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(fusion_inputs, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        )
        
    def forward(self, x, grid=None, adj=None):
        """
        Forward pass of the multi-scale neural operator.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            grid: Optional coordinate grid (batch, 2, height, width)
            adj: Optional adjacency matrix for GNN (batch, nodes, nodes)
            
        Returns:
            Output tensor (batch, output_channels, height, width)
        """
        batch_size, _, height, width = x.shape
        
        # Generate grid if not provided
        if grid is None:
            y, z = torch.meshgrid(
                torch.linspace(0, 1, height, device=x.device),
                torch.linspace(0, 1, width, device=x.device)
            )
            grid = torch.stack([y, z], dim=0).unsqueeze(0)
            grid = grid.expand(batch_size, 2, height, width)
        
        # Apply FNO at different scales
        outputs = []
        
        for i in range(self.num_scales):
            # Downsample for multi-scale processing
            scale_factor = 2 ** i
            if scale_factor > 1:
                x_scaled = F.avg_pool2d(x, scale_factor)
                grid_scaled = F.avg_pool2d(grid, scale_factor)
            else:
                x_scaled = x
                grid_scaled = grid
                
            # Apply FNO
            out_scaled = self.fno_components[i](x_scaled, grid_scaled)
            
            # Upsample back to original resolution
            if scale_factor > 1:
                out_scaled = F.interpolate(
                    out_scaled, 
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                
            outputs.append(out_scaled)
        
        # Apply GNN if requested
        if self.use_gnn and self.gnn_component is not None:
            # Create adjacency matrix if not provided
            if adj is None:
                adj = self.create_adjacency_from_grid(grid)
                
            # Reshape for GNN
            x_gnn = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
            x_gnn = x_gnn.reshape(batch_size, height * width, -1)
            
            # Add coordinates
            grid_flat = grid.permute(0, 2, 3, 1)  # (batch, height, width, 2)
            grid_flat = grid_flat.reshape(batch_size, height * width, 2)
            x_gnn = torch.cat([x_gnn, grid_flat], dim=2)
            
            # Apply GNN
            out_gnn = self.gnn_component(x_gnn, adj)
            
            # Reshape back
            out_gnn = out_gnn.reshape(batch_size, height, width, self.output_channels)
            out_gnn = out_gnn.permute(0, 3, 1, 2)  # (batch, out_channels, height, width)
            
            outputs.append(out_gnn)
        
        # Combine outputs
        x = torch.cat(outputs, dim=1)
        
        # Apply fusion layer
        x = self.fusion_layer(x)
        
        return x
    
    def create_adjacency_from_grid(self, grid, threshold=0.1):
        """
        Create an adjacency matrix from grid coordinates.
        
        Args:
            grid: Grid coordinates (batch, 2, height, width)
            threshold: Distance threshold for connecting nodes
            
        Returns:
            Adjacency matrix
        """
        batch_size = grid.shape[0]
        height, width = grid.shape[2], grid.shape[3]
        num_nodes = height * width
        
        # Reshape grid to (batch, nodes, 2)
        grid_flat = grid.permute(0, 2, 3, 1).reshape(batch_size, num_nodes, 2)
        
        # Compute pairwise distances
        adj_batch = []
        for b in range(batch_size):
            grid_nodes = grid_flat[b]  # (nodes, 2)
            
            # Compute squared distances
            diff = grid_nodes.unsqueeze(1) - grid_nodes.unsqueeze(0)  # (nodes, nodes, 2)
            dist_sq = torch.sum(diff**2, dim=-1)  # (nodes, nodes)
            
            # Create adjacency matrix based on threshold
            adj = (dist_sq < threshold**2).float()
            adj_batch.append(adj)
            
        return torch.stack(adj_batch)

class PhysicsInformedNeuralOperator(nn.Module):
    """
    Physics-informed neural operator for flood modeling.
    
    This model combines neural operators with physics constraints from
    shallow water equations to improve prediction accuracy.
    """
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        fno_modes: Tuple[int, int] = (12, 12),
        hidden_channels: int = 64,
        num_scales: int = 3,
        use_gnn: bool = True,
        gravity: float = 9.81,
        manning_coef: float = 0.035,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        super(PhysicsInformedNeuralOperator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fno_modes = fno_modes
        self.hidden_channels = hidden_channels
        self.num_scales = num_scales
        self.use_gnn = use_gnn
        self.gravity = gravity
        self.manning_coef = manning_coef
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Initialize the neural operator
        self.neural_operator = MultiScaleNeuralOperator(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            fno_modes=fno_modes,
            num_scales=num_scales,
            use_gnn=use_gnn,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # Register physical parameters as buffers
        self.register_buffer('g_tensor', torch.tensor(gravity, dtype=torch.float))
        self.register_buffer('n_tensor', torch.tensor(manning_coef, dtype=torch.float))
        
    def forward(self, x, grid=None, adj=None, dx=None, dy=None, dt=None):
        """
        Forward pass of the physics-informed neural operator.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            grid: Optional coordinate grid (batch, 2, height, width)
            adj: Optional adjacency matrix for GNN (batch, nodes, nodes)
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            dt: Time step size
            
        Returns:
            Output tensor (batch, output_channels, height, width)
        """
        # Apply neural operator
        return self.neural_operator(x, grid, adj)
        
    def compute_shallow_water_residuals(self, h, u, v, h_prev, u_prev, v_prev, dx, dy, dt, elevation=None):
        """
        Compute residuals of the shallow water equations.
        
        Args:
            h: Water depth at current time (batch, height, width)
            u: x-velocity at current time (batch, height, width)
            v: y-velocity at current time (batch, height, width)
            h_prev: Water depth at previous time (batch, height, width)
            u_prev: x-velocity at previous time (batch, height, width)
            v_prev: y-velocity at previous time (batch, height, width)
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            dt: Time step size
            elevation: Optional bed elevation (batch, height, width)
            
        Returns:
            Dictionary of residuals for each equation
        """
        batch_size, height, width = h.shape
        g = self.g_tensor
        n = self.n_tensor
        
        # Ensure positive water depth for stability
        h = torch.clamp(h, min=1e-6)
        
        # Compute fluxes
        qx = h * u
        qy = h * v
        qx_prev = h_prev * u_prev
        qy_prev = h_prev * v_prev
        
        # Compute spatial derivatives using central differences
        # Mass equation: dh/dt + d(hu)/dx + d(hv)/dy = 0
        dqx_dx = (qx[:, :, 2:] - qx[:, :, :-2]) / (2 * dx)
        dqy_dy = (qy[:, 2:, :] - qy[:, :-2, :]) / (2 * dy)
        
        # Pad derivatives to maintain shape
        dqx_dx = F.pad(dqx_dx, (1, 1, 0, 0), mode='replicate')
        dqy_dy = F.pad(dqy_dy, (0, 0, 1, 1), mode='replicate')
        
        # Compute continuity equation residual: dh/dt + d(hu)/dx + d(hv)/dy = 0
        continuity_residual = (h - h_prev) / dt + dqx_dx + dqy_dy
        
        # Compute momentum equation terms
        # X-momentum: du/dt + u*du/dx + v*du/dy + g*dh/dx + g*n^2*u*sqrt(u^2+v^2)/h^(4/3) = 0
        dh_dx = (h[:, :, 2:] - h[:, :, :-2]) / (2 * dx)
        dh_dx = F.pad(dh_dx, (1, 1, 0, 0), mode='replicate')
        
        # Add bed slope if elevation is provided
        if elevation is not None:
            dz_dx = (elevation[:, :, 2:] - elevation[:, :, :-2]) / (2 * dx)
            dz_dx = F.pad(dz_dx, (1, 1, 0, 0), mode='replicate')
            dh_dx = dh_dx + dz_dx
        
        # Friction term for x-momentum
        friction_x = g * n**2 * u * torch.sqrt(u**2 + v**2) / h**(4/3)
        
        # X-momentum residual
        x_momentum_residual = (u - u_prev) / dt + g * dh_dx + friction_x
        
        # Y-momentum: dv/dt + u*dv/dx + v*dv/dy + g*dh/dy + g*n^2*v*sqrt(u^2+v^2)/h^(4/3) = 0
        dh_dy = (h[:, 2:, :] - h[:, :-2, :]) / (2 * dy)
        dh_dy = F.pad(dh_dy, (0, 0, 1, 1), mode='replicate')
        
        # Add bed slope if elevation is provided
        if elevation is not None:
            dz_dy = (elevation[:, 2:, :] - elevation[:, :-2, :]) / (2 * dy)
            dz_dy = F.pad(dz_dy, (0, 0, 1, 1), mode='replicate')
            dh_dy = dh_dy + dz_dy
        
        # Friction term for y-momentum
        friction_y = g * n**2 * v * torch.sqrt(u**2 + v**2) / h**(4/3)
        
        # Y-momentum residual
        y_momentum_residual = (v - v_prev) / dt + g * dh_dy + friction_y
        
        return {
            'continuity': continuity_residual,
            'x_momentum': x_momentum_residual,
            'y_momentum': y_momentum_residual
        }
        
    def compute_physics_loss(self, predictions, inputs, dx, dy, dt, elevation=None):
        """
        Compute physics-informed loss based on shallow water equations.
        
        Args:
            predictions: Model predictions (batch, channels, height, width)
            inputs: Input data (batch, channels, height, width)
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            dt: Time step size
            elevation: Optional bed elevation (batch, height, width)
            
        Returns:
            Dictionary of physics losses
        """
        batch_size = predictions.shape[0]
        
        # Extract predicted variables (h, u, v)
        h_pred = predictions[:, 0, :, :]
        u_pred = predictions[:, 1, :, :]
        v_pred = predictions[:, 2, :, :]
        
        # Extract previous time step variables from inputs
        h_prev = inputs[:, 0, :, :]
        u_prev = inputs[:, 1, :, :]
        v_prev = inputs[:, 2, :, :]
        
        # Get elevation if provided
        elev = None
        if elevation is not None:
            if len(elevation.shape) == 4:
                elev = elevation[:, 0, :, :]
            else:
                elev = elevation
        
        # Compute residuals
        residuals = self.compute_shallow_water_residuals(
            h_pred, u_pred, v_pred, 
            h_prev, u_prev, v_prev,
            dx, dy, dt, elev
        )
        
        # Compute mean squared residuals as physics losses
        continuity_loss = torch.mean(residuals['continuity']**2)
        x_momentum_loss = torch.mean(residuals['x_momentum']**2)
        y_momentum_loss = torch.mean(residuals['y_momentum']**2)
        
        # Combined physics loss
        physics_loss = continuity_loss + x_momentum_loss + y_momentum_loss
        
        return {
            'physics_loss': physics_loss,
            'continuity_loss': continuity_loss,
            'x_momentum_loss': x_momentum_loss,
            'y_momentum_loss': y_momentum_loss
        }
        
    def train_step(self, inputs, targets, dx, dy, dt, elevation=None, lambda_physics=0.1):
        """
        Perform a training step with combined data and physics losses.
        
        Args:
            inputs: Input data (batch, channels, height, width)
            targets: Target data (batch, channels, height, width)
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            dt: Time step size
            elevation: Optional bed elevation (batch, height, width)
            lambda_physics: Weight for physics loss
            
        Returns:
            Dictionary of losses and predictions
        """
        # Forward pass
        predictions = self(inputs)
        
        # Compute data loss (MSE)
        data_loss = F.mse_loss(predictions, targets)
        
        # Compute physics loss
        physics_losses = self.compute_physics_loss(predictions, inputs, dx, dy, dt, elevation)
        physics_loss = physics_losses['physics_loss']
        
        # Combined loss
        total_loss = data_loss + lambda_physics * physics_loss
        
        return {
            'loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'continuity_loss': physics_losses['continuity_loss'],
            'x_momentum_loss': physics_losses['x_momentum_loss'],
            'y_momentum_loss': physics_losses['y_momentum_loss'],
            'predictions': predictions
        } 