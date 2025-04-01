# Flood Warning System - PINN Model

This project implements a Physics-Informed Neural Network (PINN) model for flood prediction. I use ANUGA simulations to generate training data, which helps the model understand the physics behind flood behavior.

## Project Structure

```
flood_warning_system/
├── config/                      # Configuration files
│   ├── problem_definition.yaml  # Flood problem definition
│   ├── data_processing.yaml     # Data processing settings
│   ├── simulation_base.yaml     # ANUGA simulation base settings
│   ├── simulation_scenarios.yaml# Different simulation scenarios
│   └── config.py               # Configuration loading utilities
├── models/                      # Model architecture definitions
│   ├── pinn.py                 # Core PINN model
│   ├── multi_scale_pinn.py     # Multi-scale version with scale adaptivity
│   ├── neural_operators.py     # Neural operators implementation
│   └── anuga_simulator.py      # ANUGA simulation wrapper
├── utils/                       # Utility functions
│   ├── visualization.py        # Data visualization tools
│   ├── gis_utils.py            # Geographic information utilities
│   └── validation_metrics.py   # Model validation metrics
├── run_simulations.py          # Runs flood simulations with ANUGA
├── prepare_pinn_data.py        # Prepares data for PINN training
├── train_pinn_optimized.py     # Memory-optimized training script
├── train_multi_scale_pinn.py   # Multi-scale model training
├── validate_pinn_model.py      # Model validation against test data
├── integrate_pinn_model.py     # Integration utilities
├── optimize_hyperparameters.py # Hyperparameter optimization
└── efficient_batch_processor.py# Memory-efficient batch processing
```

## Core Components

### 1. Model Architecture (`models/`)
- **PINN Model** (`pinn.py`): Base physics-informed neural network
- **Multi-scale PINN** (`multi_scale_pinn.py`): Extended version that works at multiple spatial scales
- **Neural Operators** (`neural_operators.py`): Implementation of neural operators for PDE solving
- **ANUGA Interface** (`anuga_simulator.py`): Interface to the ANUGA hydrodynamic engine

### 2. Training & Validation Scripts
- **Data Generation**: `run_simulations.py` creates synthetic flood data via ANUGA
- **Data Preparation**: `prepare_pinn_data.py` processes simulation results for training
- **Model Training**: Choose between `train_pinn_optimized.py` (memory-efficient) or `train_multi_scale_pinn.py`
- **Model Validation**: `validate_pinn_model.py` assesses model performance

### 3. Utilities
- **Hyperparameter Optimization**: `optimize_hyperparameters.py` finds optimal settings
- **Memory Management**: `efficient_batch_processor.py` enables training on limited hardware
- **Model Integration**: `integrate_pinn_model.py` helps integrate the model with other systems

## Installation & Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Complete Workflow

### 1. Generate Training Data
```bash
# Run simulations to generate synthetic flood data
python run_simulations.py --config config/simulation_base.yaml --scenarios config/simulation_scenarios.yaml --output_dir simulations/results
```

### 2. Prepare Data for PINN
```bash
# Process simulation results into PINN-compatible format
python prepare_pinn_data.py --input simulations/results --output_dir pinn_data
```

### 3. Train the Model
```bash
# Standard training
python train_pinn_optimized.py --config config/simulation_base.yaml --data_path pinn_data --output_dir model_outputs

# OR for multi-scale version
python train_multi_scale_pinn.py --config config/multi_scale_pinn_config.yaml --data_path pinn_data --output_dir model_outputs
```

### 4. Validate Model Performance
```bash
# Validate against test data
python validate_pinn_model.py --model_path model_outputs/model_best.pth --test_data simulations/test_cases
```

## The Physics: Shallow Water Equations

The model is based on the Shallow Water Equations (SWE):

1. **Mass Conservation**:
   ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0

2. **Momentum Conservation**:
   ∂(hu)/∂t + ∂(hu²)/∂x + ∂(huv)/∂y = -gh∂z/∂x + S_fx
   ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv²)/∂y = -gh∂z/∂y + S_fy

Where:
- h: water depth
- u, v: flow velocity components
- g: gravity (9.81 m/s²)
- z: terrain elevation
- S_fx, S_fy: friction terms 