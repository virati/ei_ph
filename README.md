# ei_ph: Port-Hamiltonian Treatment of E-I Networks

A JAX-based implementation of port-Hamiltonian framework for analyzing and controlling Excitatory-Inhibitory (E-I) neural networks.

## Overview

This package provides a port-Hamiltonian formulation for E-I neural networks, enabling principled analysis and control design using energy-based methods. The port-Hamiltonian framework represents system dynamics as:

```
dx/dt = (J - R) * ∂H/∂x + B * u
```

where:
- `H(x)` is the Hamiltonian (energy function)
- `J` is the skew-symmetric interconnection matrix
- `R` is the symmetric damping/dissipation matrix
- `B` is the input matrix
- `u` is the control input

## Features

- **JAX-based implementation** for automatic differentiation and GPU acceleration
- **Port-Hamiltonian structure** preserving energy-based modeling
- **Multiple integration methods** (Euler, RK4) with JIT compilation
- **Control strategies**: Simple proportional control, passivity-based control
- **Passivity analysis** tools for verification
- **Energy trajectory** computation and analysis

## Installation

### From source

```bash
git clone https://github.com/virati/ei_ph.git
cd ei_ph
pip install -r requirements.txt
```

## Quick Start

```python
import jax.numpy as jnp
from ei_ph import PortHamiltonianEINetwork, simulate_ei_network, SimpleController

# Create a port-Hamiltonian E-I network
network = PortHamiltonianEINetwork(
    w_EE=1.0,    # E-E connection weight
    w_EI=-1.5,   # I-E connection weight
    w_IE=1.2,    # E-I connection weight
    w_II=-0.8,   # I-I connection weight
    damping_E=0.3,
    damping_I=0.3
)

# Initial state [Excitatory, Inhibitory]
x0 = jnp.array([1.0, 0.5])

# Simulate free dynamics
t, x_traj, u_traj = simulate_ei_network(
    network=network,
    x0=x0,
    t_span=(0.0, 20.0),
    dt=0.01,
    method="rk4"
)

# With control
controller = SimpleController(
    K=jnp.eye(2) * 2.0,
    x_ref=jnp.array([0.5, 0.3])
)

t, x_traj, u_traj = simulate_ei_network(
    network=network,
    x0=x0,
    t_span=(0.0, 20.0),
    dt=0.01,
    controller=controller
)
```

## Examples

Run the basic analysis examples:

```bash
cd examples
python basic_analysis.py
```

This will demonstrate:
1. Free dynamics of E-I network
2. Controlled dynamics with feedback
3. Passivity verification
4. Port-Hamiltonian structure analysis

## Project Structure

```
ei_ph/
├── ei_ph/
│   ├── __init__.py              # Package initialization
│   ├── port_hamiltonian.py      # Port-Hamiltonian E-I network model
│   ├── dynamics.py              # Simulation and integration routines
│   └── control.py               # Control strategies
├── examples/
│   └── basic_analysis.py        # Example usage and analysis
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Core Components

### PortHamiltonianEINetwork

The main class representing an E-I network with port-Hamiltonian structure:
- `hamiltonian(x)`: Compute energy function
- `dynamics(x, u)`: Compute state derivatives
- `gradient_hamiltonian(x)`: Compute energy gradient (uses JAX autodiff)
- `get_J_matrix()`: Get interconnection matrix (skew-symmetric)
- `get_R_matrix()`: Get damping matrix (symmetric, positive semi-definite)
- `passivity_check(x, u)`: Verify passivity condition

### Simulation

- `simulate_ei_network()`: Numerical integration with optional control
- `compute_energy_trajectory()`: Compute Hamiltonian along trajectory

### Control

- `SimpleController`: Proportional feedback control
- `PassivityBasedController`: Energy-shaping control
- `TimeVaryingReference`: Time-varying reference tracking

## Mathematical Background

The port-Hamiltonian formulation provides several advantages:
1. **Energy interpretation**: The Hamiltonian represents total system energy
2. **Passivity**: Guarantees on energy dissipation and stability
3. **Structure preservation**: Natural framework for control design
4. **Compositionality**: Easy to interconnect subsystems

For E-I networks, the state `x = [x_E, x_I]` represents excitatory and inhibitory population activities. The interconnection matrix `J` encodes conservative coupling, while `R` represents damping/adaptation.

## References

This implementation follows the port-Hamiltonian framework for neural systems. Key concepts:
- Port-Hamiltonian systems theory
- Passivity-based control
- Energy-based methods in neuroscience

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
