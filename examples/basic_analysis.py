"""
Example analysis of E-I network using port-Hamiltonian framework.

This script demonstrates:
1. Creating a port-Hamiltonian E-I network model
2. Simulating network dynamics
3. Analyzing energy evolution
4. Applying control strategies
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from ei_ph import PortHamiltonianEINetwork, simulate_ei_network, SimpleController
from ei_ph.dynamics import compute_energy_trajectory
from ei_ph.control import PassivityBasedController


def example_1_free_dynamics():
    """Example 1: Free dynamics without control."""
    print("=" * 60)
    print("Example 1: Free dynamics of E-I network")
    print("=" * 60)
    
    # Create network
    network = PortHamiltonianEINetwork(
        w_EE=1.0,
        w_EI=-1.5,
        w_IE=1.2,
        w_II=-0.8,
        damping_E=0.3,
        damping_I=0.3
    )
    
    # Initial condition
    x0 = jnp.array([1.0, 0.5])
    
    # Simulate
    t, x_traj, u_traj = simulate_ei_network(
        network=network,
        x0=x0,
        t_span=(0.0, 20.0),
        dt=0.01,
        controller=None,  # No control
        method="rk4"
    )
    
    # Compute energy
    energy = compute_energy_trajectory(network, x_traj)
    
    print(f"Initial state: E={x0[0]:.3f}, I={x0[1]:.3f}")
    print(f"Final state: E={x_traj[-1, 0]:.3f}, I={x_traj[-1, 1]:.3f}")
    print(f"Initial energy: {energy[0]:.3f}")
    print(f"Final energy: {energy[-1]:.3f}")
    print(f"Energy dissipated: {energy[0] - energy[-1]:.3f}")
    print()
    
    return t, x_traj, energy


def example_2_controlled_dynamics():
    """Example 2: Controlled dynamics with simple controller."""
    print("=" * 60)
    print("Example 2: Controlled E-I network dynamics")
    print("=" * 60)
    
    # Create network
    network = PortHamiltonianEINetwork(
        w_EE=1.0,
        w_EI=-1.5,
        w_IE=1.2,
        w_II=-0.8,
        damping_E=0.2,
        damping_I=0.2
    )
    
    # Target state
    x_target = jnp.array([0.5, 0.3])
    
    # Create controller
    controller = SimpleController(
        K=jnp.eye(2) * 2.0,
        x_ref=x_target
    )
    
    # Initial condition
    x0 = jnp.array([1.5, 1.0])
    
    # Simulate
    t, x_traj, u_traj = simulate_ei_network(
        network=network,
        x0=x0,
        t_span=(0.0, 20.0),
        dt=0.01,
        controller=controller,
        method="rk4"
    )
    
    # Compute energy
    energy = compute_energy_trajectory(network, x_traj)
    
    print(f"Initial state: E={x0[0]:.3f}, I={x0[1]:.3f}")
    print(f"Target state: E={x_target[0]:.3f}, I={x_target[1]:.3f}")
    print(f"Final state: E={x_traj[-1, 0]:.3f}, I={x_traj[-1, 1]:.3f}")
    print(f"Final tracking error: {jnp.linalg.norm(x_traj[-1] - x_target):.6f}")
    print()
    
    return t, x_traj, u_traj, energy


def example_3_passivity_analysis():
    """Example 3: Passivity verification."""
    print("=" * 60)
    print("Example 3: Passivity analysis")
    print("=" * 60)
    
    # Create network
    network = PortHamiltonianEINetwork()
    
    # Test passivity at several state-input pairs
    test_points = [
        (jnp.array([1.0, 0.5]), jnp.array([0.1, 0.1])),
        (jnp.array([0.5, 1.0]), jnp.array([-0.2, 0.3])),
        (jnp.array([2.0, 1.5]), jnp.array([0.0, 0.0])),
    ]
    
    print("Testing passivity at various (state, input) pairs:")
    for i, (x, u) in enumerate(test_points):
        residual = network.passivity_check(x, u)
        is_passive = residual <= 1e-10  # Account for numerical errors
        print(f"  Point {i+1}: x={x}, u={u}")
        print(f"    Passivity residual: {residual:.6e} (Passive: {is_passive})")
    print()


def example_4_structure_analysis():
    """Example 4: Analyze port-Hamiltonian structure matrices."""
    print("=" * 60)
    print("Example 4: Port-Hamiltonian structure analysis")
    print("=" * 60)
    
    network = PortHamiltonianEINetwork()
    
    J = network.get_J_matrix()
    R = network.get_R_matrix()
    B = network.get_B_matrix()
    
    print("Interconnection matrix J (should be skew-symmetric):")
    print(J)
    print(f"  Skew-symmetry check: ||J + J^T|| = {jnp.linalg.norm(J + J.T):.6e}")
    print()
    
    print("Damping matrix R (should be symmetric and positive semi-definite):")
    print(R)
    print(f"  Symmetry check: ||R - R^T|| = {jnp.linalg.norm(R - R.T):.6e}")
    eigenvalues = jnp.linalg.eigvalsh(R)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Min eigenvalue: {jnp.min(eigenvalues):.6f} (should be >= 0)")
    print()
    
    print("Input matrix B:")
    print(B)
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Port-Hamiltonian E-I Network Analysis Examples          ║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Run examples
    example_1_free_dynamics()
    example_2_controlled_dynamics()
    example_3_passivity_analysis()
    example_4_structure_analysis()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
