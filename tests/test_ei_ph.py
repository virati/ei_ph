"""
Comprehensive test script for port-Hamiltonian E-I network.

This script validates all major functionality of the implementation.
"""

import jax.numpy as jnp
from ei_ph import PortHamiltonianEINetwork, simulate_ei_network, SimpleController
from ei_ph.dynamics import compute_energy_trajectory
from ei_ph.control import PassivityBasedController, TimeVaryingReference


def test_network_creation():
    """Test that network can be created with default and custom parameters."""
    print("Testing network creation...")
    
    # Default parameters
    network1 = PortHamiltonianEINetwork()
    assert network1 is not None
    
    # Custom parameters
    network2 = PortHamiltonianEINetwork(
        w_EE=2.0, w_EI=-2.0, w_IE=1.5, w_II=-1.0,
        damping_E=0.4, damping_I=0.6
    )
    assert network2.w_EE == 2.0
    assert network2.damping_E == 0.4
    
    print("  ✓ Network creation tests passed")


def test_hamiltonian_computation():
    """Test Hamiltonian computation."""
    print("Testing Hamiltonian computation...")
    
    network = PortHamiltonianEINetwork()
    x = jnp.array([1.0, 0.5])
    
    H = network.hamiltonian(x)
    assert jnp.isfinite(H), "Hamiltonian should be finite"
    
    # Energy should be zero at origin for this formulation
    x_zero = jnp.array([0.0, 0.0])
    H_zero = network.hamiltonian(x_zero)
    assert jnp.abs(H_zero) < 1e-10, "Energy at origin should be near zero"
    
    print("  ✓ Hamiltonian computation tests passed")


def test_structure_matrices():
    """Test that J and R have proper structure."""
    print("Testing structure matrices...")
    
    network = PortHamiltonianEINetwork()
    
    # Test J is skew-symmetric
    J = network.get_J_matrix()
    skew_error = jnp.linalg.norm(J + J.T)
    assert skew_error < 1e-10, f"J should be skew-symmetric, error: {skew_error}"
    
    # Test R is symmetric
    R = network.get_R_matrix()
    sym_error = jnp.linalg.norm(R - R.T)
    assert sym_error < 1e-10, f"R should be symmetric, error: {sym_error}"
    
    # Test R is positive semi-definite
    eigenvalues = jnp.linalg.eigvalsh(R)
    assert jnp.all(eigenvalues >= -1e-10), f"R should be PSD, min eigenvalue: {jnp.min(eigenvalues)}"
    
    print("  ✓ Structure matrix tests passed")


def test_gradient_computation():
    """Test automatic differentiation of Hamiltonian."""
    print("Testing gradient computation...")
    
    network = PortHamiltonianEINetwork()
    x = jnp.array([1.0, 0.5])
    
    grad_H = network.gradient_hamiltonian(x)
    assert grad_H.shape == (2,), "Gradient should be 2D"
    assert jnp.all(jnp.isfinite(grad_H)), "Gradient should be finite"
    
    print("  ✓ Gradient computation tests passed")


def test_dynamics_computation():
    """Test dynamics computation."""
    print("Testing dynamics computation...")
    
    network = PortHamiltonianEINetwork()
    x = jnp.array([1.0, 0.5])
    u = jnp.zeros(2)
    
    dx_dt = network.dynamics(x, u)
    assert dx_dt.shape == (2,), "Dynamics should be 2D"
    assert jnp.all(jnp.isfinite(dx_dt)), "Dynamics should be finite"
    
    # With control input
    u = jnp.array([0.1, -0.1])
    dx_dt_controlled = network.dynamics(x, u)
    assert not jnp.allclose(dx_dt, dx_dt_controlled), "Control should affect dynamics"
    
    print("  ✓ Dynamics computation tests passed")


def test_simulation():
    """Test numerical simulation."""
    print("Testing simulation...")
    
    network = PortHamiltonianEINetwork(damping_E=0.5, damping_I=0.5)
    x0 = jnp.array([1.0, 0.5])
    
    # Test without control
    t, x_traj, u_traj = simulate_ei_network(
        network, x0, t_span=(0.0, 5.0), dt=0.01, method="rk4"
    )
    
    assert len(t) == len(x_traj), "Time and trajectory lengths should match"
    assert x_traj.shape[1] == 2, "State should be 2D"
    
    # Energy should decrease (due to damping)
    energy = compute_energy_trajectory(network, x_traj)
    assert energy[0] >= energy[-1] - 1e-6, "Energy should decrease (or stay same)"
    
    print("  ✓ Simulation tests passed")


def test_controllers():
    """Test different controller types."""
    print("Testing controllers...")
    
    network = PortHamiltonianEINetwork()
    x0 = jnp.array([1.0, 0.8])
    
    # Simple controller
    controller1 = SimpleController(K=jnp.eye(2) * 1.0, x_ref=jnp.zeros(2))
    u1 = controller1(0.0, x0)
    assert u1.shape == (2,), "Control should be 2D"
    
    # Passivity-based controller
    controller2 = PassivityBasedController(K_d=jnp.eye(2) * 0.5, x_d=jnp.zeros(2))
    u2 = controller2(0.0, x0)
    assert u2.shape == (2,), "Control should be 2D"
    
    # Time-varying reference
    ref_fn = lambda t: jnp.array([jnp.sin(t), jnp.cos(t)])
    controller3 = TimeVaryingReference(ref_fn, K=jnp.eye(2))
    u3 = controller3(1.0, x0)
    assert u3.shape == (2,), "Control should be 2D"
    
    print("  ✓ Controller tests passed")


def test_passivity():
    """Test passivity property."""
    print("Testing passivity...")
    
    network = PortHamiltonianEINetwork(damping_E=0.5, damping_I=0.5)
    
    # Test several points
    test_cases = [
        (jnp.array([1.0, 0.5]), jnp.array([0.0, 0.0])),
        (jnp.array([0.5, 1.0]), jnp.array([0.1, 0.1])),
        (jnp.array([2.0, 1.5]), jnp.array([-0.2, 0.3])),
    ]
    
    for x, u in test_cases:
        residual = network.passivity_check(x, u)
        # For a passive system with damping, residual should be <= 0
        assert residual <= 1e-8, f"System should be passive at x={x}, u={u}, residual={residual}"
    
    print("  ✓ Passivity tests passed")


def test_energy_conservation_without_damping():
    """Test energy conservation in undamped system."""
    print("Testing energy conservation (undamped)...")
    
    # Create network with no damping
    network = PortHamiltonianEINetwork(damping_E=0.0, damping_I=0.0)
    x0 = jnp.array([1.0, 0.5])
    
    # Simulate without control
    t, x_traj, _ = simulate_ei_network(
        network, x0, t_span=(0.0, 10.0), dt=0.001, method="rk4"
    )
    
    energy = compute_energy_trajectory(network, x_traj)
    
    # Energy should be approximately conserved
    energy_variation = jnp.max(energy) - jnp.min(energy)
    energy_mean = jnp.mean(energy)
    relative_variation = energy_variation / (jnp.abs(energy_mean) + 1e-10)
    
    assert relative_variation < 0.01, f"Energy should be conserved, variation: {relative_variation}"
    
    print("  ✓ Energy conservation test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Port-Hamiltonian E-I Network Tests")
    print("="*60 + "\n")
    
    tests = [
        test_network_creation,
        test_hamiltonian_computation,
        test_structure_matrices,
        test_gradient_computation,
        test_dynamics_computation,
        test_simulation,
        test_controllers,
        test_passivity,
        test_energy_conservation_without_damping,
    ]
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            raise
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            raise
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
