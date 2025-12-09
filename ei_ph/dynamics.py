"""
Numerical simulation utilities for E-I port-Hamiltonian networks.

This module provides JAX-based numerical integration routines for simulating
the port-Hamiltonian E-I network dynamics.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from functools import partial

from .port_hamiltonian import PortHamiltonianEINetwork


@partial(jax.jit, static_argnums=(0, 4))
def euler_step(
    dynamics_fn: Callable,
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """
    Single Euler integration step.
    
    Parameters
    ----------
    dynamics_fn : Callable
        Function that computes dx/dt = f(x, u)
    x : jnp.ndarray
        Current state
    u : jnp.ndarray
        Control input
    dt : float
        Time step
    *args
        Additional arguments to dynamics_fn
        
    Returns
    -------
    jnp.ndarray
        Next state x_{k+1}
    """
    dx_dt = dynamics_fn(x, u, *args)
    return x + dt * dx_dt


@partial(jax.jit, static_argnums=(0, 4))
def rk4_step(
    dynamics_fn: Callable,
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """
    Single Runge-Kutta 4th order integration step.
    
    Parameters
    ----------
    dynamics_fn : Callable
        Function that computes dx/dt = f(x, u)
    x : jnp.ndarray
        Current state
    u : jnp.ndarray
        Control input
    dt : float
        Time step
    *args
        Additional arguments to dynamics_fn
        
    Returns
    -------
    jnp.ndarray
        Next state x_{k+1}
    """
    k1 = dynamics_fn(x, u, *args)
    k2 = dynamics_fn(x + 0.5 * dt * k1, u, *args)
    k3 = dynamics_fn(x + 0.5 * dt * k2, u, *args)
    k4 = dynamics_fn(x + dt * k3, u, *args)
    
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_ei_network(
    network: PortHamiltonianEINetwork,
    x0: jnp.ndarray,
    t_span: Tuple[float, float],
    dt: float = 0.01,
    controller: Optional[Callable] = None,
    method: str = "rk4"
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate the port-Hamiltonian E-I network dynamics.
    
    Parameters
    ----------
    network : PortHamiltonianEINetwork
        The port-Hamiltonian network model
    x0 : jnp.ndarray
        Initial state [x_E(0), x_I(0)]
    t_span : Tuple[float, float]
        Time span (t_start, t_end)
    dt : float, optional
        Time step for integration (default: 0.01)
    controller : Callable, optional
        Controller function u = controller(t, x)
        If None, uses zero control input
    method : str, optional
        Integration method: "euler" or "rk4" (default: "rk4")
        
    Returns
    -------
    t : jnp.ndarray
        Time points
    x : jnp.ndarray
        State trajectory (n_steps x 2)
    u : jnp.ndarray
        Control input trajectory (n_steps x 2)
    """
    # Setup time array
    t_start, t_end = t_span
    t = jnp.arange(t_start, t_end, dt)
    n_steps = len(t)
    
    # Initialize arrays
    x_traj = jnp.zeros((n_steps, 2))
    u_traj = jnp.zeros((n_steps, 2))
    
    # Set initial condition
    x_traj = x_traj.at[0].set(x0)
    
    # Choose integration method
    if method == "euler":
        step_fn = euler_step
    elif method == "rk4":
        step_fn = rk4_step
    else:
        raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'")
    
    # Simulation loop
    x_current = x0
    x_list = [x0]
    u_list = []
    
    for i in range(n_steps - 1):
        # Compute control input
        if controller is not None:
            u = controller(t[i], x_current)
        else:
            u = jnp.zeros(2)
        
        u_list.append(u)
        
        # Integration step
        x_next = step_fn(network.dynamics, x_current, u, dt)
        x_list.append(x_next)
        x_current = x_next
    
    # Add final control (same as last)
    u_list.append(u_list[-1] if u_list else jnp.zeros(2))
    
    # Convert to arrays
    x_traj = jnp.array(x_list)
    u_traj = jnp.array(u_list)
    
    return t, x_traj, u_traj


def compute_energy_trajectory(
    network: PortHamiltonianEINetwork,
    x_traj: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the Hamiltonian (energy) along a state trajectory.
    
    Parameters
    ----------
    network : PortHamiltonianEINetwork
        The port-Hamiltonian network model
    x_traj : jnp.ndarray
        State trajectory (n_steps x 2)
        
    Returns
    -------
    jnp.ndarray
        Energy values along trajectory (n_steps,)
    """
    # Vectorize hamiltonian computation over trajectory
    energy_traj = jax.vmap(network.hamiltonian)(x_traj)
    return energy_traj
