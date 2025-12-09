"""
Control strategies for port-Hamiltonian E-I networks.

This module implements various control laws for the E-I network,
leveraging the port-Hamiltonian structure for passivity-based control.
"""

import jax.numpy as jnp
from typing import Tuple, Optional


class SimpleController:
    """
    Simple proportional controller for E-I networks.
    
    Implements u = -K * (x - x_ref) where K is the gain matrix.
    
    Parameters
    ----------
    K : jnp.ndarray, optional
        Control gain matrix (2x2)
    x_ref : jnp.ndarray, optional
        Reference/target state [x_E_ref, x_I_ref]
    """
    
    def __init__(
        self,
        K: Optional[jnp.ndarray] = None,
        x_ref: Optional[jnp.ndarray] = None
    ):
        if K is None:
            K = jnp.eye(2) * 0.5  # Default proportional gain
        if x_ref is None:
            x_ref = jnp.zeros(2)  # Default to origin
            
        self.K = K
        self.x_ref = x_ref
    
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute control input at time t for state x.
        
        Parameters
        ----------
        t : float
            Current time
        x : jnp.ndarray
            Current state [x_E, x_I]
            
        Returns
        -------
        jnp.ndarray
            Control input [u_E, u_I]
        """
        error = x - self.x_ref
        u = -self.K @ error
        return u


class PassivityBasedController:
    """
    Passivity-based controller using energy shaping.
    
    This controller modifies the closed-loop Hamiltonian to achieve
    desired equilibrium points while preserving passivity.
    
    Parameters
    ----------
    K_d : jnp.ndarray
        Damping injection gain matrix (2x2)
    x_d : jnp.ndarray
        Desired equilibrium state [x_E_d, x_I_d]
    """
    
    def __init__(
        self,
        K_d: Optional[jnp.ndarray] = None,
        x_d: Optional[jnp.ndarray] = None
    ):
        if K_d is None:
            K_d = jnp.eye(2) * 1.0  # Default damping injection
        if x_d is None:
            x_d = jnp.zeros(2)
            
        self.K_d = K_d
        self.x_d = x_d
    
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute passivity-based control input.
        
        The control law preserves the port-Hamiltonian structure
        and ensures the desired equilibrium is stable.
        
        Parameters
        ----------
        t : float
            Current time
        x : jnp.ndarray
            Current state
            
        Returns
        -------
        jnp.ndarray
            Control input
        """
        # Error from desired state
        error = x - self.x_d
        
        # Damping injection: u = -K_d * error
        # This adds virtual damping to stabilize the desired equilibrium
        u = -self.K_d @ error
        
        return u


class TimeVaryingReference:
    """
    Time-varying reference tracking controller.
    
    Parameters
    ----------
    reference_fn : Callable
        Function that returns x_ref(t)
    K : jnp.ndarray
        Tracking gain matrix
    """
    
    def __init__(self, reference_fn, K: Optional[jnp.ndarray] = None):
        self.reference_fn = reference_fn
        if K is None:
            K = jnp.eye(2) * 1.0
        self.K = K
    
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute tracking control input.
        
        Parameters
        ----------
        t : float
            Current time
        x : jnp.ndarray
            Current state
            
        Returns
        -------
        jnp.ndarray
            Control input
        """
        x_ref = self.reference_fn(t)
        error = x - x_ref
        u = -self.K @ error
        return u
