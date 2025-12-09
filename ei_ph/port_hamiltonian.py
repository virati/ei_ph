"""
Port-Hamiltonian formulation for Excitatory-Inhibitory networks.

This module implements the port-Hamiltonian framework for modeling and analyzing
E-I neural networks. The port-Hamiltonian formulation expresses dynamics as:
    dx/dt = (J - R) * ∂H/∂x + B * u
where:
    - H(x) is the Hamiltonian (energy function)
    - J is the skew-symmetric interconnection matrix
    - R is the symmetric damping/dissipation matrix
    - B is the input matrix
    - u is the control input
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Optional
from functools import partial


class PortHamiltonianEINetwork:
    """
    Port-Hamiltonian model of an Excitatory-Inhibitory neural network.
    
    The state vector x = [x_E, x_I] represents the activity of excitatory
    and inhibitory populations respectively.
    
    Parameters
    ----------
    w_EE : float
        Excitatory to excitatory connection weight
    w_EI : float
        Inhibitory to excitatory connection weight (typically negative)
    w_IE : float
        Excitatory to inhibitory connection weight
    w_II : float
        Inhibitory to inhibitory connection weight (typically negative)
    damping_E : float
        Damping coefficient for excitatory population
    damping_I : float
        Damping coefficient for inhibitory population
    """
    
    def __init__(
        self,
        w_EE: float = 1.0,
        w_EI: float = -1.5,
        w_IE: float = 1.2,
        w_II: float = -0.8,
        damping_E: float = 0.5,
        damping_I: float = 0.5,
    ):
        self.w_EE = w_EE
        self.w_EI = w_EI
        self.w_IE = w_IE
        self.w_II = w_II
        self.damping_E = damping_E
        self.damping_I = damping_I
        
    def hamiltonian(self, x: jnp.ndarray) -> float:
        """
        Compute the Hamiltonian (energy) of the E-I network.
        
        The Hamiltonian represents the total energy stored in the network,
        combining kinetic-like terms and potential-like interaction terms.
        
        Parameters
        ----------
        x : jnp.ndarray
            State vector [x_E, x_I]
            
        Returns
        -------
        float
            Hamiltonian energy value
        """
        x_E, x_I = x[0], x[1]
        
        # Quadratic energy terms (analogous to kinetic energy)
        kinetic_term = 0.5 * (x_E**2 + x_I**2)
        
        # Interaction energy (analogous to potential energy)
        interaction_term = (
            0.5 * self.w_EE * x_E**2 +
            (self.w_EI + self.w_IE) * x_E * x_I +
            0.5 * self.w_II * x_I**2
        )
        
        return kinetic_term + interaction_term
    
    def get_J_matrix(self) -> jnp.ndarray:
        """
        Get the skew-symmetric interconnection matrix J.
        
        The interconnection matrix encodes the conservative (energy-preserving)
        coupling between system components.
        
        Returns
        -------
        jnp.ndarray
            2x2 skew-symmetric interconnection matrix
        """
        # Skew-symmetric matrix: J = -J^T
        # Represents conservative energy exchange between populations
        J = jnp.array([
            [0.0, 0.5],      # E-I coupling (conservative)
            [-0.5, 0.0]      # I-E coupling (antisymmetric)
        ])
        return J
    
    def get_R_matrix(self) -> jnp.ndarray:
        """
        Get the symmetric damping/dissipation matrix R.
        
        The damping matrix encodes energy dissipation in the system.
        
        Returns
        -------
        jnp.ndarray
            2x2 symmetric damping matrix
        """
        # Symmetric positive semi-definite matrix: R = R^T, R >= 0
        # Represents energy dissipation
        R = jnp.array([
            [self.damping_E, 0.0],
            [0.0, self.damping_I]
        ])
        return R
    
    def get_B_matrix(self) -> jnp.ndarray:
        """
        Get the input matrix B.
        
        Returns
        -------
        jnp.ndarray
            2x2 input distribution matrix
        """
        # Input matrix determines how external control affects each population
        B = jnp.eye(2)
        return B
    
    @partial(jax.jit, static_argnums=(0,))
    def gradient_hamiltonian(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the gradient of the Hamiltonian with respect to state.
        
        Parameters
        ----------
        x : jnp.ndarray
            State vector [x_E, x_I]
            
        Returns
        -------
        jnp.ndarray
            Gradient vector ∂H/∂x
        """
        return jax.grad(self.hamiltonian)(x)
    
    @partial(jax.jit, static_argnums=(0,))
    def dynamics(
        self,
        x: jnp.ndarray,
        u: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the port-Hamiltonian dynamics.
        
        dx/dt = (J - R) * ∂H/∂x + B * u
        
        Parameters
        ----------
        x : jnp.ndarray
            State vector [x_E, x_I]
        u : jnp.ndarray, optional
            Control input [u_E, u_I]
            
        Returns
        -------
        jnp.ndarray
            Time derivative dx/dt
        """
        if u is None:
            u = jnp.zeros(2)
            
        # Get system matrices
        J = self.get_J_matrix()
        R = self.get_R_matrix()
        B = self.get_B_matrix()
        
        # Compute gradient of Hamiltonian
        grad_H = self.gradient_hamiltonian(x)
        
        # Port-Hamiltonian dynamics: dx/dt = (J - R) * ∂H/∂x + B * u
        dx_dt = (J - R) @ grad_H + B @ u
        
        return dx_dt
    
    def passivity_check(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """
        Verify passivity condition for the port-Hamiltonian system.
        
        For a passive system: dH/dt <= u^T * y, where y is the output.
        
        Parameters
        ----------
        x : jnp.ndarray
            State vector
        u : jnp.ndarray
            Control input
            
        Returns
        -------
        float
            Passivity residual (should be <= 0 for passive systems)
        """
        # Compute dH/dt
        grad_H = self.gradient_hamiltonian(x)
        dx_dt = self.dynamics(x, u)
        dH_dt = jnp.dot(grad_H, dx_dt)
        
        # Output y = B^T * ∂H/∂x (collocated port)
        B = self.get_B_matrix()
        y = B.T @ grad_H
        
        # Passivity condition: dH/dt <= u^T * y
        passivity_residual = dH_dt - jnp.dot(u, y)
        
        return passivity_residual
