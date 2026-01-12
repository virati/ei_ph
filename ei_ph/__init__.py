"""
ei_ph: Port-Hamiltonian treatment of Excitatory-Inhibitory networks
"""

__version__ = "0.1.0"

from .port_hamiltonian import PortHamiltonianEINetwork
from .dynamics import simulate_ei_network
from .control import SimpleController

__all__ = [
    "PortHamiltonianEINetwork",
    "simulate_ei_network",
    "SimpleController",
]
