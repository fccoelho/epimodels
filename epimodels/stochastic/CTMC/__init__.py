"""
Continuous-Time Markov Chain models.

This module provides stochastic epidemic models based on CTMCs,
solved using the Gillespie Stochastic Simulation Algorithm (SSA)
and related methods.

Usage:
    from epimodels.stochastic.CTMC import SIR, GillespieSolver

    model = SIR()
    model([990, 10, 0], [0, 100], 1000,
          {'beta': 0.3, 'gamma': 0.1}, reps=100, seed=42)
    model.plot_traces()
"""

from epimodels.stochastic.CTMC.models import SEIR, SIR, SIRS, SIS, CTMCModel
from epimodels.stochastic.CTMC.solvers import (
    CTMCSolverBase,
    CTMCTrajectory,
    GillespieSolver,
)

__all__ = [
    "CTMCModel",
    "SIR",
    "SIS",
    "SIRS",
    "SEIR",
    "CTMCSolverBase",
    "GillespieSolver",
    "CTMCTrajectory",
]
