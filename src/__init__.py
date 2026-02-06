"""
ES for Non-Differentiable RL

Core implementation of Evolution Strategies for sparse reward reinforcement learning.
"""

from .model import (
    GridWorld,
    HarderGridWorld,
    PolicyNetwork,
    ValueNetwork
)

from .utils import (
    evaluate_policy,
    es_gradient_estimate,
    train_es,
    plot_training_curves,
    compute_statistics,
    print_comparison_table
)

__version__ = "0.1.0"
__all__ = [
    "GridWorld",
    "HarderGridWorld",
    "PolicyNetwork",
    "ValueNetwork",
    "evaluate_policy",
    "es_gradient_estimate",
    "train_es",
    "plot_training_curves",
    "compute_statistics",
    "print_comparison_table",
]
