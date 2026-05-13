"""
ES for Non-Differentiable RL

Core implementation of Evolution Strategies for sparse reward reinforcement learning.

Note on usage:

This directory is laid out as a flat module collection rather than a real
installable package. Notebooks, scripts, and tests bring it onto ``sys.path``
explicitly (e.g. ``sys.path.insert(0, str(Path(__file__).resolve().parent.parent
/ 'src'))``) and then import individual modules at the top level::

    from wordle_env import load_wordle_environment
    from es_wordle import train_es_wordle
    from gridworld import GridWorld

The re-exports below cover only the GridWorld stack and exist for
``import src as es`` convenience; new Wordle modules deliberately are not
re-exported here so the flat-module pattern stays the single import surface.
"""

from .gridworld import (
    GridWorld,
    HarderGridWorld,
    PolicyNetwork,
    ValueNetwork
)

from .es_gridworld import (
    NOISE_TYPES,
    sample_perturbation,
    score_function,
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
    "NOISE_TYPES",
    "sample_perturbation",
    "score_function",
    "evaluate_policy",
    "es_gradient_estimate",
    "train_es",
    "plot_training_curves",
    "compute_statistics",
    "print_comparison_table",
]
