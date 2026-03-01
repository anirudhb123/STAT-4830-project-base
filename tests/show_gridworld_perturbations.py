"""
Visual demo of seeded GridWorld obstacle perturbations.

Run:
    python tests/show_gridworld_perturbations.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gridworld import GridWorld


class PerturbedLayoutGridWorld(GridWorld):
    """Mirror of the notebook's layout perturbation logic."""

    def __init__(self, layout_perturb_level: float = 0.0, max_shift: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.layout_perturb_level = float(layout_perturb_level)
        self.max_shift = int(max_shift)

    def _sample_shifted_cell(self, base_pos, forbidden):
        """Sample a nearby valid cell around base_pos."""
        for _ in range(50):
            dx = int(self.rng.randint(-self.max_shift, self.max_shift + 1))
            dy = int(self.rng.randint(-self.max_shift, self.max_shift + 1))
            nx = int(np.clip(base_pos[0] + dx, 0, self.size - 1))
            ny = int(np.clip(base_pos[1] + dy, 0, self.size - 1))
            cand = (nx, ny)
            if cand not in forbidden:
                return cand
        return base_pos

    def perturb_from_source_layout(self, source_env: GridWorld):
        """
        Move a fraction of obstacle cells and optionally the goal cell.
        """
        self.start_pos = source_env.start_pos
        self.goal_pos = source_env.goal_pos

        source_obstacles = list(source_env.obstacles)
        n_obs = len(source_obstacles)
        if n_obs == 0 or self.layout_perturb_level <= 0:
            self.obstacles = set(source_obstacles)
            return

        move_fraction = float(np.clip(self.layout_perturb_level, 0.0, 1.0))
        n_to_move = int(np.round(move_fraction * n_obs))
        n_to_move = min(max(n_to_move, 1), n_obs)

        move_idx = set(self.rng.choice(np.arange(n_obs), size=n_to_move, replace=False).tolist())
        new_obstacles = []

        for idx, obs in enumerate(source_obstacles):
            if idx not in move_idx:
                new_obstacles.append(obs)
                continue

            forbidden = {self.start_pos, self.goal_pos, *new_obstacles}
            shifted = self._sample_shifted_cell(obs, forbidden)
            new_obstacles.append(shifted)

        # Optional goal shift for higher perturbation levels.
        if self.layout_perturb_level >= 0.5:
            forbidden = {self.start_pos, *new_obstacles}
            self.goal_pos = self._sample_shifted_cell(self.goal_pos, forbidden)

        self.obstacles = set(new_obstacles)


def _draw_env(ax: plt.Axes, env: GridWorld, title: str) -> None:
    """Draw one gridworld panel on a provided axis."""
    size = env.size

    for i in range(size + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)

    for x, y in env.obstacles:
        ax.add_patch(plt.Rectangle((y, x), 1, 1, color="red", alpha=0.5))

    ax.add_patch(
        plt.Rectangle((env.goal_pos[1], env.goal_pos[0]), 1, 1, color="green", alpha=0.5)
    )
    ax.add_patch(
        plt.Circle((env.start_pos[1] + 0.5, env.start_pos[0] + 0.5), 0.3, color="blue")
    )

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def show_gridworld_perturbations(seed: int = 11) -> None:
    """Show exactly the notebook's layout-perturbation behavior."""
    size = 8
    n_obstacles = 8
    max_steps = 50
    levels = [0.0, 0.25, 0.5, 0.75]

    source_layout_env = GridWorld(
        size=size,
        n_obstacles=n_obstacles,
        max_steps=max_steps,
        seed=seed,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, level in enumerate(levels):
        env = PerturbedLayoutGridWorld(
            size=size,
            n_obstacles=n_obstacles,
            max_steps=max_steps,
            seed=seed,
            layout_perturb_level=level,
            max_shift=1,
        )
        env.perturb_from_source_layout(source_layout_env)
        row, col = divmod(i, 2)
        _draw_env(axes[row, col], env, title=f"Perturbation = {level}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_gridworld_perturbations(seed=55)
