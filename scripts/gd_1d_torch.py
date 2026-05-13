from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def run_gradient_descent(
    initial_x: float = 8.0,
    learning_rate: float = 0.1,
    steps: int = 60,
):
    x = torch.tensor([initial_x], dtype=torch.float32, requires_grad=True)
    xs = []
    losses = []

    for _ in range(steps):
        loss = (x - 3.0) ** 2
        loss.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
        x.grad.zero_()

        xs.append(float(x.detach().item()))
        losses.append(float(loss.detach().item()))

    return xs, losses


def main() -> None:
    matplotlib.use("Agg")

    repo_root = Path(__file__).resolve().parent.parent
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "gd_torch_quadratic_diagnostics.png"

    xs, losses = run_gradient_descent()
    iter_idx = np.arange(1, len(xs) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(iter_idx, xs, color="tab:blue")
    axes[0].axhline(3.0, color="tab:red", linestyle="--", label="optimum x=3")
    axes[0].set_title("x over iterations")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("x")
    axes[0].legend()

    axes[1].plot(iter_idx, losses, color="tab:green")
    axes[1].set_title("loss over iterations")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss = (x-3)^2")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
