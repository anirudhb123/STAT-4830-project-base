import os
import pathlib
import subprocess


def test_gd_script_runs_and_creates_figure():
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    if os.name == "nt":
        python_path = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        python_path = repo_root / ".venv" / "bin" / "python"

    assert python_path.exists(), (
        f"Expected virtual environment python at {python_path}. "
        "Run the installer script first."
    )

    script_path = repo_root / "script" / "gd_1d_torch.py"
    figure_path = repo_root / "figures" / "gd_torch_quadratic_diagnostics.png"
    if figure_path.exists():
        figure_path.unlink()

    subprocess.run(
        [str(python_path), str(script_path)],
        cwd=str(repo_root),
        check=True,
    )

    assert figure_path.exists(), "Expected output figure was not created."
