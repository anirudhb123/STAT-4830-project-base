# Repo Quickstart (uv)

This repository is set up to run on macOS, Linux, and Windows using `uv`.

## macOS/Linux quickstart

From the repository root:

```bash
bash script/install.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the script:

```bash
python script/gd_1d_torch.py
```

Run tests:

```bash
pytest
```

## Windows PowerShell quickstart

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\install.ps1
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the script:

```powershell
python .\script\gd_1d_torch.py
```

Run tests:

```powershell
pytest
```

## Notes

- Install scripts detect whether `uv` is already installed.
- If `uv` is missing, scripts install it using Astral's official installer.
- Dependencies are installed from `requirements.txt` into `.venv` in the repo root.
