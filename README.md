# Renewable Energy Systems

This is a tutorial for Solar Supply Forecasting with XGBoost.
Creator: Julius Rau

## Usage

### Data

You will receive the data when cloning this repo only if you have git-lfs installed on your system. If you do not have git-lfs, install it and run

```shell
git lfs install
git lfs pull 
```

### Dependencies

All packages are installable via the package manager [uv](https://docs.astral.sh/uv/) or other package managers compatible with `pyproject.toml` files.

For creating a .venv with all dependencies installed using uv, run `uv sync`

The dependencies include jupyter, so the created .venv can be used as the jupyter kernel for the notebook.
