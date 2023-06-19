# Brain Tumor Segmentation (BTS) Design Project


## Getting Started

```
pre-commit install --install-hooks
```

## Install Poetry
To install poetry, simply run
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install project dependencies. This will create a virtual environment and install the
project dependencies in that. To install the virtual environment with the same directory
as the project, do:
```bash
poetry config virtualenvs.in-project true
```
```bash
poetry install
```

If poetry fails at `poetry install` stage, with error
"Failed to create the collection: Prompt dismissed.." try:
```bash
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install
```

To add a package use
```bash
poetry add <package-name>
```
And, to remove a package
```bash
poetry remove <package-name>
```


## Notes
Before allocating a GPU be sure to ``poetry shell`` is not executed. In other terms,
execute ``poetry shell`` after a GPU has been allocated.

## Training
```bash
sbatch scripts/echidna_swin_train.sh
```

## Inference
```bash
sbatch scripts/echidna_swin_inference.sh
```
