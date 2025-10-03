# Baseline Template

A baseline template for machine learning projects.

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
# Or using pip
pip install poetry
```

### 3. Install Dependencies
```bash
poetry install
```

## Project Structure

```
├── data/               # Data directory
│   ├── raw/           # Raw data
│   ├── interim/       # Intermediate data
│   └── processed/     # Processed data
├── docs/              # Documentation
│   ├── data.md        # Data documentation
│   ├── evaluation.md  # Evaluation metrics
│   └── task.md        # Task description
├── notebooks/         # Jupyter notebooks
├── scripts/           # Utility scripts
├── src/case_baseline/ # Source code
│   ├── core/          # Core functionality
│   ├── domain/        # Domain logic
│   ├── modes/         # Different execution modes
│   └── utils/         # Utilities
└── tests/             # Unit tests
```

## Code Quality

### Formatting and Linting
Use Ruff for code formatting and linting:

```bash
# Format code
poetry run ruff format .

# Lint code
poetry run ruff check .

# Fix auto-fixable issues
poetry run ruff check --fix .
```

### Pre-commit Hooks
Install pre-commit hooks for automatic code quality checks:

```bash
poetry run pre-commit install
```

## Testing

Run tests with coverage:

```bash
poetry run pytest
```
