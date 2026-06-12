# 🤝 Contributing to CellFM

Thank you for your interest in contributing to CellFM! This guide will help you get started.

---

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/Single-cell-foundation-model.git
   cd Single-cell-foundation-model
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-new-feature
   ```

---

## 🔧 Development Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Git

### Install in development mode

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all development dependencies
pip install -e ".[all]"
```

### Verify the installation

```bash
# Run all tests
python -m pytest tests/ -v

# Check code style
ruff check cellfm/
```

---

## 📁 Project Structure

```
cellfm/
├── __init__.py          # Package exports and version
├── __main__.py          # Module entry point (python -m cellfm)
├── config.py            # Model configurations (80M, 800M)
├── embedding.py         # Gene expression embedding layer
├── retention.py         # Multi-scale retention mechanism (RetNet)
├── sglu.py              # SwiGLU feed-forward layer
├── model.py             # Full CellFM model assembly
├── lora.py              # LoRA fine-tuning adapters
├── data.py              # Data loading and preprocessing
├── trainer.py           # Training engine with LR scheduling
├── inference.py         # Pretrained loading and prediction
├── metrics.py           # Evaluation metrics (accuracy, F1, etc.)
├── visualization.py     # Plotting utilities (UMAP, training curves)
├── cli.py               # Command-line interface
└── convert.py           # MindSpore → PyTorch weight conversion

tests/
├── test_model.py        # Model architecture tests
├── test_data.py         # Data pipeline tests
├── test_trainer.py      # Training engine tests
├── test_inference.py    # Inference utility tests
├── test_metrics.py      # Evaluation metrics tests
├── test_visualization.py# Visualization tests
├── test_cli.py          # CLI tests
└── test_convert.py      # Weight conversion tests

docs/
├── ARCHITECTURE.md      # Technical architecture deep-dive
├── LAYMAN_GUIDE.md      # Non-technical introduction
├── API_REFERENCE.md     # API documentation
└── CONTRIBUTING.md      # This file
```

---

## 🎨 Code Style

We follow these conventions:

### General Rules

- **Line length**: 100 characters max
- **Docstrings**: Required for all public functions, classes, and modules
- **Type hints**: Required for all function signatures
- **Comments**: Explain WHY, not WHAT (the code should be self-explanatory)

### Docstring Format

Every module should start with a dual-audience docstring:

```python
"""
Module Name
=============

LAYMAN EXPLANATION:
    A plain-English description that anyone can understand.
    Use analogies and avoid jargon.

TECHNICAL DETAILS:
    Mathematical formulas, implementation details,
    and references to papers.
"""
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `CellFM`, `ERetNetBlock` |
| Functions | snake_case | `load_pretrained()`, `get_cell_embeddings()` |
| Constants | UPPER_SNAKE | `MINDSPORE_TO_PYTORCH_MAP` |
| Private methods | _prefix | `_build_decay_matrix()` |

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check cellfm/ tests/
ruff format cellfm/ tests/  # Auto-format
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ --cov=cellfm --cov-report=html
```

### Writing Tests

- Every new module should have a corresponding `tests/test_*.py` file
- Use pytest fixtures for shared setup
- Test both happy paths AND edge cases
- Use small model configs (64 dims, 2 layers) for speed

### Test Structure

```python
class TestMyFeature:
    """Test group description."""

    def test_basic_functionality(self):
        """What the test verifies."""
        result = my_function(input)
        assert result.shape == expected_shape

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(bad_input)
```

---

## 🔀 Pull Request Process

1. **Ensure all tests pass**: `python -m pytest tests/ -v`
2. **Add tests for new features**: Coverage should not decrease
3. **Update documentation**: If you add/change public APIs
4. **Follow commit message format**:
   ```
   feat: Short description of the feature
   fix: Description of the bug fix
   docs: Documentation changes
   test: Adding or fixing tests
   refactor: Code restructuring without behavior change
   ```
5. **Open a PR** with a clear description of:
   - What changed and why
   - How to test the changes
   - Any breaking changes

---

## 🏛️ Architecture Overview

If you're modifying core model components, please read [ARCHITECTURE.md](ARCHITECTURE.md) first. Key principles:

1. **Modularity**: Each component (embedding, retention, SGLU, LoRA) is self-contained
2. **Dual documentation**: Every module explains concepts in plain English AND technical detail
3. **Test-driven**: All components have comprehensive unit tests
4. **Config-driven**: Model sizes are defined by Config objects, not hardcoded

---

## 📧 Questions?

- Open a [GitHub Issue](https://github.com/Shahip2016/Single-cell-foundation-model/issues)
- Check existing issues and documentation first
- Include error messages, Python/PyTorch versions, and OS in bug reports

Thank you for contributing! 🧬
