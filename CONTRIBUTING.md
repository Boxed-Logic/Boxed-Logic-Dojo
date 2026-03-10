# Contributing to Boxed Logic Dojo

Thanks for your interest in contributing! Here's how to get started.

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
git clone https://github.com/Boxed-Logic/Boxed-Logic-Dojo.git
cd Boxed-Logic-Dojo
pip install torch transformers peft vllm datasets pytest
```

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a PR. The CI workflow runs the same command on push and PR.

## Code Style

- Follow existing patterns in the codebase
- Use type annotations for function signatures
- Keep modules focused and minimal — avoid unnecessary abstractions
- Add docstrings to public functions and classes
- Guard `content=None` on tool-call messages with `msg.get("content") or ""`

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Make your changes** — keep commits focused and well-described
3. **Add tests** for new functionality in `tests/`
4. **Run the test suite** to make sure everything passes
5. **Open a PR** against `main` with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub with:
- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs actual behavior
- Python version and relevant package versions
