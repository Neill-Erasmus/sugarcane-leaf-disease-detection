# Contributing to Sugarcane Leaf Disease Detection

Thank you for your interest in contributing to this project! We welcome contributions from everyone, including bug reports, feature requests, documentation improvements, and code contributions.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions. Discrimination, harassment, or hate speech of any kind will not be tolerated.

## How to Contribute

### Reporting Bugs

If you encounter a bug, please open an issue with:
- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- A clear, descriptive title
- A detailed description of the proposed enhancement
- Use cases and motivation
- Potential implementation approaches

### Code Contributions

#### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/sugarcane-leaf-disease-detection.git
   cd sugarcane-leaf-disease-detection
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Workflow

1. **Make your changes** in your feature branch
2. **Write or update tests** as needed
3. **Test locally** to ensure everything works:
   ```bash
   python -m pytest  # if tests exist
   python src/evaluation/evaluate.py  # for model validation
   ```
4. **Update documentation** if applicable (README, docstrings, config.yaml)
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: Brief description of changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request** on GitHub

#### Pull Request Guidelines

- **Description**: Provide a clear description of the changes and why they're needed
- **Scope**: Keep PRs focused on a single feature or bug fix
- **Branch naming**: Use descriptive names (e.g., `fix/data-loader-bug`, `feature/new-model`)
- **Commits**: Use clear, atomic commits
- **Testing**: Ensure all existing functionality still works
- **Documentation**: Update README or docstrings if needed

### Code Style

- Follow **PEP 8** guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Use type hints where appropriate
- Keep lines to a reasonable length (≤100 characters)

### Documentation Contributions

Documentation improvements are highly valued! You can contribute by:
- Fixing typos or unclear explanations
- Adding examples to the README
- Improving docstrings
- Adding usage guides or tutorials

## Project Structure

Key directories for contributors:

```
src/
  ├── models/           # Model architectures (baseline CNN, ResNet)
  ├── training/         # Training scripts
  ├── evaluation/       # Evaluation and validation scripts
  └── data/             # Data loading utilities

api/                    # FastAPI application
experiments/            # Model checkpoints and results
config.yaml             # Configuration file for hyperparameters
```

## Model Information

This project implements three models:
- **Baseline CNN**: Simple convolutional neural network
- **ResNet-50 (Frozen)**: Transfer learning with frozen backbone
- **ResNet-50 (Fine-Tuned)**: Transfer learning with fine-tuned layers

All models classify sugarcane leaves into 5 disease categories: Healthy, Mosaic, RedRot, Rust, Yellow

## Testing

Before submitting a PR, test:
1. **Model training**: Run training scripts to ensure convergence
2. **API functionality**: Test the FastAPI endpoint with sample images
3. **Docker build**: Ensure the Dockerfile builds successfully
4. **Cross-platform compatibility**: Test on Windows, macOS, and Linux if possible

## Questions or Need Help?

- Open an issue with your question (tag as `question`)
- Check existing issues and discussions first
- Reach out to the maintainers if needed

Thank you for contributing to advancing agricultural AI! 🌾