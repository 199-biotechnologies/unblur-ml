# Contributing to Unblur ML

We welcome contributions. This project exists to advance security awareness around the inadequacy of visual redaction methods.

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature or fix (`git checkout -b feature/your-feature`)
3. **Make your changes** and test them
4. **Submit a pull request** with a clear description of what you changed and why

## What We're Looking For

- New model architectures or training strategies that improve accuracy at high blur levels
- Support for additional visual redaction methods (pixelation, mosaic filters)
- Better inference pipelines for real-world images (screenshots with varying fonts, backgrounds, noise)
- Edge case handling and robustness improvements
- Documentation improvements
- Bug fixes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/unblur-ml.git
cd unblur-ml

# Install dependencies
pip install -r requirements.txt

# Train a model
python -m src.train --model resnet18 --epochs 40 --time-limit 90

# Run benchmark
python -m src.benchmark --model models/resnet18_best.pt --output reports
```

## Code Style

- Keep it simple. Less code solving the right problem beats more code solving problems badly.
- Delete code when it's no longer needed.
- Use type hints where practical.
- Write clear docstrings for public functions.

## Responsible Use

This project is security research. Contributions should align with the goal of raising awareness about the risks of visual redaction, not enabling attacks on specific targets.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
