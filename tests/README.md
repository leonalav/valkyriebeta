# Testing Requirements

To run the configuration validation tests:

1. Ensure Python 3.8+ is installed
2. Install required packages:
```bash
pip install pytest torch
```

3. Run tests from project root:
```bash
pytest tests/test_training_config.py -v
```

## Expected Test Coverage

- Hardware capability validation
- Memory budgeting checks
- Attention mechanism fallback behavior
- Configuration compatibility warnings

## Test Dependencies

- pytest
- torch
- All configuration classes from config/training_config.py
