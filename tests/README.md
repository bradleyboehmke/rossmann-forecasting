# Tests

Comprehensive test suite for the Rossmann forecasting project.

## Test Organization

### Unit Tests

- `test_data_validation.py` - Data quality, schema, and Great Expectations tests
- `test_features.py` - Feature engineering correctness and leakage prevention
- `test_models.py` - Model training, prediction, and serialization
- `test_utils.py` - Utility function tests

### Integration Tests

- `test_api.py` - FastAPI endpoint tests
- `test_pipeline.py` - End-to-end pipeline tests

### Fixtures

- `conftest.py` - Shared pytest fixtures (sample data, temp directories, etc.)

## Running Tests

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_features.py
```

### Run specific test

```bash
pytest tests/test_features.py::test_lag_features_no_leakage
```

### Run only unit tests

```bash
pytest -m unit
```

### Run only integration tests

```bash
pytest -m integration
```

## Test Markers

Tests are marked with pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (skip with `-m "not slow"`)

## Coverage Goals

Target: >80% code coverage for production code (`src/`)

Key areas requiring coverage:

- Data validation logic
- Feature engineering functions
- Model training and prediction
- API endpoints
- Utility functions

## CI/CD Integration

Tests are automatically run in GitHub Actions on:

- Pull requests to main
- Pushes to main
- Manual workflow dispatch

See `.github/workflows/test.yml` for configuration.
