# OpenCode Guidelines for epimodels

## Build/Test Commands
- Install: `pip install -e .`
- Run all tests: `pytest`
- Run single test: `pytest tests/test_continuous_models.py::test_SIR`
- Run with coverage: `pytest --cov=epimodels`
- Lint: `ruff check epimodels/`

## Code Style
- **Imports**: Standard library first, then third-party, then local modules
- **Type Hints**: Use type annotations for function parameters and return values
- **Naming**: 
  - Classes: CamelCase (e.g., `SIR`, `BaseModel`)
  - Functions/methods: snake_case (e.g., `plot_traces`)
  - Variables: snake_case
- **Error Handling**: Use try/except with specific exceptions, log errors
- **Documentation**: Docstrings for classes and methods (parameter descriptions)
- **Models**: Implement `_model` method for all model classes
- **Parameters**: Use OrderedDict for state variables and parameters