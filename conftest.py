# Placing a conftest.py at the project root makes pytest add this directory to
# sys.path, so tests can `import main` regardless of how pytest is invoked
# (`pytest` vs. `python -m pytest`). Without it, a bare `pytest` run — as used
# in CI — fails with ModuleNotFoundError: No module named 'main'.
