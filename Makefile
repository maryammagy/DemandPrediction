.PHONY: help install test test-fast lint format demo figures clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## editable install with dev tools (macOS: brew install libomp first)
	pip install -e ".[dev]"

test:  ## full test suite (~20 s)
	pytest

test-fast:  ## skip the end-to-end tests that train models
	pytest -m "not slow"

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## auto-fix lint issues and format
	ruff check --fix .
	ruff format .

demo:  ## whole pipeline on a small synthetic dataset (~1 min)
	demand-forecast synth --out data/synthetic --n-cities 40 --scale 0.5
	demand-forecast cluster --data-dir data/synthetic --floor 200
	demand-forecast backtest --data-dir data/synthetic --windows 1404-08
	demand-forecast train --data-dir data/synthetic
	demand-forecast forecast --data-dir data/synthetic

figures:  ## regenerate docs/images from outputs/ and artifacts/
	python scripts/make_figures.py

clean:  ## remove caches and generated files
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info outputs artifacts data/synthetic
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
