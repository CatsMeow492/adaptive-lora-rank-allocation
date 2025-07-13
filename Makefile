.PHONY: install test lint format clean run-single run-all setup

# Installation
install:
	pip install -r requirements.txt

# Development setup
setup: install
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

# Linting and formatting
lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

# Clean up
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf results/
	rm -rf wandb/

# Run single experiment (example)
run-single:
	python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased

# Run all experiments
run-all:
	python run_all_experiments.py

# Quick test run with small epochs
test-run:
	python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased --epochs 1 --batch-size 4

# Help
help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  setup      - Development setup with pre-commit"
	@echo "  test       - Run tests with coverage"
	@echo "  lint       - Run linting checks"
	@echo "  format     - Format code"
	@echo "  clean      - Clean up temporary files"
	@echo "  run-single - Run single experiment example"
	@echo "  run-all    - Run all experiments"
	@echo "  test-run   - Quick test run with minimal epochs"
	@echo "  help       - Show this help message" 