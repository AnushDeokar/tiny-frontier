.PHONY: install test lint format train sweep-lr seeds exp1 clean

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

train:
	poetry run python -m tiny_frontier.training.train --config configs/dummy.yaml

# caffeinate -i keeps the Mac awake while training (macOS only).
sweep-lr:
	caffeinate -i poetry run python scripts/sweep.py lr

seeds:
	caffeinate -i poetry run python scripts/sweep.py seeds

# Experiment 1: the 5 local/global variants, seed 1337 (~2.3 h).
exp1:
	caffeinate -i poetry run python scripts/sweep.py exp1

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
