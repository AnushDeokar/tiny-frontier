.PHONY: install test lint format train sweep-lr seeds clean

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

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
