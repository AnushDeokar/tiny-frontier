.PHONY: install test lint format train clean

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

train:
	poetry run python -m tiny_frontier.train --config configs/dummy.yaml

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
