.PHONY: sync up down test lint format eval

sync:
	uv sync --frozen

up:
	docker compose up --build

down:
	docker compose down

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

eval:
	uv run python -m ffa.evaluation.eval_retrieval
	uv run python -m ffa.evaluation.eval_generation

