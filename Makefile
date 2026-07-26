.PHONY: sync up down api ui ingest test lint format eval

sync:
	uv sync --frozen

up:
	docker compose up --build

down:
	docker compose down

api:
	uv run uvicorn ffa.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	uv run streamlit run src/ffa/ui/app.py

ingest:
	uv run python -m ffa.ingestion.run

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
