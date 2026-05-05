FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY . .
CMD ["sh", "-c", "uv run python -m src.ETL.pipelines.realtime.local_realtime_worker & exec uv run fastapi run app/app.py --port 8000"]
