FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
RUN uv sync --no-dev
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "/app/.venv/bin/python -m src.ETL.pipelines.realtime.local_realtime_worker & exec /app/.venv/bin/fastapi run app/app.py --port 8000"]
