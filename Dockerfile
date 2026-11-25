# Lightweight API container: talks to a vLLM (OpenAI-compatible) server hosting base model + LoRA adapter,
# or to Anthropic. GPU model serving lives in the vLLM container (see docker-compose.yml).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FLIGHTLAKE_ROOT=/data/lake FLEET_BASELINE=/data/sft/fleet_baseline.json
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic]"
EXPOSE 8080
CMD ["uvicorn", "flightdebrief.serve:app", "--host", "0.0.0.0", "--port", "8080"]
