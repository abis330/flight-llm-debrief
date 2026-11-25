"""FastAPI service: GET /debrief/{flight_id} -> digest + structured debrief.

Backends (env DEBRIEF_BACKEND):
  hf        local transformers model (DEBRIEF_MODEL, DEBRIEF_ADAPTER)
  openai    OpenAI-compatible server, e.g. vLLM with the LoRA adapter (DEBRIEF_URL, DEBRIEF_MODEL)
  anthropic Claude (ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from .infer import AnthropicGenerator, HFGenerator, OpenAICompatGenerator, debrief_flight

app = FastAPI(title="flight-llm-debrief")


@lru_cache(maxsize=1)
def generator():
    b = os.environ.get("DEBRIEF_BACKEND", "openai")
    if b == "hf":
        return HFGenerator(os.environ["DEBRIEF_MODEL"], os.environ.get("DEBRIEF_ADAPTER"))
    if b == "anthropic":
        return AnthropicGenerator()
    return OpenAICompatGenerator(os.environ.get("DEBRIEF_URL", "http://localhost:8000/v1"),
                                 os.environ.get("DEBRIEF_MODEL", "debrief"))


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.environ.get("DEBRIEF_BACKEND", "openai")}


@app.get("/debrief/{flight_id}")
def debrief(flight_id: str):
    return debrief_flight(os.environ.get("FLIGHTLAKE_ROOT", "data/lake"), flight_id, generator(),
                          os.environ.get("FLEET_BASELINE", "data/sft/fleet_baseline.json"))
