"""Structured output contract for the debrief model."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

FaultType = Literal["none", "motor_failure", "gnss_jamming", "gnss_spoofing", "battery_degradation",
                    "imu_vibration", "pitot_blockage", "elevator_stuck", "generator_dropout"]
Subsystem = Literal["none", "propulsion", "power", "navigation", "sensors", "flight_controls"]


class Debrief(BaseModel):
    fault_type: FaultType
    subsystem: Subsystem
    onset_t_s: float | None = Field(None, description="estimated fault onset, seconds since log start")
    phase_at_onset: str | None = None
    severity: Literal["none", "minor", "major", "critical"]
    evidence: list[str] = Field(default_factory=list, max_length=6)
    summary: str
    recommended_actions: list[str] = Field(default_factory=list, max_length=5)


SYSTEM_PROMPT = (
    "You are a flight-test data analyst for a hybrid-electric STOL aircraft. You receive a flight-log "
    "digest (phases, metrics, deviations from the fleet-normal envelope, autopilot messages). Diagnose "
    "the most likely fault (or 'none'), estimate its onset time, and write a concise debrief. Use ONLY "
    "numbers that appear in the digest. Reply with a single JSON object with keys: fault_type, subsystem, "
    "onset_t_s, phase_at_onset, severity, evidence, summary, recommended_actions."
)


def parse_debrief(text: str) -> tuple[Debrief | None, str | None]:
    """Lenient JSON extraction + strict schema validation. Returns (obj, error)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json"
    raw = m.group(0)
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):  # tolerate trailing commas
        try:
            return Debrief.model_validate(json.loads(candidate)), None
        except json.JSONDecodeError:
            continue
        except ValidationError as e:
            return None, f"schema: {e.errors()[0]['loc']}"
    return None, "invalid_json"
