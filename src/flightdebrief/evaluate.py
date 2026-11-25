"""Evaluation harness for debrief generators (fine-tuned LLM, API LLM, or baselines).

Metrics
  json_valid           output parses and validates against the Debrief schema
  fault_acc / macro_f1 fault classification (incl. 'none')
  subsystem_acc        routing correctness
  onset_mae_s          |predicted - true onset| for correctly-detected faults
  onset_within_30s     fraction of faults whose onset is within 30 s
  grounded_rate        fraction of numbers in evidence+summary that appear in the digest
  hallucination_rate   fraction of outputs containing >=1 ungrounded number
All metrics are reported separately for 'full' and 'silent' (no warning messages) digests.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .schema import parse_debrief

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
TIME = re.compile(r"t=(-?\d+(?:\.\d+)?)s?")


def digest_numbers(text: str) -> tuple[np.ndarray, np.ndarray]:
    nums = np.array([float(x) for x in NUM.findall(text)] or [np.nan])
    times = [float(x) for x in TIME.findall(text)]
    times += [float(a) for a, b in re.findall(r"\[(\d+)-(\d+)s\]", text)]
    times += [float(b) for a, b in re.findall(r"\[(\d+)-(\d+)s\]", text)]
    return nums, np.array(times or [np.nan])


def groundedness(answer_text: str, digest_text: str, time_tol: float = 30.0) -> tuple[int, int]:
    """(#grounded, #total) numbers. Times (t=..s) may be estimates near a digest time; other numbers
    must match a digest number within 1% (or 0.05 absolute)."""
    nums, times = digest_numbers(digest_text)
    ok = tot = 0
    t_spans = {m.span() for m in TIME.finditer(answer_text)}
    for m in TIME.finditer(answer_text):
        tot += 1
        ok += bool(np.nanmin(np.abs(times - float(m.group(1)))) <= time_tol)
    for m in NUM.finditer(answer_text):
        if any(a <= m.start() < b for a, b in t_spans):
            continue
        v = float(m.group(0))
        if v in (0, 1, 2, 3, 4) or abs(v) in (1.0,):  # small counts / ordinals ("motor 3", "3 checks")
            continue
        tot += 1
        ok += bool(np.nanmin(np.abs(nums - v)) <= max(0.05, 0.01 * abs(v)))
    return ok, tot


def evaluate(examples: list[dict], generate: Callable[[dict], str], name: str = "model",
             out_dir: str | None = None) -> dict:
    rows = []
    for ex in examples:
        text = generate(ex)
        obj, err = parse_debrief(text)
        lab = ex["label"]
        r = {"flight_id": ex["flight_id"], "silent": ex["silent"], "true": lab["fault_type"], "valid": obj is not None,
             "error": err, "raw": text[:2000]}
        if obj:
            r["pred"] = obj.fault_type
            r["subsystem_ok"] = obj.subsystem == json.loads(ex["completion"][0]["content"])["subsystem"] \
                if "completion" in ex else None
            if lab["onset_t_s"] is not None and obj.onset_t_s is not None and obj.fault_type == lab["fault_type"]:
                r["onset_err"] = abs(obj.onset_t_s - lab["onset_t_s"])
            g, t = groundedness(" ".join(obj.evidence) + " " + obj.summary, ex["prompt"][-1]["content"])
            r["grounded"], r["numbers"] = g, t
        else:
            r["pred"] = "invalid"
        rows.append(r)
    res = {"name": name}
    for subset, sel in (("all", lambda r: True), ("full", lambda r: not r["silent"]), ("silent", lambda r: r["silent"])):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        y, p = [r["true"] for r in rs], [r["pred"] for r in rs]
        on = [r["onset_err"] for r in rs if "onset_err" in r]
        g = sum(r.get("grounded", 0) for r in rs)
        t = sum(r.get("numbers", 0) for r in rs)
        res[subset] = {
            "n": len(rs), "json_valid": round(np.mean([r["valid"] for r in rs]), 3),
            "fault_acc": round(float(np.mean(np.array(y) == np.array(p))), 3),
            "macro_f1": round(float(f1_score(y, p, average="macro", labels=sorted(set(y)))), 3),
            "subsystem_acc": round(float(np.mean([r.get("subsystem_ok") or False for r in rs])), 3),
            "onset_mae_s": round(float(np.mean(on)), 1) if on else None,
            "onset_within_30s": round(float(np.mean(np.array(on) <= 30)), 3) if on else None,
            "grounded_rate": round(g / t, 3) if t else None,
            "hallucination_rate": round(float(np.mean([r.get("numbers", 0) > r.get("grounded", 0)
                                                       for r in rs if r["valid"]] or [0])), 3),
        }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, f"eval_{name}.json").write_text(json.dumps(res, indent=2))
        with open(Path(out_dir, f"predictions_{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r, default=str) + "\n" for r in rows)
    return res


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
