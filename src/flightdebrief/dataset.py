"""Build the SFT dataset (TRL conversational prompt/completion JSONL) from a flightlake lake.

Augmentation: every flight yields two examples - the full digest and a *silent* digest with
all non-INFO autopilot messages removed. Silent examples force the model to diagnose from
telemetry deviations instead of pattern-matching warning text (and mirror real life, where
many faults never raise a message).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flightlake import Lake

from .digest import build_digest, fit_fleet_baseline, save_baseline
from .schema import SYSTEM_PROMPT
from .targets import make_target


def split_of(fid: str, val: float = 0.15, test: float = 0.15) -> str:
    h = int(hashlib.md5(fid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def to_example(digest: dict, target_json: str | None) -> dict:
    ex = {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": digest["text"]}],
          "flight_id": digest["flight_id"], "silent": digest["silent"]}
    if target_json is not None:
        ex["completion"] = [{"role": "assistant", "content": target_json}]
    return ex


def build(lake_root: str, out_dir: str = "data/sft") -> dict:
    lake, out = Lake(lake_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat = lake.catalog()
    splits = {fid: split_of(fid) for fid in cat.flight_id}
    nominal_train = [f for f in cat.loc[cat.injected_faults == "none", "flight_id"] if splits[f] == "train"]
    baseline = fit_fleet_baseline(lake, nominal_train)
    save_baseline(baseline, out / "fleet_baseline.json")
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    counts = {s: 0 for s in files}
    for i, fid in enumerate(cat.flight_id):
        faults = lake.summary(fid)["meta"].get("faults", [])
        for silent in (False, True):
            d = build_digest(lake, fid, baseline, drop_fault_messages=silent)
            tgt = make_target(d, faults, seed=i * 2 + silent)
            ex = to_example(d, tgt.model_dump_json())
            ex["label"] = {"fault_type": tgt.fault_type, "onset_t_s": tgt.onset_t_s}
            files[splits[fid]].write(json.dumps(ex) + "\n")
            counts[splits[fid]] += 1
    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(counts))
    return counts
