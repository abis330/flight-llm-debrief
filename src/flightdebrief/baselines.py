"""Non-LLM baselines the fine-tuned model has to beat."""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .digest import DERIVED, DEV_CHANNELS
from .targets import ACTIONS, RELEVANT, SUBSYSTEM

MSG_RULES = [
    (r"ESC \d+|motor stopped", "motor_failure"), (r"GNSS: fix lost", "gnss_jamming"),
    (r"GPS position test ratio", "gnss_spoofing"), (r"Generator offline", "generator_dropout"),
    (r"Airspeed sensor failure", "pitot_blockage"), (r"Elevator actuator", "elevator_stuck"),
    (r"vibration", "imu_vibration"), (r"Battery voltage low|Battery temperature", "battery_degradation"),
]
DEV_RE = re.compile(r"^- (\w+) t=([\d.]+)-([\d.]+)s \((\w+)\): (-?[\d.]+) vs normal (-?[\d.]+) \(z=(-?[\d.]+)\)",
                    re.M)
CHANS = DEV_CHANNELS + list(DERIVED)


def _out(ft, onset, evidence=(), summary=""):
    return json.dumps({"fault_type": ft, "subsystem": SUBSYSTEM[ft], "onset_t_s": onset, "phase_at_onset": None,
                       "severity": "none" if ft == "none" else "major", "evidence": list(evidence),
                       "summary": summary or ("Nominal sortie." if ft == "none" else f"Suspected {ft}."),
                       "recommended_actions": ACTIONS[ft]})


def majority(ex: dict) -> str:
    return _out("none", None)


def message_rules(ex: dict) -> str:
    text = ex["prompt"][-1]["content"]
    for line in text.split("MESSAGES:")[-1].splitlines():
        for pat, ft in MSG_RULES:
            if re.search(pat, line, re.I):
                t = re.search(r"t=([\d.]+)s", line)
                return _out(ft, float(t.group(1)) if t else None, [line.strip("- ")])
    return _out("none", None)


def featurize(text: str) -> np.ndarray:
    v = np.zeros(2 * len(CHANS) + 1)
    devs = DEV_RE.findall(text)
    for ch, t0, _t1, _ph, _val, _nrm, z in devs:
        if ch in CHANS:
            i = CHANS.index(ch)
            v[i] = max(v[i], abs(float(z)))
            v[len(CHANS) + i] = np.sign(float(z))
    v[-1] = len(devs)
    return v


class GBMBaseline:
    """Gradient boosting on parsed deviation features (no access to messages)."""

    def fit(self, train: list[dict]):
        X = np.stack([featurize(e["prompt"][-1]["content"]) for e in train])
        y = [e["label"]["fault_type"] for e in train]
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, y)
        return self

    def __call__(self, ex: dict) -> str:
        text = ex["prompt"][-1]["content"]
        ft = str(self.clf.predict(featurize(text)[None])[0])
        onset = None
        if ft != "none":
            rel = [d for d in DEV_RE.findall(text) if d[0] in RELEVANT[ft]]
            if rel:
                onset = min(float(d[1]) for d in rel)
        return _out(ft, onset)


def oracle(ex: dict) -> str:
    """The training target itself - sanity check for the evaluator (must score ~1.0 grounded)."""
    return ex["completion"][0]["content"]
