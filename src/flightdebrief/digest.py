"""Compress a full flight log into a compact, LLM-readable *digest*.

A 15-minute sortie is ~9k rows x 37 channels plus free-text messages - far too much to
paste into a prompt, and raw numbers are a poor LLM input anyway. The digest keeps what an
engineer would look at first:

  1. header (airframe, firmware, site, wind, duration)
  2. phase timeline (from flightlake)
  3. key metrics (STOL ground roll, energy, GNSS outage, vibration ...)
  4. *deviations*: channel segments that depart from the fleet-normal envelope for the same
     flight phase (robust z >= 4 for >= 3 s), fault-agnostic - nothing in here knows about
     fault classes
  5. autopilot messages (non-INFO, plus a few INFO for context)

Every number in the digest has a stable textual form, which is what lets the evaluator
check that a model's answer is *grounded* (numbers in the answer must appear in the digest).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flightlake import Lake

DEV_CHANNELS = [
    "airspeed_mps", "groundspeed_mps", "vz_mps", "roll_deg", "pitch_deg", "q_dps", "r_dps", "az_mps2",
    "vibe_mps2", "throttle", "elevator_cmd_deg", "elevator_pos_deg", "motor1_rpm", "motor2_rpm", "motor3_rpm",
    "motor4_rpm", "batt_v", "batt_a", "batt_temp_c", "gen_kw", "gnss_nsat", "gnss_hdop", "ekf_pos_innov_m",
    "ekf_vel_innov_mps", "ekf_hacc_m",
]
DERIVED = {"elevator_err_deg": ("elevator_cmd_deg", "elevator_pos_deg"),
           "airspeed_minus_gs": ("airspeed_mps", "groundspeed_mps")}
AIR_PHASES = ["climb", "cruise", "descent", "approach"]
KEY_METRICS = [
    "duration_s", "takeoff_ground_roll_m", "landing_ground_roll_m", "max_alt_rel_m", "mean_climb_rate_mps",
    "cruise_airspeed_mean_mps", "energy_batt_kwh", "energy_gen_kwh", "min_batt_v", "max_batt_temp_c",
    "soc_used", "max_vibe_mps2", "gnss_outage_s", "max_ekf_pos_innov_m", "elevator_tracking_rmse_deg",
    "max_motor_rpm_imbalance",
]


def _frame(tel: pd.DataFrame) -> pd.DataFrame:
    f = tel[["t_s", "phase_est", *DEV_CHANNELS]].copy()
    for name, (a, b) in DERIVED.items():
        f[name] = tel[a] - tel[b]
    return f


def fit_fleet_baseline(lake: Lake, flight_ids: list[str]) -> dict:
    """Per-phase robust envelope (median, MAD) of every deviation channel from nominal flights."""
    frames = [_frame(lake.telemetry(f)) for f in flight_ids]
    df = pd.concat(frames)
    chans = DEV_CHANNELS + list(DERIVED)
    out = {}
    for ph, g in df.groupby("phase_est"):
        med = g[chans].median()
        mad = (g[chans] - med).abs().median() * 1.4826
        p99 = (g[chans] - med).abs().quantile(0.99)
        # floor the scale so near-constant channels (nsat, gen) don't explode
        scale = np.maximum(mad, p99 / 6).clip(lower=1e-3)
        out[ph] = {"median": med.round(4).to_dict(), "scale": scale.round(4).to_dict()}
    return out


def deviations(tel: pd.DataFrame, baseline: dict, z_thr: float = 4.0, min_s: float = 3.0, top: int = 10,
               rate_hz: float = 10.0) -> list[dict]:
    f = _frame(tel)
    chans = DEV_CHANNELS + list(DERIVED)
    smooth = f[chans].rolling(int(3 * rate_hz), center=True, min_periods=5).median()
    n = len(f)
    med_a, sc_a = np.full((n, len(chans)), np.nan), np.full((n, len(chans)), np.nan)
    ph_arr = f.phase_est.to_numpy()
    for ph, b in baseline.items():
        if ph not in AIR_PHASES + ["takeoff_roll"]:
            continue  # ground handling is noisy by nature -> not scored
        idx = ph_arr == ph
        med_a[idx] = [b["median"][c] for c in chans]
        sc_a[idx] = [b["scale"][c] for c in chans]
    med = pd.DataFrame(med_a, columns=chans, index=f.index)
    Z = (smooth - med) / pd.DataFrame(sc_a, columns=chans, index=f.index)
    out = []
    for c in chans:
        z = Z[c].to_numpy()
        hit = np.abs(np.nan_to_num(z)) >= z_thr
        # also flag channels that *disappear* while airborne (e.g. GNSS innovations go NaN)
        if not hit.any():
            continue
        starts = np.flatnonzero(np.r_[hit[0], hit[1:] & ~hit[:-1]])
        ends = np.flatnonzero(np.r_[hit[:-1] & ~hit[1:], hit[-1]])
        for s, e in zip(starts, ends):
            if (e - s + 1) / rate_hz < min_s:
                continue
            seg = slice(s, e + 1)
            k = s + int(np.nanargmax(np.abs(z[seg])))
            out.append({
                "channel": c, "t_start": round(float(f.t_s.iloc[s]), 1), "t_end": round(float(f.t_s.iloc[e]), 1),
                "phase": str(f.phase_est.iloc[s]), "peak_z": round(float(z[k]), 1),
                "value": round(float(smooth[c].iloc[k]), 2), "normal": round(float(med[c].iloc[k]), 2),
            })
    # merge same-channel segments separated by < 10 s, then rank by severity x duration
    out.sort(key=lambda d: (d["channel"], d["t_start"]))
    merged = []
    for d in out:
        if merged and merged[-1]["channel"] == d["channel"] and d["t_start"] - merged[-1]["t_end"] < 10:
            m = merged[-1]
            m["t_end"] = d["t_end"]
            if abs(d["peak_z"]) > abs(m["peak_z"]):
                m.update(peak_z=d["peak_z"], value=d["value"], normal=d["normal"])
        else:
            merged.append(dict(d))
    merged.sort(key=lambda d: -abs(d["peak_z"]) * min(d["t_end"] - d["t_start"] + 1, 120) ** 0.5)
    return sorted(merged[:top], key=lambda d: d["t_start"])


def build_digest(lake: Lake, flight_id: str, baseline: dict, drop_fault_messages: bool = False,
                 max_msgs: int = 14) -> dict:
    s = lake.summary(flight_id)
    tel = lake.telemetry(flight_id)
    meta, m = s["meta"], s["metrics"]
    lines = [f"FLIGHT {flight_id} | airframe {meta.get('airframe')} | fw {meta.get('firmware')} | "
             f"site {meta.get('site')} | wind {meta.get('wind_speed', float('nan')):.1f} m/s "
             f"from {meta.get('wind_dir_deg', float('nan')):.0f} deg"]
    lines.append("PHASES: " + " > ".join(f"{p['phase']}[{p['t_start']:.0f}-{p['t_end']:.0f}s]" for p in s["phases"]))
    lines.append("METRICS: " + ", ".join(f"{k}={m[k]}" for k in KEY_METRICS if m.get(k) is not None))
    devs = deviations(tel, baseline)
    lines.append("DEVIATIONS (vs fleet-normal for same phase; z=robust z-score):")
    if devs:
        for d in devs:
            lines.append(f"- {d['channel']} t={d['t_start']:.1f}-{d['t_end']:.1f}s ({d['phase']}): "
                         f"{d['value']} vs normal {d['normal']} (z={d['peak_z']})")
    else:
        lines.append("- none")
    msgs = lake.messages(flight_id)
    msgs = msgs[msgs.t_s >= 0]
    imp = msgs[msgs.level.str.upper() != "INFO"]
    if drop_fault_messages:  # augmentation: force the model to reason from telemetry, not log text
        imp = imp.iloc[0:0]
    ctx = msgs[msgs.level.str.upper() == "INFO"].iloc[: max(0, max_msgs - len(imp))]
    sel = pd.concat([imp, ctx]).sort_values("t_s").head(max_msgs)
    lines.append("MESSAGES:")
    lines += [f"- t={r.t_s:.1f}s {r.level}: {r.text}" for r in sel.itertuples()] or ["- none"]
    return {"flight_id": flight_id, "text": "\n".join(lines), "deviations": devs, "phases": s["phases"],
            "metrics": m, "silent": drop_fault_messages}


def save_baseline(b: dict, path: str | Path):
    Path(path).write_text(json.dumps(b))


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
