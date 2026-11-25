"""Ground-truth debrief targets, built *only* from facts present in the digest.

Labels (fault type, onset) come from the flight-test card (injected faults). Every number in
the evidence and summary is copied from the digest text, so the training targets themselves
are 100% grounded - the model is never taught to invent numbers.
"""
from __future__ import annotations

import random

from .schema import Debrief

SUBSYSTEM = {"none": "none", "motor_failure": "propulsion", "battery_degradation": "power",
             "generator_dropout": "power", "gnss_jamming": "navigation", "gnss_spoofing": "navigation",
             "imu_vibration": "sensors", "pitot_blockage": "sensors", "elevator_stuck": "flight_controls"}

RELEVANT = {
    "motor_failure": ["motor1_rpm", "motor2_rpm", "motor3_rpm", "motor4_rpm", "throttle", "r_dps", "roll_deg",
                      "batt_a"],
    "battery_degradation": ["batt_v", "batt_temp_c", "batt_a"],
    "generator_dropout": ["gen_kw", "batt_a", "batt_v", "batt_temp_c"],
    "gnss_jamming": ["gnss_nsat", "gnss_hdop", "ekf_hacc_m"],
    "gnss_spoofing": ["ekf_pos_innov_m", "ekf_vel_innov_mps", "gnss_nsat"],
    "imu_vibration": ["vibe_mps2", "az_mps2", "q_dps"],
    "pitot_blockage": ["airspeed_mps", "airspeed_minus_gs", "throttle", "vz_mps"],
    "elevator_stuck": ["elevator_err_deg", "elevator_cmd_deg", "elevator_pos_deg", "q_dps", "vz_mps"],
}

ACTIONS = {
    "none": ["Archive log as nominal reference", "No maintenance action required"],
    "motor_failure": ["Inspect ESC and motor on the affected nacelle; pull ESC fault log",
                      "Review one-motor-inoperative climb margin against the measured data",
                      "Ground the aircraft until propulsion unit passes run-up test"],
    "battery_degradation": ["Run cell-balance / impedance check on the pack",
                            "Compare per-cell voltages for a weak cell group",
                            "Lower depth-of-discharge limit until pack is cleared"],
    "generator_dropout": ["Pull range-extender ECU fault codes", "Inspect generator coupling and fuel supply",
                          "Verify battery-only endurance reserve logic engaged correctly"],
    "gnss_jamming": ["Check NOTAMs / interference reports for the test area",
                     "Review dead-reckoning drift and EKF fallback during the outage",
                     "Assess need for CRPA antenna or alternative navigation aiding"],
    "gnss_spoofing": ["Treat as suspected spoofing: report to range safety",
                      "Review EKF innovation gating thresholds and GPS rejection timing",
                      "Cross-check position against independent source (vision/radio aids)"],
    "imu_vibration": ["Check propeller balance and motor mounts", "Inspect IMU isolation mounts",
                      "Re-run vibration survey at representative throttle settings"],
    "pitot_blockage": ["Inspect pitot-static lines and drain holes for blockage or icing",
                       "Verify pitot heat operation", "Confirm synthetic-airspeed fallback timing"],
    "elevator_stuck": ["Ground aircraft: inspect elevator servo, linkage and hinge for binding",
                       "Pull servo current/temperature logs", "Review fallback pitch-control logic"],
}

TEMPLATES = {
    "motor_failure": ["{who} stopped at t={onset}s during {phase}; the remaining motors and throttle compensated "
                      "and the aircraft continued. {ev}",
                      "Loss of one DEP motor ({who}) at t={onset}s in {phase}. {ev} Asymmetric thrust was "
                      "absorbed by the controller."],
    "battery_degradation": ["Battery pack shows degraded behaviour from t={onset}s ({phase}): {ev} Consistent "
                            "with increased internal resistance / capacity fade.",
                            "Pack voltage sag beyond fleet norms starting t={onset}s in {phase}. {ev}"],
    "generator_dropout": ["Range-extender generator dropped out at t={onset}s during {phase}; the battery "
                          "carried the full load. {ev}",
                          "Generator offline from t={onset}s ({phase}). {ev} Flight completed on battery."],
    "gnss_jamming": ["GNSS denial (likely jamming) at t={onset}s in {phase}: satellites lost and the estimator "
                     "fell back to dead reckoning. {ev}",
                     "Loss of GNSS fix starting t={onset}s ({phase}). {ev}"],
    "gnss_spoofing": ["Suspected GNSS spoofing from t={onset}s ({phase}): satellites looked healthy while EKF "
                      "position innovations grew. {ev}",
                      "GNSS position drifted away from inertial solution starting t={onset}s in {phase}. {ev}"],
    "imu_vibration": ["Elevated airframe vibration from t={onset}s during {phase}. {ev} Likely propeller "
                      "imbalance or loose mount.",
                      "IMU vibration above fleet envelope starting t={onset}s ({phase}). {ev}"],
    "pitot_blockage": ["Airspeed indication failed from t={onset}s in {phase} while ground speed stayed "
                       "normal - consistent with pitot blockage. {ev}",
                       "Pitot-static fault at t={onset}s ({phase}): indicated airspeed decayed. {ev}"],
    "elevator_stuck": ["Elevator stopped following commands from t={onset}s during {phase} (actuator jam). {ev}",
                       "Elevator actuator tracking failure at t={onset}s in {phase}. {ev} Pitch control "
                       "degraded to thrust-based fallback."],
}

SEVERITY = {"motor_failure": "critical", "elevator_stuck": "critical", "gnss_spoofing": "major",
            "gnss_jamming": "major", "pitot_blockage": "major", "generator_dropout": "major"}


def _ev_str(d: dict) -> str:
    return f"{d['channel']} {d['value']} vs normal {d['normal']} at t={d['t_start']:.1f}s"


def _phase_at(phases: list[dict], t: float) -> str:
    for p in phases:
        if p["t_start"] <= t <= p["t_end"] + 0.1:
            return p["phase"]
    return phases[-1]["phase"] if phases else "unknown"


def make_target(digest: dict, faults: list[dict], seed: int = 0) -> Debrief:
    rnd = random.Random(seed)
    m, devs = digest["metrics"], digest["deviations"]
    if not faults:
        parts = []
        for k, label in [("takeoff_ground_roll_m", "takeoff ground roll {} m"),
                         ("cruise_airspeed_mean_mps", "cruise airspeed {} m/s"),
                         ("energy_batt_kwh", "battery energy {} kWh")]:
            if m.get(k) is not None:
                parts.append(label.format(m[k]))
        extra = (" Minor excursions within tolerance: " + "; ".join(_ev_str(d) for d in devs[:2]) + ".") if devs \
            else " No deviations from the fleet envelope."
        return Debrief(fault_type="none", subsystem="none", onset_t_s=None, phase_at_onset=None, severity="none",
                       evidence=[], summary="Nominal sortie: " + ", ".join(parts) + "." + extra,
                       recommended_actions=ACTIONS["none"])
    f = faults[0]
    ft, onset = f["type"], round(float(f["t_start"]))
    rel = [d for d in devs if d["channel"] in RELEVANT[ft] and d["t_end"] >= onset - 5]
    rel.sort(key=lambda d: (d["t_start"], -abs(d["peak_z"])))
    evidence = [_ev_str(d) for d in rel[:3]]
    phase = _phase_at(digest["phases"], onset)
    who = "one motor"
    if ft == "motor_failure":
        low = [d for d in rel if d["channel"].startswith("motor") and d["peak_z"] < 0]
        if low:
            who = f"motor {low[0]['channel'][5]}"
    ev = ("Evidence: " + "; ".join(evidence) + ".") if evidence else "Telemetry signature is weak in this log."
    summary = rnd.choice(TEMPLATES[ft]).format(onset=onset, phase=phase, ev=ev, who=who)
    summary = summary[0].upper() + summary[1:]
    sev = SEVERITY.get(ft)
    if sev is None:  # magnitude-dependent severities
        big = any(abs(d["peak_z"]) >= 10 for d in rel)
        sev = "major" if big else "minor"
    return Debrief(fault_type=ft, subsystem=SUBSYSTEM[ft], onset_t_s=float(onset), phase_at_onset=phase,
                   severity=sev, evidence=evidence, summary=summary, recommended_actions=ACTIONS[ft])
