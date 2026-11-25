import json

import pytest

from flightdebrief.evaluate import evaluate, groundedness
from flightdebrief.schema import parse_debrief
from flightdebrief.targets import make_target

DIGEST = """FLIGHT X | airframe A
PHASES: taxi[0-20s] > takeoff_roll[20-30s] > climb[30-150s] > cruise[150-600s]
METRICS: takeoff_ground_roll_m=182.4, energy_batt_kwh=3.1
DEVIATIONS (vs fleet-normal for same phase; z=robust z-score):
- motor3_rpm t=301.2-600.0s (cruise): 0.0 vs normal 2312.5 (z=-41.0)
- throttle t=302.0-600.0s (cruise): 0.41 vs normal 0.27 (z=6.1)
MESSAGES:
- t=301.8s CRITICAL: ESC 3: no telemetry / motor stopped"""


def _digest():
    devs = [{"channel": "motor3_rpm", "t_start": 301.2, "t_end": 600.0, "phase": "cruise", "peak_z": -41.0,
             "value": 0.0, "normal": 2312.5},
            {"channel": "throttle", "t_start": 302.0, "t_end": 600.0, "phase": "cruise", "peak_z": 6.1,
             "value": 0.41, "normal": 0.27}]
    return {"flight_id": "X", "text": DIGEST, "deviations": devs, "silent": False,
            "phases": [{"phase": "cruise", "t_start": 150.0, "t_end": 600.0}],
            "metrics": {"takeoff_ground_roll_m": 182.4, "energy_batt_kwh": 3.1}}


def test_target_is_grounded_and_valid():
    t = make_target(_digest(), [{"type": "motor_failure", "t_start": 300.0}])
    assert t.fault_type == "motor_failure" and t.severity == "critical" and "motor 3" in t.summary
    g, n = groundedness(" ".join(t.evidence) + " " + t.summary, DIGEST)
    assert n > 0 and g == n
    nominal = make_target(_digest(), [])
    assert nominal.fault_type == "none"


def test_groundedness_catches_invented_numbers():
    g, n = groundedness("motor3_rpm fell to 0.0 vs normal 2900.0 at t=301.2s", DIGEST)
    assert n == 2 and g == 1  # 2900.0 is not in the digest; t is near a digest time


def test_parse_debrief_lenient_json():
    txt = 'Sure! {"fault_type": "none", "subsystem": "none", "severity": "none", "summary": "ok", "evidence": [],}'
    obj, err = parse_debrief(txt)
    assert obj is not None and err is None
    assert parse_debrief("no json here")[1] == "no_json"
    assert parse_debrief('{"fault_type": "alien_attack", "subsystem": "none", "severity": "none", "summary": ""}')[0] is None


def test_evaluate_harness():
    t = make_target(_digest(), [{"type": "motor_failure", "t_start": 300.0}])
    ex = {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": DIGEST}],
          "completion": [{"role": "assistant", "content": t.model_dump_json()}], "flight_id": "X", "silent": False,
          "label": {"fault_type": "motor_failure", "onset_t_s": 300.0}}
    r = evaluate([ex], lambda e: e["completion"][0]["content"], "oracle")
    assert r["all"]["fault_acc"] == 1.0 and r["all"]["grounded_rate"] == 1.0
    r = evaluate([ex], lambda e: "garbage", "bad")
    assert r["all"]["json_valid"] == 0.0


@pytest.mark.slow
def test_sft_smoke(tmp_path, monkeypatch):
    """Full TRL+PEFT path on a tiny random model (no downloads)."""
    pytest.importorskip("trl")
    from flightdebrief.train_sft import main

    t = make_target(_digest(), [{"type": "motor_failure", "t_start": 300.0}])
    from flightdebrief.dataset import to_example

    ex = to_example(_digest(), t.model_dump_json())
    d = tmp_path / "data" / "sft"
    d.mkdir(parents=True)
    for s in ("train", "val"):
        (d / f"{s}.jsonl").write_text("\n".join(json.dumps(ex) for _ in range(8)))
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(f"base_model: tiny\ntrain_file: {d}/train.jsonl\nval_file: {d}/val.jsonl\n"
                   f"output_dir: {tmp_path}/out\nmax_steps: 2\nbatch_size: 2\ngrad_accum: 1\nmax_seq_len: 1024\n"
                   "lora_r: 4\nlora_alpha: 8\ngradient_checkpointing: false\neval_steps: 2\nlogging_steps: 1\n")
    main(["--config", str(cfg), "--smoke"])
    assert (tmp_path / "out" / "final_adapter" / "adapter_config.json").exists()
