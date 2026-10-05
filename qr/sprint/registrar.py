#!/usr/bin/env python3
"""Proposal intake: agent memos -> validated specs -> protocol registry.

The research lead emits a spec inside a ```json fence in its memo. The
orchestrator drops it in proposals/. This registrar is the deterministic
gate the charter requires: schema+power validation via factory.validate_spec,
duplicate-name refusal, closed-door key-word screen, then registration in
the project registry (same file as the 248 daily-frequency trials — one
honest denominator).  Rejections go back to proposals/rejected/ with reasons.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from qr.sprint.factory import validate_spec           # noqa: E402
from qr.research.protocol import register, Experiment, trial_count  # noqa: E402

CLOSED_DOOR_TERMS = ["taker_imbalance", "taker imbalance",     # sprint21-23: t=-9..-49
                     "daily rebalance", "overnight gap", "turn-of-month",
                     "insider", "13f", "short interest", "premarket volume"]


def process() -> list[str]:
    done = []
    for f in sorted((HERE / "proposals").glob("*.json")):
        spec = json.loads(f.read_text())
        errs = validate_spec(spec)
        name = spec.get("name", f.stem)
        if (HERE / "specs" / f"{name}.json").exists():
            errs.append("duplicate spec name (freeze rule: no re-registration)")
        blob = json.dumps(spec).lower()
        hits = [t for t in CLOSED_DOOR_TERMS if t in blob]
        if hits:
            errs.append(f"closed-door terms {hits}: registry says these "
                        "families are dead at daily frequency; a NEW "
                        "mechanism must be argued explicitly")
        if errs:
            rej = HERE / "proposals" / "rejected"
            rej.mkdir(exist_ok=True)
            (rej / f.name).write_text(json.dumps(
                {"spec": spec, "errors": errs}, indent=1))
            f.unlink()
            done.append(f"REJECTED {name}: {errs}")
            continue
        (HERE / "specs" / f"{name}.json").write_text(json.dumps(spec, indent=1))
        register(Experiment(
            name=name, family="sprint-intraday",
            hypothesis=spec["hypothesis"],
            features=[spec["signal"]],
            split="research 2024-01..2026-04 / validation 2026-05..07; 2026-08+ SEALED",
            universe=",".join(spec["symbols"]),
            rebalance_days=0,
            holding_days=round(spec["hold_minutes"] / 1440, 4),
            n_observations=0, metrics={}, alpha=None,
            outcome="PENDING", failure_reason="registered - awaiting factory run"))
        f.unlink()
        done.append(f"REGISTERED {name} (trial #{trial_count()})")
    return done


if __name__ == "__main__":
    for line in process():
        print(line)
