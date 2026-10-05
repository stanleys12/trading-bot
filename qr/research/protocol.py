"""
Research protocol: splits, registry, and the seal on the final test set.
========================================================================

This module exists because discipline that lives in a person's memory is not
discipline. Every rule below is enforced in code and every violation raises.

THE SPLITS

    RESEARCH    2010-01-01 .. 2017-12-31   explore freely
    VALIDATION  2018-01-01 .. 2022-12-31   choose ONE candidate
    TEST        2023-01-01 .. 2026-08-11   SEALED

The boundary at 2018 is not arbitrary. Our point-in-time universe is
reconstructed from archived exchange directories that begin 2018-11, so
survivorship is MEASURED from 2018 onward and merely feared before it.
Discovery can tolerate a biased sample; validation cannot. So the biased era
is used only for generating hypotheses, and every number that decides anything
comes from the era where we know what the universe was.

THE SEAL

`open_test_set()` refuses unless a candidate has been registered, frozen, and
has passed validation. It records the access permanently. A frozen candidate
cannot be modified -- `freeze()` stores a hash of the full specification, and
re-registering a changed spec under the same name raises.

THE REGISTRY

Every hypothesis ever scored, with its result, in a machine-readable table.
This is the honest denominator for multiple-testing corrections: not the
number of tests in the current script, but the number this project has run
in total. It only grows.

POWER, NOT JUST PRE-REGISTRATION (adopted 2026-08-13, after exp14)

Pre-registered criteria are necessary and not sufficient. exp14 passed five
criteria fixed honestly in advance, and the pass carried almost no
information: two were near-mechanical for the strategy class, one was
implied by the era, two were unpowered point comparisons. Every future
pre-registration must therefore state, BEFORE scoring, why its criteria
could plausibly come out FALSE under the null — an explicit power argument
(what magnitude of no-edge luck would still pass, and how often). A
criterion that cannot fail is the 86%-win-rate defect in statistical form.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE.parent / "reports" / "experiment_registry.jsonl"
SEAL_PATH = HERE.parent / "reports" / "test_set_access.jsonl"
FROZEN_PATH = HERE.parent / "reports" / "frozen_candidates.json"

RESEARCH = ("2010-01-01", "2017-12-31")
VALIDATION = ("2018-01-01", "2022-12-31")
TEST = ("2023-01-01", "2026-08-11")


class ProtocolViolation(RuntimeError):
    """Raised when the research protocol would be broken."""


def slice_split(panel: pd.DataFrame, split: str,
                warmup_days: int = 400) -> pd.DataFrame:
    """
    Return the panel restricted to a split, with warm-up history before it.

    Warm-up is needed so signals have a lookback at the split's first date.
    It is history, not outcome: the engine only TRADES from the split start.
    """
    lo, hi = {"research": RESEARCH, "validation": VALIDATION, "test": TEST}[split]
    lo_ts = pd.Timestamp(lo) - pd.Timedelta(days=warmup_days)
    return panel[(panel["date"] >= lo_ts) & (panel["date"] <= pd.Timestamp(hi))]


def split_start(split: str) -> pd.Timestamp:
    return pd.Timestamp({"research": RESEARCH, "validation": VALIDATION,
                         "test": TEST}[split][0])


# ------------------------------------------------------------------ registry
@dataclass
class Experiment:
    name: str
    family: str                     # price_volume | fundamentals | events | ...
    hypothesis: str                 # why it might work, in words
    features: list[str]
    split: str                      # research | validation | test
    universe: str
    rebalance_days: int
    holding_days: int
    n_observations: int
    metrics: dict                   # sharpe, cagr, maxdd, turnover, ...
    alpha: dict = field(default_factory=dict)   # vs SPY / SPY+IWM
    outcome: str = ""               # PASS | FAIL | HOLD
    failure_reason: str = ""        # REQUIRED when outcome == FAIL
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.outcome == "FAIL" and not self.failure_reason:
            raise ProtocolViolation(
                f"{self.name}: a failed experiment must record WHY it failed. "
                "Silent failures are how a research programme forgets that it "
                "already rejected an idea.")


def register(exp: Experiment) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a") as f:
        f.write(json.dumps(asdict(exp), default=str) + "\n")


def load_registry() -> pd.DataFrame:
    if not REGISTRY_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in REGISTRY_PATH.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def trial_count() -> int:
    """The honest K for multiple-testing corrections."""
    reg = load_registry()
    prior = 192          # everything scored before the registry existed
    return prior + len(reg)


# -------------------------------------------------------------------- freeze
def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze(name: str, spec: dict) -> str:
    """
    Lock a candidate's full specification. After this, the spec cannot change
    without a new name -- which is what stops 'just one more tweak' after
    seeing validation results.
    """
    frozen = json.loads(FROZEN_PATH.read_text()) if FROZEN_PATH.exists() else {}
    h = _spec_hash(spec)
    if name in frozen and frozen[name]["hash"] != h:
        raise ProtocolViolation(
            f"'{name}' is already frozen with a different specification "
            f"({frozen[name]['hash']} vs {h}). Modifying a frozen candidate "
            "after seeing results is the definition of overfitting. Register "
            "it under a new name and pay the multiple-testing cost.")
    frozen[name] = {"hash": h, "spec": spec,
                    "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2, default=str))
    logger.info("frozen '%s' @ %s", name, h)
    return h


# ---------------------------------------------------------------------- seal
def open_test_set(panel: pd.DataFrame, candidate: str, spec: dict,
                  validation_metrics: dict) -> pd.DataFrame:
    """
    The ONLY sanctioned path to the test set. Refuses unless:
      - the candidate is frozen with exactly this spec
      - it cleared validation on the criteria that matter
      - and every access is recorded permanently
    """
    frozen = json.loads(FROZEN_PATH.read_text()) if FROZEN_PATH.exists() else {}
    if candidate not in frozen:
        raise ProtocolViolation(
            f"'{candidate}' was never frozen. Freeze the specification before "
            "the test set is touched, or the test result means nothing.")
    if frozen[candidate]["hash"] != _spec_hash(spec):
        raise ProtocolViolation(
            f"'{candidate}' spec differs from the frozen version. The test set "
            "may only score the exact thing that passed validation.")

    sharpe = validation_metrics.get("sharpe")
    alpha_t = validation_metrics.get("alpha_t")
    if (sharpe or 0) < 0.8 or (alpha_t or 0) < 2.0:
        raise ProtocolViolation(
            f"'{candidate}' has not earned a test-set look "
            f"(validation Sharpe={sharpe}, alpha_t={alpha_t}). The test set is "
            "not for triage.")

    prior = [json.loads(l) for l in SEAL_PATH.read_text().splitlines()] \
        if SEAL_PATH.exists() else []
    SEAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEAL_PATH.open("a") as f:
        f.write(json.dumps({
            "candidate": candidate, "hash": frozen[candidate]["hash"],
            "accessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "access_number": len(prior) + 1,
            "validation_metrics": validation_metrics}, default=str) + "\n")

    logger.warning("TEST SET OPENED for '%s' -- access #%d. This is permanent.",
                   candidate, len(prior) + 1)
    return slice_split(panel, "test")


def summary() -> pd.DataFrame:
    """What has been tried, by family and outcome."""
    reg = load_registry()
    if reg.empty:
        return pd.DataFrame()
    return (reg.groupby(["family", "outcome"]).size()
            .rename("n").reset_index()
            .sort_values(["family", "outcome"]))
