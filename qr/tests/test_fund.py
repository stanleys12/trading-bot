"""Tests for the fund orchestrator's handling of agent output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fund import orchestrator


def _memo(spec):
    return f"memo text\n```json\n{json.dumps(spec)}\n```\n"


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "qr/sprint/proposals").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "feed", lambda msg: None)
    return tmp_path


def test_spec_is_queued_under_its_name(root):
    spec = {"name": "sprint47_breakout_spy", "signal": "breakout"}
    assert orchestrator.queue_proposal(_memo(spec)) == "sprint47_breakout_spy"
    out = root / "qr/sprint/proposals/sprint47_breakout_spy.json"
    assert json.loads(out.read_text()) == spec


@pytest.mark.parametrize("name", ["../../escaped", "a/b", "", ".hidden", 7])
def test_unsafe_spec_name_is_refused(root, name):
    memo = _memo({"name": name, "signal": "breakout"})
    assert orchestrator.queue_proposal(memo) is None
    assert not (root / "escaped.json").exists()
    assert list((root / "qr/sprint/proposals").iterdir()) == []
