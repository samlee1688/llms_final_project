#!/usr/bin/env python3
"""Validate EAJP scorecard schema in *_details.json outputs.

Usage examples:
  python -m experiments.validate_eajp_schema --path results/smoke_eajp_scorecard_check
  python -m experiments.validate_eajp_schema --path results/run/foo/openai_x/eajp_details.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_SCORECARD_ENTRY_FIELDS = {
    "agent_id",
    "document_id",
    "credibility_tier",
    "agent_claim",
    "span_checks",
    "document_support_assessment",
    "cross_document_consistency",
    "notes",
}

VALID_SPAN_LABELS = {"supports", "weak", "contradicts"}
VALID_DOC_SUPPORT = {"strong_support", "mixed", "weak_support", "contradicted"}
VALID_CONSISTENCY = {"consistent", "partially_consistent", "inconsistent", "unknown"}
VALID_CRED = {"high", "medium", "low", "unknown"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate EAJP scorecard schema in details outputs.")
    p.add_argument("--path", required=True, help="Details file or results directory to validate.")
    p.add_argument(
        "--strict-enums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate enum-like fields against expected label sets.",
    )
    return p.parse_args()


def _iter_detail_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*_details.json"))


def _err(errors: list[str], file: Path, idx: int, msg: str) -> None:
    errors.append(f"{file} :: trial_index={idx} :: {msg}")


_EAJP_SYSTEMS = {"eajp", "eajp_no_privilege"}


def _validate_trial(file: Path, idx: int, trial: dict[str, Any], strict_enums: bool, errors: list[str]) -> None:
    system = trial.get("system")
    if system not in _EAJP_SYSTEMS:
        return

    md = trial.get("metadata")
    if not isinstance(md, dict):
        _err(errors, file, idx, "metadata missing or not object")
        return

    if md.get("judge_mode") not in _EAJP_SYSTEMS:
        _err(errors, file, idx, f"metadata.judge_mode '{md.get('judge_mode')}' not in {_EAJP_SYSTEMS}")

    jm = md.get("judge_metadata")
    if not isinstance(jm, dict):
        _err(errors, file, idx, "metadata.judge_metadata missing or not object")
        return

    for k in ["short_circuit_used", "scorecard_privilege_applied", "final_winner_id", "final_verdict"]:
        if k not in jm:
            _err(errors, file, idx, f"metadata.judge_metadata.{k} missing")

    if "short_circuit_used" in jm and not isinstance(jm.get("short_circuit_used"), bool):
        _err(errors, file, idx, "metadata.judge_metadata.short_circuit_used not bool")
    if "scorecard_privilege_applied" in jm and not isinstance(jm.get("scorecard_privilege_applied"), bool):
        _err(errors, file, idx, "metadata.judge_metadata.scorecard_privilege_applied not bool")

    sc = jm.get("scorecard")
    if not isinstance(sc, dict):
        _err(errors, file, idx, "metadata.judge_metadata.scorecard missing or not object")
        return

    for k in ["scorecard", "preliminary_winner_id", "preliminary_verdict"]:
        if k not in sc:
            _err(errors, file, idx, f"metadata.judge_metadata.scorecard.{k} missing")

    entries = sc.get("scorecard")
    if not isinstance(entries, list):
        _err(errors, file, idx, "metadata.judge_metadata.scorecard.scorecard not list")
        return

    for e_i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _err(errors, file, idx, f"scorecard entry {e_i} not object")
            continue

        missing = sorted(REQUIRED_SCORECARD_ENTRY_FIELDS - set(entry.keys()))
        if missing:
            _err(errors, file, idx, f"scorecard entry {e_i} missing fields: {missing}")

        span_checks = entry.get("span_checks")
        if not isinstance(span_checks, list):
            _err(errors, file, idx, f"scorecard entry {e_i}.span_checks not list")
        else:
            for s_i, span_item in enumerate(span_checks):
                if not isinstance(span_item, dict):
                    _err(errors, file, idx, f"scorecard entry {e_i}.span_checks[{s_i}] not object")
                    continue
                if "span" not in span_item or "label" not in span_item:
                    _err(errors, file, idx, f"scorecard entry {e_i}.span_checks[{s_i}] missing span/label")
                if strict_enums and "label" in span_item:
                    if str(span_item.get("label")) not in VALID_SPAN_LABELS:
                        _err(errors, file, idx, f"invalid span_checks label in entry {e_i}: {span_item.get('label')}")

        if strict_enums:
            cred = str(entry.get("credibility_tier", ""))
            if cred and cred not in VALID_CRED:
                _err(errors, file, idx, f"invalid credibility_tier in entry {e_i}: {cred}")

            doc_support = str(entry.get("document_support_assessment", ""))
            if doc_support and doc_support not in VALID_DOC_SUPPORT:
                _err(errors, file, idx, f"invalid document_support_assessment in entry {e_i}: {doc_support}")

            consistency = str(entry.get("cross_document_consistency", ""))
            if consistency and consistency not in VALID_CONSISTENCY:
                _err(errors, file, idx, f"invalid cross_document_consistency in entry {e_i}: {consistency}")


def main() -> int:
    args = parse_args()
    target = Path(args.path)
    if not target.exists():
        print(f"Path does not exist: {target}", file=sys.stderr)
        return 2

    files = _iter_detail_files(target)
    if not files:
        print(f"No *_details.json files found under: {target}", file=sys.stderr)
        return 2

    errors: list[str] = []
    eajp_trials = 0
    for file in files:
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{file} :: unable to parse JSON: {exc}")
            continue

        if not isinstance(payload, list):
            errors.append(f"{file} :: top-level JSON is not a list")
            continue

        for i, trial in enumerate(payload):
            if not isinstance(trial, dict):
                errors.append(f"{file} :: trial_index={i} :: trial is not object")
                continue
            if trial.get("system") in _EAJP_SYSTEMS:
                eajp_trials += 1
            _validate_trial(file, i, trial, args.strict_enums, errors)

    print(f"Checked detail files: {len(files)}")
    print(f"EAJP-family trials checked ({', '.join(sorted(_EAJP_SYSTEMS))}): {eajp_trials}")
    if errors:
        print(f"Schema validation FAILED with {len(errors)} issue(s):", file=sys.stderr)
        for err in errors[:200]:
            print(f"- {err}", file=sys.stderr)
        if len(errors) > 200:
            print(f"... plus {len(errors) - 200} more", file=sys.stderr)
        return 1

    print("Schema validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
