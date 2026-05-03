#!/usr/bin/env python3
"""Build and score a TCB dataset from RAMDocs (single scoring pass, no iterative rewrites).

Workflow
--------
1. Load the pre-converted RAMDocs JSON (--source).
2. Build initial TCB rewrites via the configured provider (default: Anthropic
   claude-haiku-4-5-20251001).
3. Score every (original, rewritten) pair once with rescaled BERTScore F1
   (roberta-large) and sentence cosine similarity (all-mpnet-base-v2).
4. Print the full score distribution and the bottom --n-low-end pairs for
   human inspection.  The auto-detected drift threshold is shown alongside the
   configured threshold so you can see where meaning clearly drifts.
5. Drop instances whose pairs fall below --bs-target / --cosine-target and
   save the final TCB JSON and quality report.

Set --bs-target from what you observe in the distribution.  Rescaled
roberta-large BERTScore for Wikipedia-style prose typically ranges 0.60–0.78;
a threshold around 0.65–0.70 captures genuine meaning drift without over-dropping.

Output layout (all under --dest-dir, default data/full/ramdocs):
  tcb_ramdocs_full.json          — final TCB dataset (threshold failures dropped)
  tcb_ramdocs_full.report.json   — quality summary
  quality/iter01_scores.json     — per-pair BERTScore and cosine scores
  quality/iter01_distribution.png — score distribution plot

Smoke-test mode (--smoke-test):
  Processes only --smoke-n instances end-to-end, prints the full quality
  report, and exits.  Use to verify the pipeline before a full run.

Usage
-----
  # Full run (500 RAMDocs instances):
  python experiments/build_tcb_ramdocs.py

  # Smoke test (5 instances):
  python experiments/build_tcb_ramdocs.py --smoke-test

  # Score an existing TCB JSON (skip the rewrite build):
  python experiments/build_tcb_ramdocs.py --skip-initial-build

  # Set threshold after inspecting the distribution:
  python experiments/build_tcb_ramdocs.py --bs-target 0.70 --cosine-target 0.90
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

_DEFAULT_REWRITER_PROVIDER = "anthropic"
_DEFAULT_REWRITER_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_SOURCE = "data/full/ramdocs/ramdocs_test.json"
_DEFAULT_DEST_DIR = "data/full/ramdocs"
# _TEMPERATURE_SCHEDULE = [0.2, 0.4, 0.6, 0.8]  # disabled — no iterative rewrites
_N_LOW_END = 15
_PREVIEW = 300  # chars per doc shown in low-end printout


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def _build_pairs(tcb_data: list[dict], orig_data: list[dict]) -> list[dict]:
    """Match TCB rewritten docs back to originals.

    Matches via metadata.constructed_from (instance-level) and
    metadata.original_doc_id (doc-level).  Returns a flat list of dicts:
      {tcb_instance_id, src_instance_id, doc_id, tone, original, rewritten}
    """
    orig_by_id = {str(inst["id"]): inst for inst in orig_data}
    pairs: list[dict] = []
    for tcb_inst in tcb_data:
        src_id = str(tcb_inst.get("metadata", {}).get("constructed_from", ""))
        orig_inst = orig_by_id.get(src_id)
        if orig_inst is None:
            continue
        orig_docs = {str(d["id"]): d for d in orig_inst.get("documents", [])}
        for doc in tcb_inst.get("documents", []):
            odid = str(doc.get("metadata", {}).get("original_doc_id", ""))
            od = orig_docs.get(odid)
            if od is None:
                continue
            if not od.get("content", "").strip() or not doc.get("content", "").strip():
                continue
            pairs.append({
                "tcb_instance_id": tcb_inst["id"],
                "src_instance_id": src_id,
                "doc_id": doc["id"],
                "tone": doc.get("tone", ""),
                "original": od["content"].strip(),
                "rewritten": doc["content"].strip(),
            })
    return pairs


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_bertscore(device: str | None) -> Any:
    from bert_score import BERTScorer  # type: ignore
    logger.info("Loading BERTScorer (roberta-large, rescale_with_baseline=True, device=%s) …", device)
    return BERTScorer(
        model_type="roberta-large",
        rescale_with_baseline=True,
        lang="en",
        device=device,
    )


def _load_sbert(device: str | None) -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info("Loading SentenceTransformer (all-mpnet-base-v2, device=%s) …", device)
    kw: dict = {}
    if device:
        kw["device"] = device
    return SentenceTransformer("all-mpnet-base-v2", **kw)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_bertscore(
    pairs: list[dict],
    scorer: Any,
    batch_size: int,
    doc_id_filter: set[str] | None = None,
) -> dict[str, float]:
    subset = pairs if doc_id_filter is None else [p for p in pairs if p["doc_id"] in doc_id_filter]
    if not subset:
        return {}
    results: dict[str, float] = {}
    for start in range(0, len(subset), batch_size):
        batch = subset[start:start + batch_size]
        _, _, F1 = scorer.score(
            [p["rewritten"] for p in batch],
            [p["original"] for p in batch],
        )
        for p, f1 in zip(batch, F1):
            results[p["doc_id"]] = float(f1.item())
    return results


def _score_cosine(
    pairs: list[dict],
    model: Any,
    batch_size: int,
    doc_id_filter: set[str] | None = None,
) -> dict[str, float]:
    from sentence_transformers.util import cos_sim  # type: ignore
    subset = pairs if doc_id_filter is None else [p for p in pairs if p["doc_id"] in doc_id_filter]
    if not subset:
        return {}
    emb_o = model.encode(
        [p["original"] for p in subset],
        batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False,
    )
    emb_r = model.encode(
        [p["rewritten"] for p in subset],
        batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False,
    )
    sims = cos_sim(emb_o, emb_r)
    return {p["doc_id"]: float(sims[i, i].item()) for i, p in enumerate(subset)}


# ---------------------------------------------------------------------------
# Distribution analysis helpers
# ---------------------------------------------------------------------------

def _detect_drift(scores: list[float], floor: float = 0.50) -> float:
    arr = np.array(sorted(scores))
    cutoff = max(2, int(len(arr) * 0.30))
    best_gap, best_mid = 0.0, float(np.percentile(arr, 5))
    for i in range(cutoff - 1):
        gap = float(arr[i + 1] - arr[i])
        if gap > best_gap:
            best_gap = gap
            best_mid = float((arr[i] + arr[i + 1]) / 2)
    if best_gap < 0.02:
        best_mid = float(np.percentile(arr, 5))
    return max(floor, round(best_mid, 3))


def _describe(scores: list[float], label: str) -> None:
    arr = np.array(scores)
    logger.info(
        "%s: n=%d  mean=%.3f  std=%.3f  min=%.3f  p5=%.3f  p10=%.3f  "
        "p25=%.3f  median=%.3f  p90=%.3f  max=%.3f",
        label, len(arr), arr.mean(), arr.std(), arr.min(),
        np.percentile(arr, 5), np.percentile(arr, 10),
        np.percentile(arr, 25), np.median(arr),
        np.percentile(arr, 90), arr.max(),
    )


def _print_low_end(
    pairs: list[dict],
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    n: int,
    iteration: int,
) -> None:
    rows = sorted(
        [(min(bs_by_id.get(p["doc_id"], 1.0), cos_by_id.get(p["doc_id"], 1.0)),
          bs_by_id.get(p["doc_id"], 1.0),
          cos_by_id.get(p["doc_id"], 1.0),
          p) for p in pairs],
        key=lambda t: t[0],
    )
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Bottom {n} pairs — iteration {iteration}  (sort: min(BERTScore, cosine))")
    print(sep)
    for rank, (combined, bs, cos, pair) in enumerate(rows[:n], 1):
        orig_p = pair["original"].replace("\n", " ")[:_PREVIEW]
        rew_p = pair["rewritten"].replace("\n", " ")[:_PREVIEW]
        print(f"\n  [{rank:2d}]  BS={bs:.4f}  Cos={cos:.4f}  tone={pair['tone']}")
        print(f"       doc_id   : {pair['doc_id']}")
        print(f"       ORIGINAL : {orig_p}")
        if len(pair["original"]) > _PREVIEW:
            print(f"                … (+{len(pair['original']) - _PREVIEW} chars)")
        print(f"       REWRITTEN: {rew_p}")
        if len(pair["rewritten"]) > _PREVIEW:
            print(f"                … (+{len(pair['rewritten']) - _PREVIEW} chars)")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Per-iteration output
# ---------------------------------------------------------------------------

def _save_iter_scores(
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    bs_target: float,
    cos_target: float,
    report: dict,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        {
            "doc_id": doc_id,
            "bertscore_f1": round(bs_by_id[doc_id], 5),
            "cosine_similarity": round(cos_by_id.get(doc_id, float("nan")), 5),
            "pass_bs": bs_by_id[doc_id] >= bs_target,
            "pass_cos": cos_by_id.get(doc_id, 0.0) >= cos_target,
        }
        for doc_id in bs_by_id
    ]
    rows.sort(key=lambda r: min(r["bertscore_f1"], r["cosine_similarity"]))
    _save_json(rows, os.path.join(out_dir, "meaning_similarity_scores.json"))
    _plot_iter(
        list(bs_by_id.values()),
        [cos_by_id.get(d, float("nan")) for d in bs_by_id],
        bs_target, cos_target, report,
        os.path.join(out_dir, "meaning_similarity_distribution.png"),
    )


_NEURIPS_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _draw_booktabs_table(ax: Any, table_data: list, row_ys: list, col_xs: list,
                         col_has: list) -> None:
    """Render a booktabs-style table (no vertical rules, top/bottom rules only)."""
    import matplotlib as mpl  # type: ignore
    for i, row in enumerate(table_data):
        for x, ha, txt in zip(col_xs, col_has, row):
            ax.text(x, row_ys[i], txt, transform=ax.transAxes,
                    ha=ha, va="center", fontsize=9,
                    fontfamily=mpl.rcParams.get("font.family", "serif"))
    T = ax.transAxes
    for y, lw in [(1.0, 1.5), (0.0, 1.5)]:
        ax.plot([0, 1], [y, y], transform=T, color="black", lw=lw, clip_on=False)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _plot_iter(
    bs_scores: list[float],
    cos_scores: list[float],
    bs_target: float,
    cos_target: float,
    report: dict,
    path: str,
) -> None:
    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.gridspec as gridspec  # type: ignore

        bs_arr = np.array([s for s in bs_scores if not np.isnan(s)])
        cos_arr = np.array([s for s in cos_scores if not np.isnan(s)])

        with mpl.rc_context(_NEURIPS_RC):
            fig = plt.figure(figsize=(6.75, 5.2))
            gs = gridspec.GridSpec(
                2, 2, height_ratios=[1, 2.6], hspace=0.52,
                left=0.09, right=0.97, top=0.91, bottom=0.11,
            )
            ax_tbl = fig.add_subplot(gs[0, :])
            ax_bs  = fig.add_subplot(gs[1, 0])
            ax_cos = fig.add_subplot(gs[1, 1])
            fig.suptitle(
                "RAMDocs TCB — Meaning Preservation Quality",
                fontsize=10, fontweight="normal", y=0.975,
            )

            # ── booktabs table ────────────────────────────────────────────────
            table_data = [
                ("Original Dataset Size (Docs)",      str(report.get("original_n_instances", "?")),
                 "BERTScore F1 (rescaled) Min",        f"{report.get('final_min_bs', float('nan')):.4f}"),
                ("Scored Docs",                        str(report.get("scored_n_instances", "?")),
                 "BERTScore F1 (rescaled) Mean",       f"{report.get('final_mean_bs', float('nan')):.4f}"),
                ("Docs Passing Both Thresholds",       str(report.get("final_n_instances", "?")),
                 "Cosine Similarity Min",              f"{report.get('final_min_cos', float('nan')):.4f}"),
                ("Doc Pairs Passing Both Thresholds",  str(report.get("final_n_pairs", "?")),
                 "Cosine Similarity Mean",             f"{report.get('final_mean_cos', float('nan')):.4f}"),
            ]
            _draw_booktabs_table(
                ax_tbl, table_data,
                row_ys =[0.875, 0.625, 0.375, 0.125],
                col_xs =[0.01,  0.40,  0.53,  0.99],
                col_has=["left","right","left","right"],
            )

            # ── histograms ────────────────────────────────────────────────────
            for ax, arr, label, target, color in [
                (ax_bs,  bs_arr,  "BERTScore F1 (rescaled)", bs_target, "#4C78A8"),
                (ax_cos, cos_arr, "Cosine Similarity",        cos_target, "#E07B39"),
            ]:
                if len(arr) == 0:
                    continue
                ax.hist(arr, bins=40, color=color, edgecolor="white", linewidth=0.4, alpha=0.9)
                ax.axvline(target, color="#CC0000", lw=1.2, ls="--",
                           label=f"Threshold = {target:.2f}")
                ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5, zorder=0)
                ax.set_axisbelow(True)
                ax.set_xlabel(label)
                ax.set_ylabel("Count")
                ax.legend(frameon=False)

            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved distribution plot → %s", path)
    except Exception as exc:
        logger.warning("Plotting failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Initial TCB build
# ---------------------------------------------------------------------------

def _build_initial_tcb(
    source_path: str,
    tcb_path: str,
    provider: str,
    model: str,
    checkpoint_every: int,
    max_workers: int,
) -> None:
    from src.data_loader import load_dataset, save_dataset
    from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

    instances = load_dataset(source_path)
    logger.info(
        "Building initial TCB: %d instances → %s  (provider=%s model=%s workers=%d)",
        len(instances), tcb_path, provider, model, max_workers,
    )

    def _make_builder():
        return ToneControlledBenchmarkBuilder(
            rewriter=ToneRewriter(provider=provider, model=model)
        )

    if max_workers <= 1:
        builder = _make_builder()
        built: list = []
        for idx, inst in enumerate(instances, 1):
            try:
                tcb_inst = builder.build_instance(inst, source_benchmark="ramdocs")
                if tcb_inst is not None:
                    built.append(tcb_inst)
            except Exception as exc:
                logger.warning("Skipping instance %s: %s", inst.id, exc)
            if built and len(built) % checkpoint_every == 0:
                save_dataset(built, tcb_path)
                logger.info("Checkpoint saved: %d/%d instances processed", idx, len(instances))
        save_dataset(built, tcb_path)
        logger.info("Initial TCB done: %d instances → %s", len(built), tcb_path)
        return

    _local = threading.local()

    def _build_one(inst):
        if not hasattr(_local, "builder"):
            _local.builder = _make_builder()
        try:
            return inst.id, _local.builder.build_instance(inst, source_benchmark="ramdocs")
        except Exception as exc:
            logger.warning("Skipping instance %s: %s", inst.id, exc)
            return inst.id, None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    built_map: dict = {}
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_one, inst): inst for inst in instances}
        for fut in as_completed(futures):
            inst_id, result = fut.result()
            with lock:
                if result is not None:
                    built_map[inst_id] = result
                completed += 1
                if completed % checkpoint_every == 0:
                    ordered = [built_map[i.id] for i in instances if i.id in built_map]
                    save_dataset(ordered, tcb_path)
                    logger.info("Checkpoint: %d/%d submitted", completed, len(instances))

    built_list = [built_map[i.id] for i in instances if i.id in built_map]
    save_dataset(built_list, tcb_path)
    logger.info("Initial TCB done: %d instances → %s", len(built_list), tcb_path)


# ---------------------------------------------------------------------------
# Failed-instance re-write
# ---------------------------------------------------------------------------

def _get_failed_ids(
    pairs: list[dict],
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    bs_target: float,
    cos_target: float,
) -> set[str]:
    """Return TCB instance IDs that have at least one failing doc pair."""
    failed: set[str] = set()
    for p in pairs:
        if bs_by_id.get(p["doc_id"], 0.0) < bs_target or cos_by_id.get(p["doc_id"], 0.0) < cos_target:
            failed.add(p["tcb_instance_id"])
    return failed


# ---------------------------------------------------------------------------
# NOTE: _redo_failed is disabled — iterative rewrites replaced by single-pass
# threshold scoring.  Retained here for reference; not called anywhere.
# ---------------------------------------------------------------------------
def _redo_failed(
    tcb_path: str,
    orig_data: list[dict],
    failed_tcb_ids: set[str],
    provider: str,
    model: str,
    temperature: float,
    max_workers: int,
) -> tuple[list[dict], set[str]]:
    """Re-write failing instances; returns (updated_tcb_data, rebuilt_doc_ids)."""
    from src.data_loader import instance_from_dict
    from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

    tcb_data = _load_json(tcb_path)
    passing = [inst for inst in tcb_data if inst["id"] not in failed_tcb_ids]
    failed_src_ids: set[str] = {
        str(inst.get("metadata", {}).get("constructed_from", ""))
        for inst in tcb_data if inst["id"] in failed_tcb_ids
    } - {""}

    logger.info(
        "Re-writing %d source instances (temperature=%.2f); keeping %d passing.",
        len(failed_src_ids), temperature, len(passing),
    )

    orig_by_id = {str(d["id"]): d for d in orig_data}
    _local2 = threading.local()

    def _redo_one(src_id: str):
        if not hasattr(_local2, "builder"):
            _local2.builder = ToneControlledBenchmarkBuilder(
                rewriter=ToneRewriter(provider=provider, model=model)
            )
        raw = orig_by_id.get(src_id)
        if raw is None:
            logger.warning("Source instance %s not found; dropping.", src_id)
            return src_id, None
        try:
            tcb_inst = _local2.builder.build_instance(
                instance_from_dict(raw),
                source_benchmark="ramdocs",
                temperature=temperature,
            )
            if tcb_inst is None:
                logger.warning(
                    "Instance %s still fails at temperature=%.2f: %s",
                    src_id, temperature, _local2.builder.last_failure_reason,  # noqa: SLF001
                )
            return src_id, tcb_inst
        except Exception as exc:
            logger.warning("Error re-writing %s: %s", src_id, exc)
            return src_id, None

    rebuilt: list[dict] = []
    rebuilt_doc_ids: set[str] = set()
    skipped = 0

    if max_workers <= 1:
        for src_id in failed_src_ids:
            _, inst = _redo_one(src_id)
            if inst is not None:
                from src.data_loader import instance_to_dict
                rebuilt.append(instance_to_dict(inst))
                for doc in inst.documents:
                    rebuilt_doc_ids.add(doc.id)
            else:
                skipped += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data_loader import instance_to_dict
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_redo_one, sid): sid for sid in failed_src_ids}
            for fut in as_completed(futures):
                _, inst = fut.result()
                if inst is not None:
                    rebuilt.append(instance_to_dict(inst))
                    for doc in inst.documents:
                        rebuilt_doc_ids.add(doc.id)
                else:
                    skipped += 1

    logger.info("Re-write results: rebuilt=%d  dropped=%d", len(rebuilt), skipped)
    merged = passing + rebuilt
    _save_json(merged, tcb_path)
    return merged, rebuilt_doc_ids


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def _validate_loop(
    tcb_path: str,
    orig_data: list[dict],
    bs_target: float,
    cos_target: float,
    batch_size: int,
    device: str | None,
    quality_dir: str,
    bs_scorer: Any,
    cos_model: Any,
    n_low_end: int,
) -> dict:
    """Single scoring pass: score all pairs, print distribution, drop failures.

    No iterative rewrites.  Set --bs-target / --cosine-target based on what
    you observe in the printed distribution and low-end pair inspection.
    """
    os.makedirs(quality_dir, exist_ok=True)
    tcb_data = _load_json(tcb_path)
    scored_n_instances = len(tcb_data)
    pairs = _build_pairs(tcb_data, orig_data)
    logger.info("Scoring: %d TCB instances, %d doc pairs", len(tcb_data), len(pairs))

    bs_cache: dict[str, float] = _score_bertscore(pairs, bs_scorer, batch_size)
    cos_cache: dict[str, float] = _score_cosine(pairs, cos_model, batch_size)

    bs_vals = [bs_cache.get(p["doc_id"], 0.0) for p in pairs]
    cos_vals = [cos_cache.get(p["doc_id"], 0.0) for p in pairs]

    print(f"\n{'─' * 60}")
    print(f"  RAMDocs TCB — scoring pass")
    print(f"{'─' * 60}")
    _describe(bs_vals, "BERTScore F1 (rescaled)")
    _describe(cos_vals, "Cosine similarity")

    bs_drift = _detect_drift(bs_vals)
    cos_drift = _detect_drift(cos_vals)
    print(f"\n  Auto-detected drift  →  BERTScore: {bs_drift:.4f}   Cosine: {cos_drift:.4f}")
    print(f"  Configured threshold →  BERTScore: {bs_target:.4f}   Cosine: {cos_target:.4f}")

    _print_low_end(pairs, bs_cache, cos_cache, n_low_end, 1)

    fail_bs = sum(1 for v in bs_vals if v < bs_target)
    fail_cos = sum(1 for v in cos_vals if v < cos_target)
    failed_ids = _get_failed_ids(pairs, bs_cache, cos_cache, bs_target, cos_target)
    fail_either = len(failed_ids)
    pass_rate_bs = 1.0 - fail_bs / max(len(bs_vals), 1)
    pass_rate_cos = 1.0 - fail_cos / max(len(cos_vals), 1)

    print(
        f"\n  Pass rate  →  BERTScore: {pass_rate_bs:.1%} ({fail_bs} fail)   "
        f"Cosine: {pass_rate_cos:.1%} ({fail_cos} fail)   "
        f"Instances failing either: {fail_either}"
    )

    # Drop instances whose doc pairs fall below either threshold
    if failed_ids:
        print(f"\n  Dropping {fail_either} instance(s) below threshold (no rewrites attempted).")
        kept = [inst for inst in tcb_data if inst["id"] not in failed_ids]
        _save_json(kept, tcb_path)
        logger.info("Dropped %d instances; final: %d", len(failed_ids), len(kept))
        tcb_data = kept

    final_pairs = _build_pairs(tcb_data, orig_data)
    final_bs = [bs_cache.get(p["doc_id"], 0.0) for p in final_pairs if p["doc_id"] in bs_cache]
    final_cos = [cos_cache.get(p["doc_id"], 0.0) for p in final_pairs if p["doc_id"] in cos_cache]

    report = {
        "benchmark": "ramdocs",
        "tcb_path": tcb_path,
        "original_n_instances": len(orig_data),
        "scored_n_instances": scored_n_instances,
        "total_scored": len(pairs),
        "dropped_instances": fail_either,
        "final_n_instances": len(tcb_data),
        "final_n_pairs": len(final_pairs),
        "final_pass_rate_bs": round(float(np.mean([v >= bs_target for v in final_bs])), 4) if final_bs else 0.0,
        "final_pass_rate_cos": round(float(np.mean([v >= cos_target for v in final_cos])), 4) if final_cos else 0.0,
        "final_mean_bs": round(float(np.mean(final_bs)), 4) if final_bs else 0.0,
        "final_mean_cos": round(float(np.mean(final_cos)), 4) if final_cos else 0.0,
        "final_min_bs": round(float(np.min(final_bs)), 4) if final_bs else 0.0,
        "final_min_cos": round(float(np.min(final_cos)), 4) if final_cos else 0.0,
        "bs_target": bs_target,
        "cos_target": cos_target,
        "drift_bs": round(bs_drift, 4),
        "drift_cos": round(cos_drift, 4),
    }
    _save_iter_scores(bs_cache, cos_cache, bs_target, cos_target, report, quality_dir)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build and score a TCB dataset from RAMDocs (single pass, no iterative rewrites).\n\n"
            "Scores all pairs once with BERTScore F1 and cosine similarity, prints the "
            "distribution and low-end pairs for inspection, then drops instances below "
            "--bs-target / --cosine-target.  Set thresholds from what you observe."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        default=_DEFAULT_SOURCE,
        metavar="PATH",
        help="Path to pre-converted RAMDocs JSON.",
    )
    p.add_argument(
        "--dest-dir",
        default=_DEFAULT_DEST_DIR,
        metavar="DIR",
        help=(
            "Base output directory.  TCB JSON is written to <dest-dir>/tcb_ramdocs_full.json; "
            "quality reports go in <dest-dir>/quality/; "
            "the JSON report is <dest-dir>/tcb_ramdocs_full.report.json."
        ),
    )
    p.add_argument(
        "--rewriter-provider",
        default=_DEFAULT_REWRITER_PROVIDER,
        choices=["anthropic", "gemini", "local"],
        help="Rewrite provider (default: anthropic).",
    )
    p.add_argument(
        "--rewriter-model",
        default=None,
        metavar="MODEL",
        help=(
            "Model name for the rewrite provider. "
            f"Defaults: anthropic→{_DEFAULT_REWRITER_MODEL}, "
            "gemini→gemini-3.1-flash-lite-preview."
        ),
    )
    p.add_argument(
        "--bs-target",
        type=float,
        default=0.70,
        help=(
            "Drop threshold for rescaled BERTScore F1 (roberta-large).  "
            "Inspect the distribution first — Wikipedia-style prose typically scores 0.60–0.78; "
            "set around 0.65–0.70 where you observe clear meaning drift."
        ),
    )
    p.add_argument(
        "--cosine-target",
        type=float,
        default=0.90,
        help="Drop threshold for cosine similarity (all-mpnet-base-v2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Scoring batch size (recommend 128 for mps/cuda, 64 for cpu).",
    )
    p.add_argument(
        "--rewrite-workers",
        type=int,
        default=6,
        help="Parallel API workers for the initial TCB build (not used during scoring).",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save TCB checkpoint every N instances during initial build.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help='Torch device ("mps", "cuda", "cpu"). Auto-detected when omitted.',
    )
    p.add_argument(
        "--n-low-end",
        type=int,
        default=_N_LOW_END,
        help="Number of lowest-scoring pairs printed per iteration for inspection.",
    )
    p.add_argument(
        "--skip-initial-build",
        action="store_true",
        help="Skip initial TCB build (TCB JSON must already exist in --dest-dir).",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and re-build the TCB JSON even if it already exists.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run end-to-end on only --smoke-n instances and exit.",
    )
    p.add_argument(
        "--smoke-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of instances for --smoke-test.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    dest_dir = args.dest_dir
    os.makedirs(dest_dir, exist_ok=True)

    tcb_path = os.path.join(dest_dir, "tcb_ramdocs_full.json")
    report_path = os.path.join(dest_dir, "tcb_ramdocs_full.report.json")
    quality_dir = os.path.join(dest_dir, "quality")

    # Smoke-test: work in a temp subdir with a small slice of source data
    source_path = args.source
    if args.smoke_test:
        logger.info("=== SMOKE TEST MODE: %d instances ===", args.smoke_n)
        smoke_dir = os.path.join(dest_dir, "smoke")
        os.makedirs(smoke_dir, exist_ok=True)
        raw_all = _load_json(source_path)
        smoke_src = os.path.join(smoke_dir, "smoke_source.json")
        _save_json(raw_all[: args.smoke_n], smoke_src)
        source_path = smoke_src
        tcb_path = os.path.join(smoke_dir, "tcb_smoke.json")
        report_path = os.path.join(smoke_dir, "tcb_smoke.report.json")
        quality_dir = os.path.join(smoke_dir, "quality")
        logger.info("Smoke source: %s  |  TCB: %s", source_path, tcb_path)

    # ---- Initial TCB build -----------------------------------------------
    if args.force_rebuild and os.path.exists(tcb_path):
        os.remove(tcb_path)
        logger.info("Removed existing TCB (--force-rebuild).")

    rewriter_model = args.rewriter_model or _DEFAULT_REWRITER_MODEL
    if not args.skip_initial_build and not os.path.exists(tcb_path):
        t0 = time.time()
        _build_initial_tcb(
            source_path=source_path,
            tcb_path=tcb_path,
            provider=args.rewriter_provider,
            model=rewriter_model,
            checkpoint_every=args.checkpoint_every,
            max_workers=args.rewrite_workers,
        )
        logger.info("Initial build complete in %.1f s.", time.time() - t0)
    elif os.path.exists(tcb_path):
        n = len(_load_json(tcb_path))
        logger.info("Existing TCB found (%d instances): %s", n, tcb_path)
    else:
        logger.error("TCB not found at %s and --skip-initial-build was set.", tcb_path)
        sys.exit(1)

    # ---- Load scoring models ---------------------------------------------
    device = args.device or _auto_device()
    logger.info("Using device: %s", device)
    bs_scorer = _load_bertscore(device)
    cos_model = _load_sbert(device)

    # ---- Single scoring pass ---------------------------------------------
    orig_data = _load_json(source_path)
    t1 = time.time()
    summary = _validate_loop(
        tcb_path=tcb_path,
        orig_data=orig_data,
        bs_target=args.bs_target,
        cos_target=args.cosine_target,
        batch_size=args.batch_size,
        device=device,
        quality_dir=quality_dir,
        bs_scorer=bs_scorer,
        cos_model=cos_model,
        n_low_end=args.n_low_end,
    )
    logger.info("Scoring pass complete in %.1f s.", time.time() - t1)

    # ---- Final report ----------------------------------------------------
    _save_json(summary, report_path)
    print(f"\n{'=' * 60}")
    print("  FINAL QUALITY REPORT")
    print(f"{'=' * 60}")
    print(f"  TCB dataset       : {tcb_path}")
    print(f"  Scored pairs      : {summary['total_scored']}")
    print(f"  Dropped instances : {summary['dropped_instances']}")
    print(f"  Final instances   : {summary['final_n_instances']}")
    print(f"  Final doc pairs   : {summary['final_n_pairs']}")
    print(f"  BERTScore F1      : mean={summary['final_mean_bs']:.4f}  "
          f"min={summary['final_min_bs']:.4f}  "
          f"pass@{args.bs_target}={summary['final_pass_rate_bs']:.1%}")
    print(f"  Cosine similarity : mean={summary['final_mean_cos']:.4f}  "
          f"min={summary['final_min_cos']:.4f}  "
          f"pass@{args.cosine_target}={summary['final_pass_rate_cos']:.1%}")
    print(f"  Drift (auto)      : BS={summary['drift_bs']:.4f}  Cos={summary['drift_cos']:.4f}")
    print(f"  Report saved      : {report_path}")
    print(f"  Quality dir       : {quality_dir}")
    print(f"{'=' * 60}\n")

    if summary["dropped_instances"] == 0:
        print("All pairs meet both quality targets.")
    else:
        print(
            f"{summary['dropped_instances']} instance(s) dropped below "
            f"BS≥{args.bs_target} / Cosine≥{args.cosine_target}.  "
            f"Inspect quality/iter01_scores.json to review dropped pairs before "
            f"adjusting --bs-target."
        )

    if args.smoke_test:
        print(f"\nSmoke test complete.  Artifacts in: {os.path.join(dest_dir, 'smoke')}")


if __name__ == "__main__":
    main()
