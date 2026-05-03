#!/usr/bin/env python3
"""Single-click pipeline: download, build, and score TCB datasets (single pass, no iterative rewrites).

Downloads configured corpora from Hugging Face, builds Tone-Controlled
Benchmark (TCB) datasets, then scores every (original, rewritten) pair once
with rescaled BERTScore F1 and cosine similarity.

Scoring pass per dataset:
  1.  Score all (original, rewritten) pairs.
  2.  Detect where meaning clearly drifts in the score distribution
      (largest gap in the bottom 30 %; printed for human inspection).
  3.  Print the bottom N pairs side-by-side for visual inspection.
  4.  Drop instances below --bs-target / --cosine-target.

Set --bs-target from what you observe in the printed distribution.
Wikipedia-style prose (roberta-large rescaled) typically scores 0.60–0.78;
a threshold around 0.65–0.70 captures genuine meaning drift.

Outputs (under DATA_ROOT, default data/full/):
  ramdocs_test.json          – converted RAMDocs test split
  tcb_ramdocs_test.json      – final TCB for RAMDocs (threshold failures dropped)
  quality_report.json        – per-dataset quality summary
  quality_ramdocs/           – score files + distribution plot

Usage:
  python experiments/build_and_validate_tcb_full.py

  # Override targets or paths:
  python experiments/build_and_validate_tcb_full.py \\
      --data-root data/full \\
      --bs-target 0.70 --cosine-target 0.90 \\
      --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

# _TEMPERATURE_SCHEDULE = [0.2, 0.4, 0.6, 0.8]  # disabled — no iterative rewrites

# Number of low-scoring pairs printed for human inspection
_N_LOW_END = 15
_PREVIEW = 300  # chars of each document to show


# ---------------------------------------------------------------------------
# Helpers shared with the analysis scripts
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _build_pairs(tcb_data: list[dict], orig_data: list[dict]) -> list[dict]:
    """Return (original_text, rewritten_text, metadata) pairs."""
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
# Scoring
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


def _load_bertscore(device: str | None) -> Any:
    from bert_score import BERTScorer  # type: ignore
    logger.info("Loading BERTScorer (roberta-large, rescale_with_baseline=True, device=%s) …", device)
    # lang="en" required for rescale_with_baseline=True (bert-score>=0.3.12).
    # verbose is not a valid BERTScorer parameter.
    return BERTScorer(
        model_type="roberta-large",
        rescale_with_baseline=True,
        lang="en",
        device=device,
    )


def _load_sbert(device: str | None) -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info("Loading SentenceTransformer (all-mpnet-base-v2) …")
    kw: dict = {}
    if device:
        kw["device"] = device
    return SentenceTransformer("all-mpnet-base-v2", **kw)


def _score_bertscore(
    pairs: list[dict],
    scorer: Any,
    batch_size: int,
    doc_id_filter: set[str] | None = None,
) -> dict[str, float]:
    """Return {doc_id: f1_score}.  If doc_id_filter is given, only score those."""
    subset = pairs if doc_id_filter is None else [p for p in pairs if p["doc_id"] in doc_id_filter]
    if not subset:
        return {}
    orig = [p["original"] for p in subset]
    rew = [p["rewritten"] for p in subset]
    results: dict[str, float] = {}
    for start in range(0, len(orig), batch_size):
        b_orig = orig[start:start + batch_size]
        b_rew = rew[start:start + batch_size]
        _, _, F1 = scorer.score(b_rew, b_orig)
        for pair, f1 in zip(subset[start:start + batch_size], F1):
            results[pair["doc_id"]] = float(f1.item())
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
    orig = [p["original"] for p in subset]
    rew = [p["rewritten"] for p in subset]
    emb_o = model.encode(orig, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    emb_r = model.encode(rew, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    sims = cos_sim(emb_o, emb_r)
    return {p["doc_id"]: float(sims[i, i].item()) for i, p in enumerate(subset)}


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def _detect_drift_threshold(scores: list[float], floor: float = 0.50) -> float:
    """Find where meaning clearly drifts (largest gap in the bottom 30 %)."""
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


def _describe_scores(scores: list[float], label: str) -> None:
    arr = np.array(scores)
    logger.info(
        "%s: n=%d  mean=%.3f  std=%.3f  min=%.3f  p5=%.3f  p10=%.3f  "
        "p25=%.3f  median=%.3f  p90=%.3f  max=%.3f",
        label, len(arr),
        arr.mean(), arr.std(), arr.min(),
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
    """Print the bottom-n pairs sorted by min(BERTScore, cosine) for inspection."""
    rows = []
    for p in pairs:
        bs = bs_by_id.get(p["doc_id"], 1.0)
        cos = cos_by_id.get(p["doc_id"], 1.0)
        rows.append((min(bs, cos), bs, cos, p))
    rows.sort(key=lambda t: t[0])
    low = rows[:n]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Bottom {n} pairs — iteration {iteration}  (sort: min(BERTScore, cosine))")
    print(sep)
    for rank, (combined, bs, cos, pair) in enumerate(low, 1):
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


def _save_iteration_report(
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    bs_target: float,
    cos_target: float,
    report: dict,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for doc_id in bs_by_id:
        rows.append({
            "doc_id": doc_id,
            "bertscore_f1": round(bs_by_id[doc_id], 5),
            "cosine_similarity": round(cos_by_id.get(doc_id, float("nan")), 5),
            "pass_bs": bs_by_id[doc_id] >= bs_target,
            "pass_cos": cos_by_id.get(doc_id, 0.0) >= cos_target,
        })
    rows.sort(key=lambda r: min(r["bertscore_f1"], r["cosine_similarity"]))
    _save_json(rows, os.path.join(out_dir, "meaning_similarity_scores.json"))

    _plot_iteration(
        list(bs_by_id.values()),
        list(cos_by_id.get(d, float("nan")) for d in bs_by_id),
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


def _plot_iteration(
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
        benchmark = report.get("benchmark", "TCB").upper()

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
                f"{benchmark} TCB — Meaning Preservation Quality",
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
# Download + initial TCB build
# ---------------------------------------------------------------------------

def _download_ramdocs(output_path: str, split: str = "test") -> None:
    import importlib
    logger.info("Downloading RAMDocs (HanNight/RAMDocs split=%s) …", split)
    datasets_mod = importlib.import_module("datasets")
    hf = datasets_mod.load_dataset("HanNight/RAMDocs", split=split)

    sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))
    from convert_ramdocs import _convert_example  # type: ignore

    from src.data_loader import save_dataset
    instances = []
    for i, row in enumerate(hf):
        try:
            instances.append(_convert_example(dict(row), i, split))
        except Exception as exc:
            logger.warning("Skipping RAMDocs row %d: %s", i, exc)
    save_dataset(instances, output_path)
    logger.info("Saved %d RAMDocs instances → %s", len(instances), output_path)


def _build_initial_tcb(
    orig_path: str,
    tcb_path: str,
    benchmark: str,
    checkpoint_every: int = 100,
    max_workers: int = 1,
    rewriter_provider: str = "gemini",
) -> None:
    from src.data_loader import load_benchmark, save_dataset
    from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

    logger.info(
        "Building initial TCB: %s → %s  (workers=%d, provider=%s)",
        orig_path, tcb_path, max_workers, rewriter_provider,
    )
    instances = load_benchmark(orig_path, benchmark)

    if max_workers <= 1:
        builder = ToneControlledBenchmarkBuilder(rewriter=ToneRewriter(provider=rewriter_provider))
        built: list = []
        for idx, inst in enumerate(instances, 1):
            try:
                tcb_inst = builder.build_instance(inst, source_benchmark=benchmark)
                if tcb_inst is not None:
                    built.append(tcb_inst)
            except Exception as exc:
                logger.warning("Skipping instance %s: %s", inst.id, exc)
            if len(built) > 0 and len(built) % checkpoint_every == 0:
                save_dataset(built, tcb_path)
                logger.info("Checkpoint: %d/%d instances", idx, len(instances))
        save_dataset(built, tcb_path)
        logger.info("Initial TCB saved: %d instances → %s", len(built), tcb_path)
        return

    # Parallel path: each worker thread gets its own builder to avoid shared
    # mutable state (ToneControlledBenchmarkBuilder.last_failure_reason).
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _local = threading.local()

    def _build_one(inst):
        if not hasattr(_local, "builder"):
            _local.builder = ToneControlledBenchmarkBuilder(
                rewriter=ToneRewriter(provider=rewriter_provider)
            )
        try:
            return inst.id, _local.builder.build_instance(inst, source_benchmark=benchmark)
        except Exception as exc:
            logger.warning("Skipping instance %s: %s", inst.id, exc)
            return inst.id, None

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

    built = [built_map[inst.id] for inst in instances if inst.id in built_map]
    save_dataset(built, tcb_path)
    logger.info("Initial TCB saved: %d instances → %s", len(built), tcb_path)


# ---------------------------------------------------------------------------
# Iterative re-write loop
# ---------------------------------------------------------------------------

def _get_failed_instance_ids(
    pairs: list[dict],
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    bs_target: float,
    cos_target: float,
) -> set[str]:
    """Return TCB instance IDs that have at least one failing document pair."""
    failed: set[str] = set()
    for p in pairs:
        bs = bs_by_id.get(p["doc_id"], 0.0)
        cos = cos_by_id.get(p["doc_id"], 0.0)
        if bs < bs_target or cos < cos_target:
            failed.add(p["tcb_instance_id"])
    return failed


# ---------------------------------------------------------------------------
# NOTE: _redo_failed_instances is disabled — iterative rewrites replaced by
# single-pass threshold scoring.  Retained here for reference; not called.
# ---------------------------------------------------------------------------
def _redo_failed_instances(
    tcb_path: str,
    orig_data: list[dict],
    failed_tcb_ids: set[str],
    benchmark: str,
    temperature: float,
    max_workers: int = 1,
) -> tuple[list[dict], set[str]]:
    """Remove failed TCB instances and re-write their source docs.

    Returns (updated_tcb_data, rebuilt_doc_ids) where rebuilt_doc_ids is the
    set of doc_ids that were just re-written (so the caller can re-score only
    those pairs).
    """
    from src.data_loader import load_benchmark, QueryInstance
    from src.tcb_builder import ToneControlledBenchmarkBuilder, ToneRewriter

    tcb_data = _load_json(tcb_path)

    # Split into passing / to-rebuild
    passing = [inst for inst in tcb_data if inst["id"] not in failed_tcb_ids]
    failed_src_ids = {
        str(inst.get("metadata", {}).get("constructed_from", ""))
        for inst in tcb_data
        if inst["id"] in failed_tcb_ids
    } - {""}

    logger.info(
        "Re-writing %d TCB instances (temperature=%.2f); keeping %d passing.",
        len(failed_tcb_ids), temperature, len(passing),
    )

    from src.data_loader import instance_from_dict
    orig_by_id = {str(inst["id"]): inst for inst in orig_data}

    def _redo_one(src_id: str) -> tuple[str, Any]:
        """Return (src_id, tcb_instance_or_None). Each thread owns its builder."""
        import threading
        if not hasattr(_redo_one, "_local"):
            _redo_one._local = threading.local()  # type: ignore[attr-defined]
        local = _redo_one._local  # type: ignore[attr-defined]
        if not hasattr(local, "builder"):
            local.builder = ToneControlledBenchmarkBuilder(
                rewriter=ToneRewriter(provider="gemini")
            )
        raw = orig_by_id.get(src_id)
        if raw is None:
            logger.warning("Source instance %s not found; dropping.", src_id)
            return src_id, None
        src_inst = instance_from_dict(raw)
        try:
            tcb_inst = local.builder.build_instance(
                src_inst, source_benchmark=benchmark, temperature=temperature
            )
            if tcb_inst is None:
                logger.warning(
                    "Instance %s still fails heuristic at temperature=%.2f: %s",
                    src_id, temperature, local.builder.last_failure_reason,
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
            _, tcb_inst = _redo_one(src_id)
            if tcb_inst is not None:
                rebuilt.append(_instance_to_raw_dict(tcb_inst))
                for doc in tcb_inst.documents:
                    rebuilt_doc_ids.add(doc.id)
            else:
                skipped += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_redo_one, sid): sid for sid in failed_src_ids}
            for fut in as_completed(futures):
                _, tcb_inst = fut.result()
                if tcb_inst is not None:
                    rebuilt.append(_instance_to_raw_dict(tcb_inst))
                    for doc in tcb_inst.documents:
                        rebuilt_doc_ids.add(doc.id)
                else:
                    skipped += 1

    logger.info(
        "Re-write results: rebuilt=%d  skipped/dropped=%d", len(rebuilt), skipped
    )

    merged = passing + rebuilt
    _save_json(merged, tcb_path)
    return merged, rebuilt_doc_ids


def _instance_to_raw_dict(inst: Any) -> dict:
    from src.data_loader import instance_to_dict
    return instance_to_dict(inst)


# ---------------------------------------------------------------------------
# Main quality loop for a single dataset
# ---------------------------------------------------------------------------

def _validate_dataset(
    tcb_path: str,
    orig_path: str,
    benchmark: str,
    bs_target: float,
    cos_target: float,
    batch_size: int,
    device: str | None,
    out_dir: str,
    bs_scorer: Any,
    cos_model: Any,
    n_low_end: int,
) -> dict:
    """Single scoring pass for one dataset: score, inspect, drop failures below threshold.

    No iterative rewrites.  Set --bs-target / --cosine-target from the printed distribution.
    """
    os.makedirs(out_dir, exist_ok=True)
    orig_data = _load_json(orig_path)

    tcb_data = _load_json(tcb_path)
    scored_n_instances = len(tcb_data)
    pairs = _build_pairs(tcb_data, orig_data)
    logger.info("Dataset: %s  |  %d TCB instances, %d doc pairs", benchmark, len(tcb_data), len(pairs))

    bs_cache: dict[str, float] = _score_bertscore(pairs, bs_scorer, batch_size)
    cos_cache: dict[str, float] = _score_cosine(pairs, cos_model, batch_size)

    bs_vals = [bs_cache.get(p["doc_id"], 0.0) for p in pairs]
    cos_vals = [cos_cache.get(p["doc_id"], 0.0) for p in pairs]

    print(f"\n{'─' * 60}")
    print(f"  {benchmark.upper()}  — scoring pass")
    print(f"{'─' * 60}")
    _describe_scores(bs_vals, "BERTScore F1 (rescaled)")
    _describe_scores(cos_vals, "Cosine similarity")

    bs_drift = _detect_drift_threshold(bs_vals)
    cos_drift = _detect_drift_threshold(cos_vals)
    print(f"\n  Auto-detected drift  →  BERTScore: {bs_drift:.4f}   Cosine: {cos_drift:.4f}")
    print(f"  Configured threshold →  BERTScore: {bs_target:.4f}   Cosine: {cos_target:.4f}")

    _print_low_end(pairs, bs_cache, cos_cache, n_low_end, 1)

    fail_bs = sum(1 for v in bs_vals if v < bs_target)
    fail_cos = sum(1 for v in cos_vals if v < cos_target)
    failed_ids = _get_failed_instance_ids(pairs, bs_cache, cos_cache, bs_target, cos_target)
    fail_either = len(failed_ids)
    pass_rate_bs = 1.0 - fail_bs / max(len(bs_vals), 1)
    pass_rate_cos = 1.0 - fail_cos / max(len(cos_vals), 1)

    print(f"\n  Pass rate  →  BERTScore: {pass_rate_bs:.1%} ({fail_bs} fail)   "
          f"Cosine: {pass_rate_cos:.1%} ({fail_cos} fail)   "
          f"Instances failing either: {fail_either}")

    # Drop instances below threshold
    if failed_ids:
        print(f"\n  Dropping {fail_either} instance(s) below threshold (no rewrites attempted).")
        kept = [inst for inst in tcb_data if inst["id"] not in failed_ids]
        _save_json(kept, tcb_path)
        logger.info("Dropped %d instances; final count: %d", len(failed_ids), len(kept))
        tcb_data = kept

    final_pairs = _build_pairs(tcb_data, orig_data)
    final_bs = [bs_cache.get(p["doc_id"], 0.0) for p in final_pairs if p["doc_id"] in bs_cache]
    final_cos = [cos_cache.get(p["doc_id"], 0.0) for p in final_pairs if p["doc_id"] in cos_cache]

    summary = {
        "benchmark": benchmark,
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
    _save_iteration_report(bs_cache, cos_cache, bs_target, cos_target, summary, out_dir)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download, build, and score TCB datasets — single pass, no iterative rewrites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data/full",
                   help="Root directory for all downloaded and built files.")
    p.add_argument("--bs-target", type=float, default=0.70,
                   help=(
                       "Drop threshold for rescaled BERTScore F1 (roberta-large).  "
                       "Inspect the printed distribution first — Wikipedia prose typically "
                       "scores 0.60–0.78; set around 0.65–0.70 where meaning clearly drifts."
                   ))
    p.add_argument("--cosine-target", type=float, default=0.90,
                   help="Drop threshold for cosine similarity (all-mpnet-base-v2).")
    p.add_argument("--batch-size", type=int, default=32,
                   help=(
                       "Scoring batch size. Default 32. "
                       "Recommended: 128 for mps/cuda, 32 for cpu."
                   ))
    p.add_argument("--device", default=None,
                   help=(
                       'Torch device for quality scoring (e.g. "mps", "cuda:0", "cpu"). '
                       'Auto-detected (mps > cuda > cpu) when omitted.'
                   ))
    p.add_argument("--rewrite-workers", type=int, default=1,
                   help=(
                       "Parallel API workers for the initial TCB build only.  "
                       "Not used during scoring.  Use 4-8 to saturate provider rate limits."
                   ))
    p.add_argument("--n-low-end", type=int, default=_N_LOW_END,
                   help="Number of low-scoring pairs printed each iteration for inspection.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip HuggingFace downloads (files must already exist).")
    p.add_argument("--skip-initial-build", action="store_true",
                   help="Skip initial TCB build (TCB files must already exist).")
    p.add_argument("--skip-ramdocs", action="store_true", help="Skip RAMDocs entirely.")
    p.add_argument("--skip-confact", action="store_true",
                   help=(
                       "Skip ConfACT entirely.  "
                       "Note: ConfACT has no HF download step — the source JSON must be "
                       "generated first with: python experiments/clean_confact_v2.py"
                   ))
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Save TCB checkpoint every N instances during initial build.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)

    datasets_cfg = []
    if not args.skip_ramdocs:
        datasets_cfg.append({
            "name": "ramdocs",
            "benchmark": "ramdocs",
            "orig_path": os.path.join(data_root, "ramdocs_test.json"),
            "tcb_path": os.path.join(data_root, "tcb_ramdocs_test.json"),
            "download_fn": lambda p: _download_ramdocs(p, split="test"),
            "rewriter_provider": "gemini",
            "quality_dir": os.path.join(data_root, "quality_ramdocs"),
        })
    if not args.skip_confact:
        confact_dir = os.path.join(data_root, "confact")
        datasets_cfg.append({
            "name": "confact",
            "benchmark": "confact",
            "orig_path": os.path.join(confact_dir, "humc_processed_cleaned.json"),
            "tcb_path": os.path.join(confact_dir, "tcb_humc_processed_cleaned.json"),
            "download_fn": None,  # no HF source; generate with clean_confact_v2.py
            "rewriter_provider": "anthropic",
            "quality_dir": os.path.join(confact_dir, "quality"),
        })
    if not datasets_cfg:
        print("Nothing to do (all datasets skipped).")
        return

    # Auto-detect device when not explicitly specified
    if args.device is None:
        args.device = _auto_device()
    logger.info(
        "Device: %s  |  Scoring batch-size: %d  |  Build workers: %d",
        args.device, args.batch_size, args.rewrite_workers,
    )

    t0 = time.time()

    # ---------------------------------------------------------------
    # Phase 1: Download
    # ---------------------------------------------------------------
    if not args.skip_download:
        print("\n" + "=" * 70)
        print("  PHASE 1: Downloading datasets from Hugging Face")
        print("=" * 70)
        for cfg in datasets_cfg:
            if cfg["download_fn"] is None:
                if os.path.exists(cfg["orig_path"]):
                    n = len(_load_json(cfg["orig_path"]))
                    logger.info(
                        "Found existing %s (%d instances); no HF download needed.",
                        cfg["orig_path"], n,
                    )
                else:
                    logger.warning(
                        "No HF source for %s and file not found: %s\n"
                        "  Generate it with: python experiments/clean_confact_v2.py "
                        "--output %s",
                        cfg["name"], cfg["orig_path"], cfg["orig_path"],
                    )
            elif not os.path.exists(cfg["orig_path"]):
                cfg["download_fn"](cfg["orig_path"])
            else:
                n = len(_load_json(cfg["orig_path"]))
                logger.info("Found existing %s (%d instances); skipping download.", cfg["orig_path"], n)
    else:
        logger.info("--skip-download: assuming source files exist.")

    # ---------------------------------------------------------------
    # Phase 2: Initial TCB build
    # ---------------------------------------------------------------
    if not args.skip_initial_build:
        print("\n" + "=" * 70)
        print("  PHASE 2: Building initial TCB datasets")
        print("=" * 70)
        for cfg in datasets_cfg:
            if not os.path.exists(cfg["orig_path"]):
                logger.warning(
                    "Source file missing for %s: %s — skipping TCB build.",
                    cfg["name"], cfg["orig_path"],
                )
                continue
            if not os.path.exists(cfg["tcb_path"]):
                _build_initial_tcb(
                    cfg["orig_path"],
                    cfg["tcb_path"],
                    cfg["benchmark"],
                    checkpoint_every=args.checkpoint_every,
                    max_workers=args.rewrite_workers,
                    rewriter_provider=cfg.get("rewriter_provider", "gemini"),
                )
            else:
                n = len(_load_json(cfg["tcb_path"]))
                logger.info("Found existing %s (%d instances); skipping initial build.", cfg["tcb_path"], n)
    else:
        logger.info("--skip-initial-build: assuming TCB files exist.")

    # ---------------------------------------------------------------
    # Phase 3: Load scoring models (once, shared across datasets)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3: Loading scoring models")
    print("=" * 70)
    bs_scorer = _load_bertscore(args.device)
    cos_model = _load_sbert(args.device)

    # ---------------------------------------------------------------
    # Phase 4: Single scoring pass + threshold drop
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  PHASE 4: Scoring pass (single pass, no iterative rewrites)")
    print(f"  Drop thresholds: BERTScore F1 ≥ {args.bs_target}  |  Cosine ≥ {args.cosine_target}")
    print("=" * 70)

    all_summaries = []
    for cfg in datasets_cfg:
        if not os.path.exists(cfg["tcb_path"]):
            logger.warning("TCB file missing for %s: %s — skipping scoring.", cfg["name"], cfg["tcb_path"])
            continue
        print(f"\n{'█' * 70}")
        print(f"  Dataset: {cfg['name'].upper()}")
        print(f"{'█' * 70}")
        summary = _validate_dataset(
            tcb_path=cfg["tcb_path"],
            orig_path=cfg["orig_path"],
            benchmark=cfg["benchmark"],
            bs_target=args.bs_target,
            cos_target=args.cosine_target,
            batch_size=args.batch_size,
            device=args.device,
            out_dir=cfg["quality_dir"],
            bs_scorer=bs_scorer,
            cos_model=cos_model,
            n_low_end=args.n_low_end,
        )
        all_summaries.append(summary)

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    report_path = os.path.join(data_root, "quality_report.json")
    report = {
        "bs_target": args.bs_target,
        "cosine_target": args.cosine_target,
        "total_elapsed_seconds": round(elapsed, 1),
        "datasets": all_summaries,
    }
    _save_json(report, report_path)

    print("\n" + "=" * 70)
    print("  FINAL QUALITY REPORT")
    print("=" * 70)
    for s in all_summaries:
        print(f"\n  {s['benchmark'].upper()}")
        print(f"    Final instances  : {s['final_n_instances']}")
        print(f"    Final doc pairs  : {s['final_n_pairs']}")
        print(f"    Pass rate BS     : {s['final_pass_rate_bs']:.1%}  (target ≥ {s['bs_target']})")
        print(f"    Pass rate Cosine : {s['final_pass_rate_cos']:.1%}  (target ≥ {s['cosine_target']})")
        print(f"    Mean BS          : {s['final_mean_bs']:.4f}")
        print(f"    Mean Cosine      : {s['final_mean_cos']:.4f}")
        print(f"    Min BS           : {s['final_min_bs']:.4f}")
        print(f"    Min Cosine       : {s['final_min_cos']:.4f}")
        print(f"    Dropped instances: {s['dropped_instances']}")
        print(f"    Output file      : {s['tcb_path']}")

    print(f"\n  Total elapsed: {elapsed/60:.1f} min")
    print(f"  Quality report: {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
