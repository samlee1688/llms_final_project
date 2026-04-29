#!/usr/bin/env python3
"""Compute rescaled BERTScore F1 on all TCB document pairs and validate thresholds.

Workflow
--------
1. Load every (original_text, rewritten_text) pair from a TCB dataset and its
   corresponding source dataset.
2. Compute BERTScore F1 (rescaled) using roberta-large in batches.
3. Plot and describe the score distribution.
4. Print the 15 lowest-scoring pairs so you can manually inspect where meaning
   clearly drifts.
5. Report a suggested threshold derived from the distribution, print it
   explicitly, and show what fraction of pairs pass.
6. Save per-pair scores and a list of failed instance IDs.
7. Optionally (--fix-failed): remove failed TCB instances, re-run the Gemini
   rewriter on their source instances, and merge the new results back.

Usage
-----
    python experiments/analyze_tcb_bertscore.py \
        --tcb-file  datasets/full/tcb_ramdocs_test.json \
        --orig-file datasets/full/ramdocs_test.json \
        --output-dir results/quality_analysis/ramdocs

    # With auto-fix for pairs below threshold:
    python experiments/analyze_tcb_bertscore.py \
        --tcb-file  datasets/full/tcb_ramdocs_test.json \
        --orig-file datasets/full/ramdocs_test.json \
        --output-dir results/quality_analysis/ramdocs \
        --fix-failed

    # Override threshold (e.g. after visual inspection):
    python experiments/analyze_tcb_bertscore.py ... --threshold 0.60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.quality_checker import BERTSCORE_THRESHOLD

logger = logging.getLogger(__name__)

# Number of low-scoring pairs printed for visual inspection
N_LOW_END = 15
# How many chars to show per document in the inspection table
PREVIEW_CHARS = 350
# Default batch size for BERTScore; reduce if you hit OOM
DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_pairs(tcb_data: list[dict], orig_data: list[dict]) -> list[dict[str, Any]]:
    """Return a list of (original_text, rewritten_text, metadata) dicts.

    Each entry has keys:
        tcb_instance_id, orig_instance_id, doc_id, tone,
        original, rewritten
    """
    orig_by_id: dict[str, dict] = {}
    for inst in orig_data:
        orig_by_id[str(inst["id"])] = inst

    pairs: list[dict] = []
    missing_source = 0
    missing_doc = 0

    for tcb_inst in tcb_data:
        src_id = str(tcb_inst.get("metadata", {}).get("constructed_from", ""))
        orig_inst = orig_by_id.get(src_id)
        if orig_inst is None:
            missing_source += 1
            continue

        orig_docs_by_id = {str(d["id"]): d for d in orig_inst.get("documents", [])}

        for doc in tcb_inst.get("documents", []):
            orig_doc_id = str(doc.get("metadata", {}).get("original_doc_id", ""))
            orig_doc = orig_docs_by_id.get(orig_doc_id)
            if orig_doc is None:
                missing_doc += 1
                continue
            orig_text = orig_doc.get("content", "").strip()
            rew_text = doc.get("content", "").strip()
            if not orig_text or not rew_text:
                continue
            pairs.append(
                {
                    "tcb_instance_id": tcb_inst["id"],
                    "orig_instance_id": src_id,
                    "doc_id": doc["id"],
                    "tone": doc.get("tone", ""),
                    "original": orig_text,
                    "rewritten": rew_text,
                }
            )

    if missing_source:
        logger.warning("%d TCB instances had no matching source instance", missing_source)
    if missing_doc:
        logger.warning("%d TCB documents had no matching original document", missing_doc)

    return pairs


# ---------------------------------------------------------------------------
# BERTScore computation
# ---------------------------------------------------------------------------

def compute_bertscore(
    pairs: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> list[float]:
    """Compute rescaled BERTScore F1 (roberta-large) for each pair."""
    from bert_score import BERTScorer  # type: ignore

    logger.info("Loading BERTScorer (roberta-large, rescale_with_baseline=True) …")
    # lang="en" required for rescale_with_baseline=True (bert-score>=0.3.12).
    # verbose is not a valid BERTScorer parameter.
    scorer = BERTScorer(
        model_type="roberta-large",
        rescale_with_baseline=True,
        lang="en",
        device=device,
    )

    originals = [p["original"] for p in pairs]
    rewrittens = [p["rewritten"] for p in pairs]

    all_f1: list[float] = []
    total_batches = (len(originals) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(originals), batch_size), 1):
        end = start + batch_size
        b_orig = originals[start:end]
        b_rew = rewrittens[start:end]
        _, _, F1 = scorer.score(b_rew, b_orig)
        all_f1.extend(float(x.item()) for x in F1)
        logger.info("BERTScore batch %d/%d done", batch_idx, total_batches)

    return all_f1


# ---------------------------------------------------------------------------
# Distribution analysis helpers
# ---------------------------------------------------------------------------

def _describe_distribution(scores: list[float]) -> None:
    arr = np.array(scores)
    print("\n" + "=" * 60)
    print("BERTScore F1 (rescaled, roberta-large) — Distribution")
    print("=" * 60)
    print(f"  N pairs    : {len(arr)}")
    print(f"  Mean       : {arr.mean():.4f}")
    print(f"  Std        : {arr.std():.4f}")
    print(f"  Min        : {arr.min():.4f}")
    print(f"  p5         : {np.percentile(arr, 5):.4f}")
    print(f"  p10        : {np.percentile(arr, 10):.4f}")
    print(f"  p25        : {np.percentile(arr, 25):.4f}")
    print(f"  Median     : {np.median(arr):.4f}")
    print(f"  p75        : {np.percentile(arr, 75):.4f}")
    print(f"  p90        : {np.percentile(arr, 90):.4f}")
    print(f"  Max        : {arr.max():.4f}")
    print("=" * 60)


def _suggest_threshold(scores: list[float]) -> float:
    """Suggest a threshold from the distribution.

    Strategy:
    1. Look at the sorted bottom 30 % for the largest gap (>= 0.03) — this
       often corresponds to a natural split between rewrites that drifted and
       those that didn't.
    2. Fall back to the 5th-percentile value.
    3. Never go below 0.50 (anything lower signals fundamental failure).
    """
    arr = np.array(sorted(scores))
    cutoff_idx = max(1, int(len(arr) * 0.30))

    best_gap = 0.0
    best_mid = float(np.percentile(arr, 5))

    for i in range(cutoff_idx - 1):
        gap = float(arr[i + 1] - arr[i])
        if gap > best_gap:
            best_gap = gap
            best_mid = float((arr[i] + arr[i + 1]) / 2)

    # Only use the gap if it's meaningfully large
    if best_gap < 0.03:
        best_mid = float(np.percentile(arr, 5))

    return max(0.50, round(best_mid, 3))


def _print_low_end_pairs(
    pairs: list[dict],
    scores: list[float],
    n: int = N_LOW_END,
) -> None:
    ranked = sorted(zip(scores, pairs), key=lambda t: t[0])
    low = ranked[:n]

    print(f"\n{'=' * 70}")
    print(f"Bottom {n} pairs by BERTScore F1 (manual inspection)")
    print(f"{'=' * 70}")

    for rank, (score, pair) in enumerate(low, 1):
        orig_preview = pair["original"].replace("\n", " ")[:PREVIEW_CHARS]
        rew_preview = pair["rewritten"].replace("\n", " ")[:PREVIEW_CHARS]
        print(f"\n[{rank:2d}]  score={score:.4f}  tone={pair['tone']}")
        print(f"     doc_id   : {pair['doc_id']}")
        print(f"     ORIGINAL : {orig_preview}")
        if len(pair["original"]) > PREVIEW_CHARS:
            print(f"              … (+{len(pair['original']) - PREVIEW_CHARS} chars)")
        print(f"     REWRITTEN: {rew_preview}")
        if len(pair["rewritten"]) > PREVIEW_CHARS:
            print(f"              … (+{len(pair['rewritten']) - PREVIEW_CHARS} chars)")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_distribution(
    scores: list[float],
    threshold: float,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    arr = np.array(scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram ---
    ax = axes[0]
    ax.hist(arr, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="crimson", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("BERTScore F1 (rescaled)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("TCB Pair Score Distribution", fontsize=13)
    ax.legend(fontsize=11)

    fail_frac = float((arr < threshold).mean())
    ax.text(
        0.03, 0.97,
        f"N={len(arr)}\nFail rate = {fail_frac:.1%}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    # --- CDF ---
    ax2 = axes[1]
    x_sorted = np.sort(arr)
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax2.plot(x_sorted, cdf, color="steelblue", linewidth=1.5)
    ax2.axvline(threshold, color="crimson", linewidth=2, linestyle="--",
                label=f"Threshold = {threshold:.3f}")
    ax2.axhline(fail_frac, color="orange", linewidth=1.2, linestyle=":",
                label=f"Fail fraction = {fail_frac:.1%}")
    ax2.set_xlabel("BERTScore F1 (rescaled)", fontsize=12)
    ax2.set_ylabel("Cumulative fraction", fontsize=12)
    ax2.set_title("Cumulative Distribution", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35)

    fig.suptitle("TCB Rewrite Quality: Rescaled BERTScore F1 (roberta-large)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] Saved distribution plot → {output_path}")


# ---------------------------------------------------------------------------
# Fix-failed: re-run TCB building for instances below threshold
# ---------------------------------------------------------------------------

def _fix_failed(
    tcb_file: str,
    orig_file: str,
    failed_tcb_ids: set[str],
    output_dir: str,
    rewriter_provider: str = "gemini",
) -> None:
    """Remove failed TCB instances, re-run Gemini rewriter, merge results back."""
    print(f"\n{'=' * 60}")
    print(f"FIX-FAILED: re-writing {len(failed_tcb_ids)} failed instances …")
    print(f"{'=' * 60}")

    # --- Load current TCB and split into passing / failed ---
    tcb_data = _load_dataset(tcb_file)
    passing = [inst for inst in tcb_data if inst["id"] not in failed_tcb_ids]
    failed_src_ids = {
        str(inst.get("metadata", {}).get("constructed_from", ""))
        for inst in tcb_data
        if inst["id"] in failed_tcb_ids
    } - {""}

    print(f"  Keeping    {len(passing)} passing TCB instances.")
    print(f"  Removing   {len(tcb_data) - len(passing)} failed TCB instances.")
    print(f"  Re-writing {len(failed_src_ids)} source instances.")

    # --- Build source subset ---
    orig_data = _load_dataset(orig_file)
    failed_source_instances = [
        inst for inst in orig_data if str(inst["id"]) in failed_src_ids
    ]
    failed_source_path = os.path.join(output_dir, "_failed_source_instances.json")
    with open(failed_source_path, "w", encoding="utf-8") as fh:
        json.dump(failed_source_instances, fh, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(failed_source_instances)} source instances → {failed_source_path}")

    # --- Re-run TCB builder on failed instances (no resume so they're rebuilt) ---
    retry_tcb_path = os.path.join(output_dir, "_tcb_retry.json")
    build_cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "experiments", "build_tcb.py"),
        "--input", failed_source_path,
        "--output", retry_tcb_path,
        "--rewriter-provider", rewriter_provider,
        "--no-resume",
        "--checkpoint-every", "50",
        "--verbose",
    ]
    print(f"\n  Running: {' '.join(build_cmd)}\n")
    subprocess.run(build_cmd, check=True)

    # --- Merge rebuilt instances back ---
    retry_data = _load_dataset(retry_tcb_path) if os.path.exists(retry_tcb_path) else []
    merged = passing + retry_data

    with open(tcb_file, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)

    print(f"\n  Merged {len(retry_data)} re-built instances into {tcb_file}")
    print(f"  Final TCB size: {len(merged)} instances")

    # Clean up temp files
    for tmp in [failed_source_path, retry_tcb_path]:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_results(
    pairs: list[dict],
    scores: list[float],
    threshold: float,
    output_dir: str,
) -> tuple[set[str], str]:
    os.makedirs(output_dir, exist_ok=True)

    # Per-pair scores JSON
    per_pair = [
        {
            "tcb_instance_id": p["tcb_instance_id"],
            "doc_id": p["doc_id"],
            "tone": p["tone"],
            "bertscore_f1": round(s, 6),
            "pass": s >= threshold,
        }
        for p, s in zip(pairs, scores)
    ]
    scores_path = os.path.join(output_dir, "bertscore_per_pair.json")
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(per_pair, fh, ensure_ascii=False, indent=2)
    print(f"[saved] Per-pair scores → {scores_path}")

    # Failed instance IDs
    failed_ids = {
        p["tcb_instance_id"]
        for p, s in zip(pairs, scores)
        if s < threshold
    }
    failed_path = os.path.join(output_dir, "bertscore_failed_ids.json")
    with open(failed_path, "w", encoding="utf-8") as fh:
        json.dump(sorted(failed_ids), fh, ensure_ascii=False, indent=2)
    print(f"[saved] Failed instance IDs → {failed_path}")

    # Summary JSON
    import numpy as _np
    arr = _np.array(scores)
    summary = {
        "n_pairs": len(scores),
        "n_failed": int((arr < threshold).sum()),
        "pass_rate": float((arr >= threshold).mean()),
        "threshold": threshold,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p5": float(_np.percentile(arr, 5)),
        "p10": float(_np.percentile(arr, 10)),
        "median": float(_np.median(arr)),
        "p90": float(_np.percentile(arr, 90)),
        "max": float(arr.max()),
    }
    summary_path = os.path.join(output_dir, "bertscore_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[saved] Summary → {summary_path}")

    return failed_ids, failed_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute rescaled BERTScore F1 on TCB pairs and validate thresholds."
    )
    p.add_argument("--tcb-file", required=True, help="TCB dataset JSON file.")
    p.add_argument("--orig-file", required=True, help="Original (pre-TCB) source dataset JSON.")
    p.add_argument(
        "--output-dir",
        default="results/quality_analysis",
        help="Directory for plots, score JSONs, and failed-ID lists.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"BERTScore F1 threshold (default: auto-suggest from distribution; "
            f"configured default is {BERTSCORE_THRESHOLD})."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"BERTScore batch size (default: {DEFAULT_BATCH_SIZE}). Reduce if OOM.",
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            'Torch device string. Default: auto-detected (mps > cuda > cpu). '
            'Recommended batch sizes: mps/cuda → 128, cpu → 32.'
        ),
    )
    p.add_argument(
        "--fix-failed",
        action="store_true",
        help="After analysis, remove failed TCB instances and re-run Gemini rewriter.",
    )
    p.add_argument(
        "--rewriter-provider",
        default="gemini",
        choices=["gemini", "local"],
        help="Rewriter to use when --fix-failed is set.",
    )
    p.add_argument(
        "--n-low-end",
        type=int,
        default=N_LOW_END,
        help=f"Number of low-end pairs to print for inspection (default: {N_LOW_END}).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    if args.device is None:
        args.device = _auto_device()
    logger.info("Using device: %s  batch-size: %d", args.device, args.batch_size)

    # ------------------------------------------------------------------
    # 1. Load data and build pairs
    # ------------------------------------------------------------------
    print(f"\nLoading TCB data from  : {args.tcb_file}")
    print(f"Loading source data from: {args.orig_file}")
    tcb_data = _load_dataset(args.tcb_file)
    orig_data = _load_dataset(args.orig_file)

    pairs = _build_pairs(tcb_data, orig_data)
    if not pairs:
        print("ERROR: No matching (original, rewritten) pairs found. "
              "Check that --tcb-file and --orig-file correspond to the same benchmark.")
        sys.exit(1)
    print(f"Built {len(pairs)} document pairs from {len(tcb_data)} TCB instances.")

    # ------------------------------------------------------------------
    # 2. Compute BERTScore
    # ------------------------------------------------------------------
    print(f"\nComputing rescaled BERTScore F1 (roberta-large) on {len(pairs)} pairs …")
    scores = compute_bertscore(pairs, batch_size=args.batch_size, device=args.device)

    # ------------------------------------------------------------------
    # 3. Distribution summary
    # ------------------------------------------------------------------
    _describe_distribution(scores)

    # ------------------------------------------------------------------
    # 4. Print low-end pairs for manual inspection
    # ------------------------------------------------------------------
    _print_low_end_pairs(pairs, scores, n=args.n_low_end)

    # ------------------------------------------------------------------
    # 5. Determine threshold
    # ------------------------------------------------------------------
    if args.threshold is not None:
        threshold = args.threshold
        source = "user-specified"
    else:
        threshold = _suggest_threshold(scores)
        source = "auto-suggested from distribution"

    import numpy as np
    arr = np.array(scores)
    pass_rate = float((arr >= threshold).mean())
    fail_count = int((arr < threshold).sum())

    print(f"\n{'=' * 60}")
    print(f"THRESHOLD ({source}): {threshold:.4f}")
    print(f"  Pass rate : {pass_rate:.1%}  ({len(scores) - fail_count}/{len(scores)} pairs)")
    print(f"  Fail count: {fail_count} pairs below threshold")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 6. Save results and plot
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    failed_ids, _ = _save_results(pairs, scores, threshold, args.output_dir)

    plot_path = os.path.join(args.output_dir, "bertscore_distribution.png")
    try:
        _plot_distribution(scores, threshold, plot_path)
    except Exception as exc:
        logger.warning("Plotting failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # 7. Optionally fix failed instances
    # ------------------------------------------------------------------
    if args.fix_failed and failed_ids:
        print(f"\n--fix-failed requested: {len(failed_ids)} instance(s) below threshold.")
        _fix_failed(
            tcb_file=args.tcb_file,
            orig_file=args.orig_file,
            failed_tcb_ids=failed_ids,
            output_dir=args.output_dir,
            rewriter_provider=args.rewriter_provider,
        )
    elif args.fix_failed and not failed_ids:
        print("\n--fix-failed: all pairs pass threshold, nothing to fix.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset        : {args.tcb_file}")
    print(f"  Total pairs    : {len(scores)}")
    print(f"  Threshold      : {threshold:.4f}  ({source})")
    print(f"  Pass rate      : {pass_rate:.1%}")
    print(f"  Failed pairs   : {fail_count}")
    print(f"  Output dir     : {args.output_dir}")
    if fail_count > 0:
        print(
            f"\n  ACTION NEEDED: {fail_count} pair(s) below threshold. "
            "Re-run with --fix-failed to rebuild them, or "
            "inspect the low-end pairs above and adjust --threshold."
        )
    else:
        print("\n  All pairs pass the threshold. No action needed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
