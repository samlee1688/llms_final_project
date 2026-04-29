#!/usr/bin/env python3
"""Sanity-test TCB rewrite quality using sentence-embedding cosine similarity.

This script is the companion to analyze_tcb_bertscore.py.  While BERTScore
captures token-level similarity, sentence embeddings capture global semantic
similarity.  Running both provides cross-validation:

  - High BERTScore + High cosine → clear meaning preservation
  - High BERTScore + Low cosine  → style preserved but topic may have drifted
  - Low BERTScore + High cosine  → heavy paraphrase (acceptable) or superficial
  - Low BERTScore + Low cosine   → meaning likely lost

Workflow
--------
1. Load (original, rewritten) pairs from the TCB + source datasets.
2. Compute cosine similarity using sentence-transformers (all-mpnet-base-v2).
3. Plot and describe the distribution.
4. Print the 15 lowest-scoring pairs for manual inspection.
5. Report the pass rate at the BERTScore-calibrated threshold and at the
   cosine-specific threshold.
6. If --bertscore-results is given, produce a cross-validation scatter plot
   (BERTScore vs cosine) and print concordance statistics.
7. Save per-pair scores and summary.

Usage
-----
    python experiments/analyze_tcb_sentence_similarity.py \
        --tcb-file  datasets/full/tcb_ramdocs_test.json \
        --orig-file datasets/full/ramdocs_test.json \
        --output-dir results/quality_analysis/ramdocs

    # With cross-validation against BERTScore results:
    python experiments/analyze_tcb_sentence_similarity.py \
        --tcb-file  datasets/full/tcb_ramdocs_test.json \
        --orig-file datasets/full/ramdocs_test.json \
        --output-dir results/quality_analysis/ramdocs \
        --bertscore-results results/quality_analysis/ramdocs/bertscore_per_pair.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.quality_checker import BERTSCORE_THRESHOLD, COSINE_SIM_THRESHOLD

logger = logging.getLogger(__name__)

N_LOW_END = 15
PREVIEW_CHARS = 350
DEFAULT_BATCH_SIZE = 64
EMBEDDING_MODEL = "all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Data loading (shared logic with BERTScore script)
# ---------------------------------------------------------------------------

def _load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_pairs(tcb_data: list[dict], orig_data: list[dict]) -> list[dict[str, Any]]:
    orig_by_id = {str(inst["id"]): inst for inst in orig_data}
    pairs: list[dict] = []

    for tcb_inst in tcb_data:
        src_id = str(tcb_inst.get("metadata", {}).get("constructed_from", ""))
        orig_inst = orig_by_id.get(src_id)
        if orig_inst is None:
            continue
        orig_docs_by_id = {str(d["id"]): d for d in orig_inst.get("documents", [])}
        for doc in tcb_inst.get("documents", []):
            orig_doc_id = str(doc.get("metadata", {}).get("original_doc_id", ""))
            orig_doc = orig_docs_by_id.get(orig_doc_id)
            if orig_doc is None:
                continue
            orig_text = orig_doc.get("content", "").strip()
            rew_text = doc.get("content", "").strip()
            if not orig_text or not rew_text:
                continue
            pairs.append(
                {
                    "tcb_instance_id": tcb_inst["id"],
                    "doc_id": doc["id"],
                    "tone": doc.get("tone", ""),
                    "original": orig_text,
                    "rewritten": rew_text,
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Cosine similarity computation
# ---------------------------------------------------------------------------

def compute_cosine_similarity(
    pairs: list[dict],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> list[float]:
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sentence_transformers.util import cos_sim  # type: ignore

    logger.info("Loading SentenceTransformer model=%s …", model_name)
    kwargs: dict = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)

    originals = [p["original"] for p in pairs]
    rewrittens = [p["rewritten"] for p in pairs]

    logger.info("Encoding %d original texts …", len(originals))
    embs_orig = model.encode(originals, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    logger.info("Encoding %d rewritten texts …", len(rewrittens))
    embs_rew = model.encode(rewrittens, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

    sims = cos_sim(embs_orig, embs_rew)  # (N, N) matrix
    return [float(sims[i, i].item()) for i in range(len(pairs))]


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _describe_distribution(scores: list[float], label: str = "Cosine Similarity") -> None:
    arr = np.array(scores)
    print("\n" + "=" * 60)
    print(f"{label} — Distribution")
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


def _print_low_end_pairs(
    pairs: list[dict],
    scores: list[float],
    n: int = N_LOW_END,
    label: str = "Cosine Similarity",
) -> None:
    ranked = sorted(zip(scores, pairs), key=lambda t: t[0])
    low = ranked[:n]

    print(f"\n{'=' * 70}")
    print(f"Bottom {n} pairs by {label} (manual inspection)")
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
    cosine_scores: list[float],
    cosine_threshold: float,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    arr = np.array(cosine_scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(arr, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
    ax.axvline(cosine_threshold, color="crimson", linewidth=2, linestyle="--",
               label=f"Threshold = {cosine_threshold:.3f}")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("TCB Pair Score Distribution", fontsize=13)
    ax.legend(fontsize=11)
    fail_frac = float((arr < cosine_threshold).mean())
    ax.text(
        0.03, 0.97,
        f"N={len(arr)}\nFail rate = {fail_frac:.1%}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax2 = axes[1]
    x_sorted = np.sort(arr)
    cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax2.plot(x_sorted, cdf, color="darkorange", linewidth=1.5)
    ax2.axvline(cosine_threshold, color="crimson", linewidth=2, linestyle="--",
                label=f"Threshold = {cosine_threshold:.3f}")
    ax2.axhline(fail_frac, color="steelblue", linewidth=1.2, linestyle=":",
                label=f"Fail fraction = {fail_frac:.1%}")
    ax2.set_xlabel("Cosine Similarity", fontsize=12)
    ax2.set_ylabel("Cumulative fraction", fontsize=12)
    ax2.set_title("Cumulative Distribution", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35)

    fig.suptitle(
        f"TCB Rewrite Quality: Cosine Similarity ({EMBEDDING_MODEL})", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] Saved distribution plot → {output_path}")


def _plot_crossval(
    cosine_scores: list[float],
    bertscore_scores: list[float],
    cosine_threshold: float,
    bertscore_threshold: float,
    output_path: str,
) -> None:
    """Scatter plot: BERTScore vs cosine similarity with quadrant lines."""
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    c_arr = np.array(cosine_scores)
    b_arr = np.array(bertscore_scores)

    # Quadrant labels
    both_pass = (c_arr >= cosine_threshold) & (b_arr >= bertscore_threshold)
    only_cos = (c_arr >= cosine_threshold) & (b_arr < bertscore_threshold)
    only_bs = (c_arr < cosine_threshold) & (b_arr >= bertscore_threshold)
    both_fail = (c_arr < cosine_threshold) & (b_arr < bertscore_threshold)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {
        "both_pass": "steelblue",
        "only_cos": "orange",
        "only_bs": "mediumpurple",
        "both_fail": "crimson",
    }

    ax.scatter(b_arr[both_pass], c_arr[both_pass], s=8, alpha=0.5,
               color=colors["both_pass"], label=f"Both pass ({both_pass.sum()})")
    ax.scatter(b_arr[only_cos], c_arr[only_cos], s=8, alpha=0.5,
               color=colors["only_cos"], label=f"Only cosine passes ({only_cos.sum()})")
    ax.scatter(b_arr[only_bs], c_arr[only_bs], s=8, alpha=0.5,
               color=colors["only_bs"], label=f"Only BERTScore passes ({only_bs.sum()})")
    ax.scatter(b_arr[both_fail], c_arr[both_fail], s=8, alpha=0.7,
               color=colors["both_fail"], label=f"Both fail ({both_fail.sum()})")

    ax.axvline(bertscore_threshold, color="grey", linewidth=1.2, linestyle="--",
               label=f"BERTScore thr={bertscore_threshold:.3f}")
    ax.axhline(cosine_threshold, color="black", linewidth=1.2, linestyle="--",
               label=f"Cosine thr={cosine_threshold:.3f}")

    # Pearson correlation
    r = float(np.corrcoef(b_arr, c_arr)[0, 1])
    ax.text(
        0.03, 0.97,
        f"r = {r:.3f}\nN = {len(c_arr)}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax.set_xlabel("BERTScore F1 (rescaled, roberta-large)", fontsize=12)
    ax.set_ylabel(f"Cosine Similarity ({EMBEDDING_MODEL})", fontsize=12)
    ax.set_title("Cross-Validation: BERTScore vs Cosine Similarity", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved cross-validation scatter → {output_path}")

    # Print concordance stats
    total = len(c_arr)
    print(f"\n{'=' * 60}")
    print("Cross-Validation Concordance")
    print(f"{'=' * 60}")
    print(f"  Pearson r          : {r:.4f}")
    print(f"  Both pass          : {both_pass.sum():4d} / {total}  ({both_pass.mean():.1%})")
    print(f"  Only cosine passes : {only_cos.sum():4d} / {total}  ({only_cos.mean():.1%})")
    print(f"  Only BERTScore passes: {only_bs.sum():4d} / {total}  ({only_bs.mean():.1%})")
    print(f"  Both fail          : {both_fail.sum():4d} / {total}  ({both_fail.mean():.1%})")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_results(
    pairs: list[dict],
    scores: list[float],
    threshold: float,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    per_pair = [
        {
            "tcb_instance_id": p["tcb_instance_id"],
            "doc_id": p["doc_id"],
            "tone": p["tone"],
            "cosine_similarity": round(s, 6),
            "pass": s >= threshold,
        }
        for p, s in zip(pairs, scores)
    ]
    scores_path = os.path.join(output_dir, "cosine_per_pair.json")
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(per_pair, fh, ensure_ascii=False, indent=2)
    print(f"[saved] Per-pair cosine scores → {scores_path}")

    arr = np.array(scores)
    summary = {
        "n_pairs": len(scores),
        "n_failed": int((arr < threshold).sum()),
        "pass_rate": float((arr >= threshold).mean()),
        "threshold": threshold,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "embedding_model": EMBEDDING_MODEL,
    }
    summary_path = os.path.join(output_dir, "cosine_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[saved] Summary → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-test TCB rewrite quality with sentence-embedding cosine similarity."
    )
    p.add_argument("--tcb-file", required=True, help="TCB dataset JSON file.")
    p.add_argument("--orig-file", required=True, help="Original source dataset JSON.")
    p.add_argument(
        "--output-dir",
        default="results/quality_analysis",
        help="Directory for plots and score JSONs.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=COSINE_SIM_THRESHOLD,
        help=f"Cosine similarity threshold (default: {COSINE_SIM_THRESHOLD}).",
    )
    p.add_argument(
        "--bertscore-results",
        default=None,
        help=(
            "Path to bertscore_per_pair.json from analyze_tcb_bertscore.py. "
            "When provided, generates a BERTScore vs cosine cross-validation plot."
        ),
    )
    p.add_argument(
        "--bertscore-threshold",
        type=float,
        default=BERTSCORE_THRESHOLD,
        help=f"BERTScore threshold for cross-validation plot (default: {BERTSCORE_THRESHOLD}).",
    )
    p.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Sentence-Transformer model name (default: {EMBEDDING_MODEL}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            'Torch device string. Default: auto-detected (mps > cuda > cpu). '
            'Recommended batch sizes: mps/cuda → 128, cpu → 64.'
        ),
    )
    p.add_argument(
        "--n-low-end",
        type=int,
        default=N_LOW_END,
        help=f"Number of low-end pairs to print (default: {N_LOW_END}).",
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
        print("ERROR: No matching pairs found.")
        sys.exit(1)
    print(f"Built {len(pairs)} document pairs.")

    # ------------------------------------------------------------------
    # 2. Compute cosine similarity
    # ------------------------------------------------------------------
    print(f"\nComputing cosine similarity ({args.embedding_model}) on {len(pairs)} pairs …")
    scores = compute_cosine_similarity(
        pairs,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 3. Distribution summary and low-end inspection
    # ------------------------------------------------------------------
    _describe_distribution(scores, label=f"Cosine Similarity ({args.embedding_model})")
    _print_low_end_pairs(pairs, scores, n=args.n_low_end, label="Cosine Similarity")

    # ------------------------------------------------------------------
    # 4. Report pass rate at the cosine threshold
    # ------------------------------------------------------------------
    threshold = args.threshold
    arr = np.array(scores)
    pass_rate = float((arr >= threshold).mean())
    fail_count = int((arr < threshold).sum())

    print(f"\n{'=' * 60}")
    print(f"THRESHOLD (cosine, {args.embedding_model}): {threshold:.4f}")
    print(f"  Pass rate : {pass_rate:.1%}  ({len(scores) - fail_count}/{len(scores)} pairs)")
    print(f"  Fail count: {fail_count} pairs below threshold")
    print(f"{'=' * 60}")

    # Also report at the BERTScore-calibrated threshold for comparison
    print(f"\nPass rate at BERTScore-calibrated threshold ({args.bertscore_threshold:.4f}): "
          f"{float((arr >= args.bertscore_threshold).mean()):.1%}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    _save_results(pairs, scores, threshold, args.output_dir)

    plot_path = os.path.join(args.output_dir, "cosine_distribution.png")
    try:
        _plot_distribution(scores, threshold, plot_path)
    except Exception as exc:
        logger.warning("Plotting failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # 6. Cross-validation against BERTScore (optional)
    # ------------------------------------------------------------------
    if args.bertscore_results:
        print(f"\nLoading BERTScore results from: {args.bertscore_results}")
        try:
            bs_data = _load_dataset(args.bertscore_results)
            # Align by doc_id
            bs_by_doc_id = {row["doc_id"]: row["bertscore_f1"] for row in bs_data}
            aligned_cos: list[float] = []
            aligned_bs: list[float] = []
            for pair, cos_score in zip(pairs, scores):
                bs_score = bs_by_doc_id.get(pair["doc_id"])
                if bs_score is not None:
                    aligned_cos.append(cos_score)
                    aligned_bs.append(bs_score)

            if aligned_cos:
                print(f"Aligned {len(aligned_cos)} pairs for cross-validation.")
                xval_path = os.path.join(args.output_dir, "crossval_scatter.png")
                try:
                    _plot_crossval(
                        aligned_cos,
                        aligned_bs,
                        cosine_threshold=threshold,
                        bertscore_threshold=args.bertscore_threshold,
                        output_path=xval_path,
                    )
                except Exception as exc:
                    logger.warning("Cross-val plotting failed (non-fatal): %s", exc)
            else:
                print("WARNING: No overlapping doc_ids between cosine and BERTScore results.")
        except Exception as exc:
            logger.warning("Could not load BERTScore results for cross-validation: %s", exc)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset      : {args.tcb_file}")
    print(f"  Total pairs  : {len(scores)}")
    print(f"  Model        : {args.embedding_model}")
    print(f"  Threshold    : {threshold:.4f}")
    print(f"  Pass rate    : {pass_rate:.1%}")
    print(f"  Failed pairs : {fail_count}")
    print(f"  Output dir   : {args.output_dir}")
    if fail_count > 0:
        print(
            f"\n  ACTION NEEDED: {fail_count} pair(s) below cosine threshold. "
            "Cross-check with BERTScore results and re-run "
            "analyze_tcb_bertscore.py --fix-failed if confirmed."
        )
    else:
        print("\n  All pairs pass the cosine threshold.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
