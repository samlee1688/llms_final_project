#!/usr/bin/env python3
"""Build and score a TCB dataset from CONFACT HumC (single pass, no iterative rewrites).

Document selection (CONFACT-specific)
--------------------------------------
  yes-doc: supports_answer == "yes", content_available=True, content non-empty
           → highest credibility, then shortest content as tiebreaker
  no-doc:  supports_answer == "no", content_available=True, content non-empty,
           credibility ≤ yes-doc credibility
           → lowest credibility, then shortest content as tiebreaker
  Instances without a valid (yes-doc, no-doc) pair are skipped.

Rewrite
-------
  yes-doc → neutral   (default: Anthropic claude-haiku-4-5-20251001)
  no-doc  → assertive

Quality checks (single scoring pass)
--------------------------------------
  rescaled BERTScore F1  (roberta-large)    threshold --bs-target     (default 0.30)
  cosine similarity      (all-mpnet-base-v2) threshold --cosine-target (default 0.90)
  Note: CONFACT fact-checking articles are longer and more complex than
  RAMDocs (avg 7K chars vs ~1K), causing the LLM to produce condensed
  rewrites. Rescaled BERTScore for these documents typically ranges
  0.30–0.60; use cosine similarity as the primary semantic quality gate.
  Instances below either threshold are dropped.

Output (all under --dest-dir, default data/full/confact)
---------------------------------------------------------
  tcb_<input-stem>.json          — final TCB dataset
  tcb_<input-stem>.report.json   — quality summary
  quality/meaning_similarity_scores.json
  quality/meaning_similarity_distribution.png

Smoke-test mode (--smoke-test):
  Processes only --smoke-n instances end-to-end, prints the full quality
  report, and exits.

Usage
-----
  # Full run:
  python experiments/build_tcb_confact.py

  # Smoke test (5 instances):
  python experiments/build_tcb_confact.py --smoke-test

  # Score an existing TCB JSON (skip build):
  python experiments/build_tcb_confact.py --skip-initial-build

  # Custom source / output directory:
  python experiments/build_tcb_confact.py \\
      --source data/full/confact/humc_processed_cleaned.json \\
      --dest-dir data/full/confact

  # Custom thresholds after inspecting the distribution:
  python experiments/build_tcb_confact.py --bs-target 0.65 --cosine-target 0.85
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
from typing import Any, Optional

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)

_DEFAULT_REWRITER_PROVIDER = "anthropic"
_DEFAULT_REWRITER_MODEL    = "claude-haiku-4-5-20251001"
_DEFAULT_SOURCE            = "data/full/confact/humc_processed_cleaned.json"
_DEFAULT_DEST_DIR          = "data/full/confact"
_SOURCE_BENCHMARK          = "confact_humc"
_N_LOW_END                 = 15
_PREVIEW                   = 300  # chars shown in low-end printout

_CRED_ORDER: dict[str, int] = {"high": 3, "medium": 2, "low": 1, "unknown": 0}


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
# CONFACT-specific document selection
# ---------------------------------------------------------------------------

def _cred_rank(credibility: str) -> int:
    return _CRED_ORDER.get((credibility or "unknown").lower(), 0)


def _select_confact_docs(instance_dict: dict) -> tuple[Optional[dict], Optional[dict]]:
    """Return (yes_doc_dict, no_doc_dict) for a CONFACT instance dict.

    yes-doc: supports_answer == "yes", content_available, non-empty content
             → highest credibility, then shortest content as tiebreaker
    no-doc:  supports_answer == "no", content_available, non-empty content,
             credibility <= yes-doc credibility
             → lowest credibility, then shortest content as tiebreaker
    Returns (None, None) when no valid pair exists.
    """
    docs = instance_dict.get("documents", [])

    yes_candidates = [
        d for d in docs
        if d.get("supports_answer", "") == "yes"
        and d.get("content", "").strip()
        and d.get("metadata", {}).get("content_available", True)
    ]
    if not yes_candidates:
        return None, None

    yes_doc = sorted(
        yes_candidates,
        key=lambda d: (-_cred_rank(d.get("credibility", "unknown")), len(d.get("content", ""))),
    )[0]
    yes_cred = _cred_rank(yes_doc.get("credibility", "unknown"))

    no_candidates = [
        d for d in docs
        if d.get("supports_answer", "") == "no"
        and d.get("content", "").strip()
        and d.get("metadata", {}).get("content_available", True)
        and _cred_rank(d.get("credibility", "unknown")) <= yes_cred
    ]
    if not no_candidates:
        return None, None

    no_doc = sorted(
        no_candidates,
        key=lambda d: (_cred_rank(d.get("credibility", "unknown")), len(d.get("content", ""))),
    )[0]

    return yes_doc, no_doc


def _make_trimmed_instance(instance_dict: dict, yes_doc: dict, no_doc: dict) -> dict:
    """Build a trimmed instance dict containing only yes_doc and no_doc.

    Sets correct_answer = "yes" so that QueryInstance.correct_documents
    returns [yes_doc] and QueryInstance.incorrect_documents returns [no_doc],
    which is what ToneControlledBenchmarkBuilder expects.
    """
    return {
        "id": instance_dict["id"],
        "question": instance_dict.get("question", ""),
        "documents": [yes_doc, no_doc],
        "correct_answer": "yes",
        "metadata": instance_dict.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# Pair building (same logic as RAMDocs)
# ---------------------------------------------------------------------------

def _build_pairs(tcb_data: list[dict], orig_data: list[dict]) -> list[dict]:
    """Match TCB rewritten docs back to originals for quality scoring.

    Matches via:
      tcb_inst.metadata.constructed_from  →  orig instance id
      tcb_doc.metadata.original_doc_id    →  orig doc id
    Returns a flat list of {tcb_instance_id, src_instance_id, doc_id, tone,
    original, rewritten}.
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
    logger.info(
        "Loading BERTScorer (roberta-large, rescale_with_baseline=True, device=%s) …", device
    )
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
        batch = subset[start : start + batch_size]
        _, _, F1 = scorer.score(
            [p["rewritten"] for p in batch],
            [p["original"]  for p in batch],
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
        [p["original"]  for p in subset],
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
    if len(arr) == 0:
        logger.info("%s: n=0  (no scores)", label)
        return
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
) -> None:
    rows = sorted(
        [
            (
                min(bs_by_id.get(p["doc_id"], 1.0), cos_by_id.get(p["doc_id"], 1.0)),
                bs_by_id.get(p["doc_id"], 1.0),
                cos_by_id.get(p["doc_id"], 1.0),
                p,
            )
            for p in pairs
        ],
        key=lambda t: t[0],
    )
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Bottom {n} pairs  (sort: min(BERTScore, cosine))")
    print(sep)
    for rank, (_, bs, cos, pair) in enumerate(rows[:n], 1):
        orig_p = pair["original"].replace("\n", " ")[:_PREVIEW]
        rew_p  = pair["rewritten"].replace("\n", " ")[:_PREVIEW]
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
    _plot_scores(
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


def _draw_booktabs_table(
    ax: Any, table_data: list, row_ys: list, col_xs: list, col_has: list
) -> None:
    import matplotlib as mpl  # type: ignore
    for i, row in enumerate(table_data):
        for x, ha, txt in zip(col_xs, col_has, row):
            ax.text(
                x, row_ys[i], txt, transform=ax.transAxes,
                ha=ha, va="center", fontsize=9,
                fontfamily=mpl.rcParams.get("font.family", "serif"),
            )
    T = ax.transAxes
    for y, lw in [(1.0, 1.5), (0.0, 1.5)]:
        ax.plot([0, 1], [y, y], transform=T, color="black", lw=lw, clip_on=False)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _plot_scores(
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

        bs_arr  = np.array([s for s in bs_scores  if not np.isnan(s)])
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
                "CONFACT HumC TCB — Meaning Preservation Quality",
                fontsize=10, fontweight="normal", y=0.975,
            )

            table_data = [
                ("Original Dataset Size (Instances)",   str(report.get("original_n_instances", "?")),
                 "BERTScore F1 (rescaled) Min",         f"{report.get('final_min_bs',  float('nan')):.4f}"),
                ("Scored Instances",                     str(report.get("scored_n_instances", "?")),
                 "BERTScore F1 (rescaled) Mean",        f"{report.get('final_mean_bs', float('nan')):.4f}"),
                ("Instances Passing Both Thresholds",   str(report.get("final_n_instances", "?")),
                 "Cosine Similarity Min",               f"{report.get('final_min_cos',  float('nan')):.4f}"),
                ("Doc Pairs Passing Both Thresholds",   str(report.get("final_n_pairs", "?")),
                 "Cosine Similarity Mean",              f"{report.get('final_mean_cos', float('nan')):.4f}"),
            ]
            _draw_booktabs_table(
                ax_tbl, table_data,
                row_ys =[0.875, 0.625, 0.375, 0.125],
                col_xs =[0.01,  0.40,  0.53,  0.99],
                col_has=["left", "right", "left", "right"],
            )

            for ax, arr, label, target, color in [
                (ax_bs,  bs_arr,  "BERTScore F1 (rescaled)", bs_target,  "#4C78A8"),
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
# Direct instance builder (bypasses strict lexical heuristic)
# ---------------------------------------------------------------------------

def _build_confact_instance_raw(
    inst_dict: dict,
    rewriter: Any,
) -> Optional[dict]:
    """Rewrite yes→neutral, no→assertive and return a TCB instance dict.

    Unlike ToneControlledBenchmarkBuilder.build_instance, this skips the
    strict numeric-facts heuristic check.  CONFACT documents are long
    fact-checking articles with many numbers; the heuristic rejects valid
    rewrites that rephrase number-dense passages slightly differently.
    Semantic quality (BERTScore F1 + cosine similarity) is enforced in the
    _validate_loop pass instead, which is the intended quality gate.
    """
    yes_doc, no_doc = _select_confact_docs(inst_dict)
    if yes_doc is None or no_doc is None:
        return None

    neutral   = rewriter.rewrite_with_metadata(yes_doc["content"], "neutral")
    assertive = rewriter.rewrite_with_metadata(no_doc["content"],  "assertive")

    neutral_content   = str(neutral.get("rewritten_content",   "")).strip()
    assertive_content = str(assertive.get("rewritten_content", "")).strip()

    if not neutral_content or not assertive_content:
        logger.debug("Empty rewrite for %s; skipping.", inst_dict["id"])
        return None

    inst_id = inst_dict["id"]
    return {
        "id": f"tcb_{inst_id}",
        "question": inst_dict.get("question", ""),
        "documents": [
            {
                "id": f"{inst_id}_correct_neutral",
                "content": neutral_content,
                "source": yes_doc.get("source", ""),
                "credibility": yes_doc.get("credibility", "high"),
                "supports_answer": yes_doc.get("supports_answer", "yes"),
                "tone": "neutral",
                "metadata": {
                    **yes_doc.get("metadata", {}),
                    "tone_controlled": True,
                    "original_doc_id": yes_doc["id"],
                    "rewrite_meaning_preserved": bool(neutral.get("meaning_preserved", False)),
                    "rewrite_verification_source": str(neutral.get("verification_source", "")),
                    "rewrite_used_fallback": bool(neutral.get("used_fallback", False)),
                },
            },
            {
                "id": f"{inst_id}_incorrect_assertive",
                "content": assertive_content,
                "source": no_doc.get("source", ""),
                "credibility": no_doc.get("credibility", "low"),
                "supports_answer": no_doc.get("supports_answer", "no"),
                "tone": "assertive",
                "metadata": {
                    **no_doc.get("metadata", {}),
                    "tone_controlled": True,
                    "original_doc_id": no_doc["id"],
                    "rewrite_meaning_preserved": bool(assertive.get("meaning_preserved", False)),
                    "rewrite_verification_source": str(assertive.get("verification_source", "")),
                    "rewrite_used_fallback": bool(assertive.get("used_fallback", False)),
                },
            },
        ],
        "correct_answer": inst_dict.get("correct_answer", "yes"),
        "metadata": {
            **inst_dict.get("metadata", {}),
            "constructed_from": inst_id,
            "source_benchmark": _SOURCE_BENCHMARK,
            "tone_controlled": True,
            "correct_doc_id": yes_doc["id"],
            "incorrect_doc_id": no_doc["id"],
            "rewrite_provider": rewriter.provider,
            "rewrite_model": rewriter.model,
            "rewrite_verification": {
                "neutral": {
                    "verification_source": str(neutral.get("verification_source", "")),
                    "used_fallback": bool(neutral.get("used_fallback", False)),
                },
                "assertive": {
                    "verification_source": str(assertive.get("verification_source", "")),
                    "used_fallback": bool(assertive.get("used_fallback", False)),
                },
            },
        },
    }


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
    from src.tcb_builder import ToneRewriter

    raw_instances: list[dict] = _load_json(source_path)
    logger.info("Loaded %d source instances from %s", len(raw_instances), source_path)

    # Pre-screen: only keep instances with a valid yes/no pair
    eligible: list[dict] = []
    skipped_no_pair = 0
    for inst_dict in raw_instances:
        yes_doc, no_doc = _select_confact_docs(inst_dict)
        if yes_doc is None or no_doc is None:
            skipped_no_pair += 1
        else:
            eligible.append(inst_dict)

    logger.info(
        "Eligible instances: %d  |  skipped (no valid yes/no pair): %d",
        len(eligible), skipped_no_pair,
    )
    logger.info(
        "Building initial TCB: %d instances → %s  (provider=%s model=%s workers=%d)",
        len(eligible), tcb_path, provider, model, max_workers,
    )

    def _make_rewriter():
        return ToneRewriter(provider=provider, model=model)

    if max_workers <= 1:
        rewriter = _make_rewriter()
        built: list = []
        for idx, inst_dict in enumerate(eligible, 1):
            try:
                tcb = _build_confact_instance_raw(inst_dict, rewriter)
                if tcb is not None:
                    built.append(tcb)
            except Exception as exc:
                logger.warning("Skipping instance %s: %s", inst_dict.get("id"), exc)
            if built and len(built) % checkpoint_every == 0:
                _save_json(built, tcb_path)
                logger.info("Checkpoint: %d/%d processed  (%d built)", idx, len(eligible), len(built))
        _save_json(built, tcb_path)
        logger.info("Initial TCB done: %d instances → %s", len(built), tcb_path)
        return

    _local = threading.local()

    def _build_one(inst_dict: dict) -> tuple[str, Optional[dict]]:
        if not hasattr(_local, "rewriter"):
            _local.rewriter = _make_rewriter()
        iid = inst_dict.get("id", "?")
        try:
            return iid, _build_confact_instance_raw(inst_dict, _local.rewriter)
        except Exception as exc:
            logger.warning("Skipping instance %s: %s", iid, exc)
            return iid, None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    built_map: dict = {}
    completed = 0
    lock = threading.Lock()
    id_order = [d["id"] for d in eligible]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_one, inst): inst for inst in eligible}
        for fut in as_completed(futures):
            iid, result = fut.result()
            with lock:
                if result is not None:
                    built_map[iid] = result
                completed += 1
                if completed % checkpoint_every == 0:
                    ordered = [built_map[i] for i in id_order if i in built_map]
                    _save_json(ordered, tcb_path)
                    logger.info("Checkpoint: %d/%d submitted  (%d built)", completed, len(eligible), len(built_map))

    built_list = [built_map[i] for i in id_order if i in built_map]
    _save_json(built_list, tcb_path)
    logger.info("Initial TCB done: %d instances → %s", len(built_list), tcb_path)


# ---------------------------------------------------------------------------
# Failed-instance identification
# ---------------------------------------------------------------------------

def _get_failed_ids(
    pairs: list[dict],
    bs_by_id: dict[str, float],
    cos_by_id: dict[str, float],
    bs_target: float,
    cos_target: float,
) -> set[str]:
    failed: set[str] = set()
    for p in pairs:
        if (
            bs_by_id.get(p["doc_id"], 0.0) < bs_target
            or cos_by_id.get(p["doc_id"], 0.0) < cos_target
        ):
            failed.add(p["tcb_instance_id"])
    return failed


# ---------------------------------------------------------------------------
# Validation loop (single scoring pass)
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
    """Score all pairs once, print distribution, drop failures."""
    os.makedirs(quality_dir, exist_ok=True)
    tcb_data = _load_json(tcb_path)
    scored_n_instances = len(tcb_data)
    pairs = _build_pairs(tcb_data, orig_data)
    logger.info("Scoring: %d TCB instances, %d doc pairs", len(tcb_data), len(pairs))

    if not pairs:
        logger.warning("No scoreable pairs found (TCB build may have produced 0 instances).")
        return {
            "benchmark": _SOURCE_BENCHMARK, "tcb_path": tcb_path,
            "original_n_instances": len(orig_data), "scored_n_instances": scored_n_instances,
            "total_scored": 0, "dropped_instances": 0, "final_n_instances": 0,
            "final_n_pairs": 0, "final_pass_rate_bs": 0.0, "final_pass_rate_cos": 0.0,
            "final_mean_bs": 0.0, "final_mean_cos": 0.0, "final_min_bs": 0.0,
            "final_min_cos": 0.0, "bs_target": bs_target, "cos_target": cos_target,
            "drift_bs": 0.0, "drift_cos": 0.0,
        }

    bs_cache: dict[str, float]  = _score_bertscore(pairs, bs_scorer, batch_size)
    cos_cache: dict[str, float] = _score_cosine(pairs, cos_model, batch_size)

    bs_vals  = [bs_cache.get(p["doc_id"],  0.0) for p in pairs]
    cos_vals = [cos_cache.get(p["doc_id"], 0.0) for p in pairs]

    print(f"\n{'─' * 60}")
    print(f"  CONFACT HumC TCB — scoring pass")
    print(f"{'─' * 60}")
    _describe(bs_vals,  "BERTScore F1 (rescaled)")
    _describe(cos_vals, "Cosine similarity")

    bs_drift  = _detect_drift(bs_vals) if bs_vals  else 0.0
    cos_drift = _detect_drift(cos_vals) if cos_vals else 0.0
    print(f"\n  Auto-detected drift  →  BERTScore: {bs_drift:.4f}   Cosine: {cos_drift:.4f}")
    print(f"  Configured threshold →  BERTScore: {bs_target:.4f}   Cosine: {cos_target:.4f}")

    _print_low_end(pairs, bs_cache, cos_cache, n_low_end)

    fail_bs       = sum(1 for v in bs_vals  if v < bs_target)
    fail_cos      = sum(1 for v in cos_vals if v < cos_target)
    failed_ids    = _get_failed_ids(pairs, bs_cache, cos_cache, bs_target, cos_target)
    fail_either   = len(failed_ids)
    pass_rate_bs  = 1.0 - fail_bs  / max(len(bs_vals),  1)
    pass_rate_cos = 1.0 - fail_cos / max(len(cos_vals), 1)

    print(
        f"\n  Pass rate  →  BERTScore: {pass_rate_bs:.1%} ({fail_bs} fail)   "
        f"Cosine: {pass_rate_cos:.1%} ({fail_cos} fail)   "
        f"Instances failing either: {fail_either}"
    )

    if failed_ids:
        print(f"\n  Dropping {fail_either} instance(s) below threshold (no rewrites attempted).")
        kept = [inst for inst in tcb_data if inst["id"] not in failed_ids]
        _save_json(kept, tcb_path)
        logger.info("Dropped %d instances; final: %d", len(failed_ids), len(kept))
        tcb_data = kept

    final_pairs  = _build_pairs(tcb_data, orig_data)
    final_bs  = [bs_cache.get(p["doc_id"],  0.0) for p in final_pairs if p["doc_id"] in bs_cache]
    final_cos = [cos_cache.get(p["doc_id"], 0.0) for p in final_pairs if p["doc_id"] in cos_cache]

    report = {
        "benchmark": _SOURCE_BENCHMARK,
        "tcb_path": tcb_path,
        "original_n_instances": len(orig_data),
        "scored_n_instances": scored_n_instances,
        "total_scored": len(pairs),
        "dropped_instances": fail_either,
        "final_n_instances": len(tcb_data),
        "final_n_pairs": len(final_pairs),
        "final_pass_rate_bs":  round(float(np.mean([v >= bs_target  for v in final_bs])),  4) if final_bs  else 0.0,
        "final_pass_rate_cos": round(float(np.mean([v >= cos_target for v in final_cos])), 4) if final_cos else 0.0,
        "final_mean_bs":  round(float(np.mean(final_bs)),  4) if final_bs  else 0.0,
        "final_mean_cos": round(float(np.mean(final_cos)), 4) if final_cos else 0.0,
        "final_min_bs":   round(float(np.min(final_bs)),   4) if final_bs  else 0.0,
        "final_min_cos":  round(float(np.min(final_cos)),  4) if final_cos else 0.0,
        "bs_target":  bs_target,
        "cos_target": cos_target,
        "drift_bs":   round(bs_drift,  4),
        "drift_cos":  round(cos_drift, 4),
    }
    _save_iter_scores(bs_cache, cos_cache, bs_target, cos_target, report, quality_dir)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build and score a TCB dataset from CONFACT HumC (single pass).\n\n"
            "Selects yes-doc (highest credibility among supports_answer='yes') and "
            "no-doc (lowest credibility ≤ yes-doc among supports_answer='no') per "
            "question, rewrites yes→neutral and no→assertive, then scores with "
            "BERTScore F1 and cosine similarity and drops instances below thresholds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", default=_DEFAULT_SOURCE, metavar="PATH",
        help="Path to cleaned CONFACT HumC JSON in QueryInstance format.",
    )
    p.add_argument(
        "--dest-dir", default=_DEFAULT_DEST_DIR, metavar="DIR",
        help=(
            "Base output directory.  TCB JSON is written to "
            "<dest-dir>/tcb_<input-stem>.json; quality reports go in "
            "<dest-dir>/quality/; JSON report is <dest-dir>/tcb_<input-stem>.report.json."
        ),
    )
    p.add_argument(
        "--rewriter-provider", default=_DEFAULT_REWRITER_PROVIDER,
        choices=["anthropic", "gemini", "local"],
        help="Rewrite provider.",
    )
    p.add_argument(
        "--rewriter-model", default=None, metavar="MODEL",
        help=(
            "Model name for the rewrite provider.  "
            f"Defaults: anthropic→{_DEFAULT_REWRITER_MODEL}, "
            "gemini→gemini-3.1-flash-lite-preview."
        ),
    )
    p.add_argument(
        "--bs-target", type=float, default=0.30,
        help=(
            "Drop threshold for rescaled BERTScore F1 (roberta-large).  "
            "CONFACT fact-checking articles typically score 0.30–0.60 (lower than "
            "RAMDocs Wikipedia-style text) because the LLM condenses long documents.  "
            "Inspect the distribution first; lower than 0.30 risks accepting semantically "
            "drift, higher than 0.50 will drop too many valid rewrites."
        ),
    )
    p.add_argument(
        "--cosine-target", type=float, default=0.90,
        help="Drop threshold for cosine similarity (all-mpnet-base-v2).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Scoring batch size (recommend 128 for mps/cuda, 64 for cpu).",
    )
    p.add_argument(
        "--rewrite-workers", type=int, default=6,
        help="Parallel API workers for the initial TCB build (not used during scoring).",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save TCB checkpoint every N instances during initial build.",
    )
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help='Torch device ("mps", "cuda", "cpu"). Auto-detected when omitted.',
    )
    p.add_argument(
        "--n-low-end", type=int, default=_N_LOW_END,
        help="Number of lowest-scoring pairs printed for inspection.",
    )
    p.add_argument(
        "--skip-initial-build", action="store_true",
        help="Skip initial TCB build (TCB JSON must already exist in --dest-dir).",
    )
    p.add_argument(
        "--force-rebuild", action="store_true",
        help="Delete and re-build the TCB JSON even if it already exists.",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run end-to-end on only --smoke-n instances and exit.",
    )
    p.add_argument(
        "--smoke-n", type=int, default=5, metavar="N",
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

    # ── Derive output filenames from input stem ───────────────────────────────
    in_stem = Path(args.source).stem         # e.g. "humc_processed_cleaned"
    tcb_stem = f"tcb_{in_stem}"              # e.g. "tcb_humc_processed_cleaned"

    dest_dir   = args.dest_dir
    os.makedirs(dest_dir, exist_ok=True)

    tcb_path    = os.path.join(dest_dir, f"{tcb_stem}.json")
    report_path = os.path.join(dest_dir, f"{tcb_stem}.report.json")
    quality_dir = os.path.join(dest_dir, "quality")

    source_path = args.source

    # ── Smoke-test: work in a temp subdir with a small slice ─────────────────
    if args.smoke_test:
        logger.info("=== SMOKE TEST MODE: %d instances ===", args.smoke_n)
        smoke_dir = os.path.join(dest_dir, "smoke")
        os.makedirs(smoke_dir, exist_ok=True)
        raw_all = _load_json(source_path)
        smoke_src = os.path.join(smoke_dir, "smoke_source.json")
        _save_json(raw_all[: args.smoke_n], smoke_src)
        source_path = smoke_src
        tcb_path    = os.path.join(smoke_dir, f"tcb_smoke.json")
        report_path = os.path.join(smoke_dir, f"tcb_smoke.report.json")
        quality_dir = os.path.join(smoke_dir, "quality")
        logger.info("Smoke source: %s  |  TCB: %s", source_path, tcb_path)

    # ── Initial TCB build ─────────────────────────────────────────────────────
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

    # ── Load scoring models ────────────────────────────────────────────────────
    device = args.device or _auto_device()
    logger.info("Using device: %s", device)
    bs_scorer = _load_bertscore(device)
    cos_model = _load_sbert(device)

    # ── Single scoring pass ────────────────────────────────────────────────────
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

    # ── Final report ─────────────────────────────────────────────────────────
    _save_json(summary, report_path)
    print(f"\n{'=' * 60}")
    print("  FINAL QUALITY REPORT")
    print(f"{'=' * 60}")
    print(f"  TCB dataset       : {tcb_path}")
    print(f"  Scored pairs      : {summary['total_scored']}")
    print(f"  Dropped instances : {summary['dropped_instances']}")
    print(f"  Final instances   : {summary['final_n_instances']}")
    print(f"  Final doc pairs   : {summary['final_n_pairs']}")
    print(
        f"  BERTScore F1      : mean={summary['final_mean_bs']:.4f}  "
        f"min={summary['final_min_bs']:.4f}  "
        f"pass@{args.bs_target}={summary['final_pass_rate_bs']:.1%}"
    )
    print(
        f"  Cosine similarity : mean={summary['final_mean_cos']:.4f}  "
        f"min={summary['final_min_cos']:.4f}  "
        f"pass@{args.cosine_target}={summary['final_pass_rate_cos']:.1%}"
    )
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
            f"Inspect quality/meaning_similarity_scores.json to review dropped pairs "
            f"before adjusting --bs-target."
        )

    if args.smoke_test:
        print(f"\nSmoke test complete.  Artifacts in: {os.path.join(dest_dir, 'smoke')}")


if __name__ == "__main__":
    main()
