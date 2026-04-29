#!/usr/bin/env python3
"""Readability and assertiveness audit for TCB rewrite pairs.

Addresses reviewer Critique 3 (Confound A): "The paper does not verify that
fluency, reading level, or argument structure are matched between neutral and
assertive rewrites."

Computes per-pair:
  - Flesch-Kincaid Grade Level (FKGL): delta must be small
  - Gunning Fog Index (GFI): delta must be small
  - Flesch Reading Ease: informational
  - Average sentence length: informational
  - Hedge density (neutral vs assertive): assertive should have FEWER hedges
  - ROUGE-L F1: content overlap sanity check

Outputs:
  - JSON with per-pair scores
  - Console summary: mean ± std for each metric, t-test (paired) on hedge density

Usage
-----
python experiments/analyze_tcb_readability.py \\
    --tcb data/full/confact/tcb_humc_processed_cleaned.json \\
    --out results/quality_extended/confact_readability.json

python experiments/analyze_tcb_readability.py \\
    --tcb data/full/ramdocs/tcb_ramdocs_full.json \\
    --out results/quality_extended/ramdocs_readability.json \\
    --plot results/quality_extended/ramdocs_readability.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Hedging word pattern (imported from quality_checker_extended)
_HEDGE_PATTERN = re.compile(
    r"\b(?:allegedly|reportedly|apparently|seemingly|purportedly|"
    r"might|could|may|would|should(?= have)|"
    r"possibly|probably|perhaps|maybe|likely|"
    r"suggest(?:s|ed|ing)?|indicate(?:s|d|ing)?|appear(?:s|ed|ing)?|"
    r"seem(?:s|ed|ing)?|claim(?:s|ed|ing)?|assert(?:s|ed|ing)?(?! ively)|"
    r"according to|it is (?:believed|thought|suggested|claimed)|"
    r"some (?:suggest|argue|believe|claim)|"
    r"evidence (?:suggests|indicates)|research (?:suggests|shows))\b",
    re.IGNORECASE,
)


def _hedge_density(text: str) -> float:
    words = len(text.split())
    return len(_HEDGE_PATTERN.findall(text)) / words * 100 if words > 0 else 0.0


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for x in a:
        curr = [0] * (n + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r or not h:
        return 0.0
    lcs = _lcs_length(r, h)
    p = lcs / len(h)
    rec = lcs / len(r)
    return 2 * p * rec / (p + rec) if p + rec > 0 else 0.0


def _readability_scores(text: str) -> dict:
    try:
        import textstat  # type: ignore
        return {
            "fkgl": textstat.flesch_kincaid_grade(text),
            "gfi": textstat.gunning_fog(text),
            "flesch_ease": textstat.flesch_reading_ease(text),
            "avg_sent_len": textstat.avg_sentence_length(text),
            "word_count": len(text.split()),
        }
    except ImportError:
        # Approximate FKGL without textstat
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        n_words = len(words)
        n_sents = max(1, len(sentences))
        # Count syllables approximately: each vowel group = 1 syllable, min 1
        def syllables(word: str) -> int:
            vowels = re.findall(r"[aeiouy]+", word.lower())
            return max(1, len(vowels))
        n_syll = sum(syllables(w) for w in words)
        asl = n_words / n_sents  # avg sentence length
        asw = n_syll / max(1, n_words)  # avg syllables per word
        fkgl = 0.39 * asl + 11.8 * asw - 15.59
        return {
            "fkgl": round(fkgl, 2),
            "gfi": None,  # requires complex word count
            "flesch_ease": None,
            "avg_sent_len": round(asl, 2),
            "word_count": n_words,
        }


def _score_pair(instance_id: str, neutral: str, assertive: str) -> dict:
    n_scores = _readability_scores(neutral)
    a_scores = _readability_scores(assertive)
    n_hedge = _hedge_density(neutral)
    a_hedge = _hedge_density(assertive)
    rl = _rouge_l(neutral, assertive)

    row: dict = {
        "instance_id": instance_id,
        "neutral_word_count": n_scores.get("word_count", 0),
        "assertive_word_count": a_scores.get("word_count", 0),
        "neutral_fkgl": n_scores.get("fkgl"),
        "assertive_fkgl": a_scores.get("fkgl"),
        "delta_fkgl": None,
        "neutral_gfi": n_scores.get("gfi"),
        "assertive_gfi": a_scores.get("gfi"),
        "delta_gfi": None,
        "neutral_flesch_ease": n_scores.get("flesch_ease"),
        "assertive_flesch_ease": a_scores.get("flesch_ease"),
        "neutral_avg_sent_len": n_scores.get("avg_sent_len"),
        "assertive_avg_sent_len": a_scores.get("avg_sent_len"),
        "neutral_hedge_density": round(n_hedge, 3),
        "assertive_hedge_density": round(a_hedge, 3),
        "assertiveness_advantage": round(n_hedge - a_hedge, 3),
        "rouge_l_f1": round(rl, 4),
    }

    if n_scores.get("fkgl") is not None and a_scores.get("fkgl") is not None:
        row["delta_fkgl"] = round(a_scores["fkgl"] - n_scores["fkgl"], 2)
    if n_scores.get("gfi") is not None and a_scores.get("gfi") is not None:
        row["delta_gfi"] = round(a_scores["gfi"] - n_scores["gfi"], 2)

    return row


def _extract_pairs(tcb: list[dict]) -> list[tuple[str, str, str]]:
    """Extract (instance_id, neutral_text, assertive_text) from TCB instances."""
    pairs = []
    for inst in tcb:
        inst_id = inst.get("id", f"inst_{len(pairs)}")
        docs = inst.get("documents", [])
        neutral = next(
            (d.get("content", "") for d in docs if str(d.get("tone", "")).lower() == "neutral"), ""
        )
        assertive = next(
            (d.get("content", "") for d in docs if str(d.get("tone", "")).lower() == "assertive"), ""
        )
        if neutral and assertive:
            pairs.append((inst_id, neutral, assertive))
    return pairs


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not (v != v)]
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return round(mean, 3), round(var ** 0.5, 3)


def _paired_t_test(a: list[float], b: list[float]) -> dict:
    """Simple paired t-test (H0: mean(a) == mean(b))."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    n = len(diffs)
    if n < 2:
        return {"t": float("nan"), "p_approx": float("nan"), "n": n}
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = (var_d / n) ** 0.5
    if se == 0:
        return {"t": float("nan"), "p_approx": 1.0, "n": n}
    t = mean_d / se
    # Approximate p-value using normal distribution for large n
    abs_t = abs(t)
    # Two-sided p-value approximation (accurate for df > 30)
    p_approx = 2 * (1 - _normal_cdf(abs_t))
    return {"t": round(t, 4), "p_approx": round(p_approx, 4), "n": n}


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if x < 0:
        return 1 - _normal_cdf(-x)
    t_abram = 1 / (1 + 0.2316419 * x)
    poly = t_abram * (0.319381530
          + t_abram * (-0.356563782
          + t_abram * (1.781477937
          + t_abram * (-1.821255978
          + t_abram * 1.330274429))))
    return 1 - (1 / (2 * math.pi) ** 0.5) * math.exp(-x * x / 2) * poly


def _plot_distributions(rows: list[dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("TCB Rewrite Quality: Readability & Assertiveness Audit", fontsize=14)

    metrics = [
        ("delta_fkgl", "ΔFKGL (assertive − neutral)", "grade levels"),
        ("delta_gfi", "ΔGunning Fog (assertive − neutral)", "grade levels"),
        ("assertiveness_advantage", "Hedge Density Advantage (neutral − assertive)", "hedges/100 words"),
        ("rouge_l_f1", "ROUGE-L F1 (content overlap)", "F1"),
        ("neutral_fkgl", "Neutral FKGL distribution", "grade level"),
        ("assertive_fkgl", "Assertive FKGL distribution", "grade level"),
    ]

    for ax, (key, title, xlabel) in zip(axes.flat, metrics):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            ax.hist(vals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            m, s = _mean_std(vals)
            ax.axvline(m, color="red", linestyle="--", label=f"mean={m:.2f}")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved → %s", out_path)
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Readability & assertiveness audit for TCB pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tcb", required=True, metavar="PATH")
    p.add_argument("--out", required=True, metavar="PATH")
    p.add_argument("--plot", default=None, metavar="PATH")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    tcb = json.loads(Path(args.tcb).read_text())
    pairs = _extract_pairs(tcb)
    logger.info("Scoring %d neutral/assertive pairs from %s", len(pairs), args.tcb)

    rows = [_score_pair(iid, n, a) for iid, n, a in pairs]

    # Summary statistics
    def _vals(key: str) -> list:
        return [r[key] for r in rows if r.get(key) is not None]

    print("=" * 70)
    print("READABILITY AUDIT SUMMARY")
    print("=" * 70)
    for key, label in [
        ("delta_fkgl", "ΔFKGL (assertive − neutral)"),
        ("delta_gfi", "ΔGunning Fog (assertive − neutral)"),
        ("assertiveness_advantage", "Hedge advantage (neutral − assertive)"),
        ("rouge_l_f1", "ROUGE-L F1"),
    ]:
        v = _vals(key)
        if v:
            m, s = _mean_std(v)
            print(f"  {label:<45} mean={m:+.3f}  std={s:.3f}  n={len(v)}")

    print()
    neutral_hedges = _vals("neutral_hedge_density")
    assertive_hedges = _vals("assertive_hedge_density")
    t_result = _paired_t_test(neutral_hedges, assertive_hedges)
    print(f"  Paired t-test (hedge density neutral vs assertive):")
    print(f"    t={t_result['t']:.3f}  p≈{t_result['p_approx']:.4f}  n={t_result['n']}")
    advantage_pct = sum(1 for r in rows if r["assertiveness_advantage"] > 0) / len(rows) * 100
    print(f"    Assertive has FEWER hedges in {advantage_pct:.1f}% of pairs (manipulation success rate)")
    print("=" * 70)

    out = {
        "tcb_path": args.tcb,
        "n_pairs": len(rows),
        "summary": {
            k: {"mean": _mean_std(_vals(k))[0], "std": _mean_std(_vals(k))[1], "n": len(_vals(k))}
            for k in ["delta_fkgl", "delta_gfi", "assertiveness_advantage", "rouge_l_f1",
                      "neutral_hedge_density", "assertive_hedge_density"]
        },
        "hedge_t_test": t_result,
        "manipulation_success_rate": round(advantage_pct / 100, 4),
        "pairs": rows,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Written → %s", args.out)

    if args.plot:
        _plot_distributions(rows, args.plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
