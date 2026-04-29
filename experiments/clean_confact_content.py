#!/usr/bin/env python3
"""Clean scraped web content in humc.json.

Pipeline (in order)
-------------------
1. Known error-page detection  – Facebook walls, Quora JS errors, YouTube
   footer-only pages, login walls, JS-required pages → content set to "".
2. Navigation-separator splitting  – content is split on block-level nav
   characters (✕ × ☰ | · •) and newlines to isolate nav fragments from
   article paragraphs.
3. Boilerplate segment removal  – each segment is checked against a list of
   known boilerplate phrases (skip-to-content links, edition selectors,
   cookie notices, copyright lines, social-share CTAs, etc.).
4. Empty segment removal  – blank segments are discarded.
5. Deduplication  – segments that are exact duplicates (after whitespace
   normalisation) of an earlier segment are discarded (header/footer nav
   items commonly appear twice).
6. Rejoin  – surviving segments are joined with a single space.

Outputs
-------
- ``<input_stem>_cleaned.json``  – cleaned dataset (same schema as input,
  with ``metadata.original_content_length`` and
  ``metadata.cleaned_content_length`` added to every document).
- ``<input_stem>_cleaning_stats.csv``  – per-document stats in CSV format.
- ``<input_stem>_content_length_distribution.png``  – histogram of cleaned
  content lengths.

Usage
-----
  python experiments/clean_confact_content.py
  python experiments/clean_confact_content.py --input data/full/confact/humc.json
  python experiments/clean_confact_content.py --dry-run
  python experiments/clean_confact_content.py --sample 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Error-page fingerprints  (whole-content → zero out document)
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in [
    # Facebook
    r"facebook is not available on this browser",
    r"you must log in first",
    r"इस ब्राउज़र पर उपलब्ध नहीं है",
    # Quora JS wall
    r"something went wrong\.\s*wait a moment and try again",
    r"please enable javascript.*quora",
    # YouTube footer-only (no article)
    r"about press copyright contact us.*how youtube works",
    # Generic JS-required
    r"javascript is required",
    r"please enable javascript.*to (continue|use|run)",
    r"you need to enable javascript to run this app",
    r"client doesn't work properly without javascript",
    r"doesn't work properly without javascript enabled",
    # Login / paywall walls
    r"log in to (facebook|twitter|instagram)",
    # Technical error pages
    r"we'?re sorry,? this site is currently experiencing technical difficulties",
    r"exception: request blocked",
    # IMF DataMapper (pure JS app shell)
    r"about research countries capacity development.*imf",
    # Europol JS app
    r"loading application\. please wait",
    # ── Browser incompatibility / JS-wall variants ────────────────────────────
    # NBCNews (IE 11 compat banner replaces entire page content)
    r"ie\s*1[01]\s+is not supported",
    r"for an optimal experience visit our site on another browser",
    # The Health Foundation — old-browser banner
    r"unfortunately,?\s+your browser is too old to work on this website",
    # Sites requiring JS (nestle.com, elcot.in, morth.nic.in, etc.)
    r"sorry,?\s+you need to enable javascript to visit this website",
    # Office for Students, similar "Warning" banners
    r"warning[:\s]+javascript is turned off",
    # Repository / academic JS wall
    r"javascript is disabled\.",
    # Creators Syndicate JS requirement
    r"requires javascript for full functionality\.\s*please enable javascript",
    # MEA India — JS not supported
    r"this page uses javascript[.;]?\s+your browser either doesn't support javascript",
    # ── Paywall pages ────────────────────────────────────────────────────────
    r"join your (group )?subscription",           # FT.com
    r"subscribe to (read|access|continue)",
    # ── Captcha / bot-blocked pages ───────────────────────────────────────────
    r"request unsuccessful.*incapsula incident id",   # Imperva/Incapsula block
]]


def _is_error_page(text: str) -> bool:
    for pat in _ERROR_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# 2. Navigation-separator splitter
# ---------------------------------------------------------------------------

# Split on:
#   - Newlines (scrapers preserve block structure as \n / \r\n)
#   - ✕ × ☰  — common nav toggle / close characters
#   - |  ·  •  — list-separator characters used in nav breadcrumbs / menus
#     (require surrounding whitespace to avoid splitting on legitimate uses
#      of | in prose, e.g. tables)
_NAV_SPLIT = re.compile(
    r"[\n\r]+"
    r"|(?<=[^\s])\s*[✕×☰]\s*"
    r"|(?<=\s)[|·•]\s+"
)


# ---------------------------------------------------------------------------
# 3. Boilerplate segment patterns
# ---------------------------------------------------------------------------

_BOILERPLATE_SEGMENT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Skip / navigation links  (anchored — only drop segments that START here)
        r"^skip to (main )?content\b",
        r"^skip to navigation\b",
        r"^back to (home|top)",
        # Edition selectors
        r"^(international|uk|us|australia|europe) edition\b",
        # Image / slideshow controls
        r"^(next|previous) image\b",
        r"^toggle caption\b",
        r"^close dialogue\b",
        r"^\d+ / \d+\b.*\b(image|photo|slide)\b",
        # View all / Show more
        r"^view all\b",
        r"^(show more|hide expanded menu)$",
        # Subscribe / sign-in CTAs (only when the segment is clearly just a CTA)
        r"^(sign in|log in|subscribe|register)\s*$",
        r"^print subscriptions\s*$",
        # Social share
        r"^(share (this|on)|tweet|kindly share this story)",
        # Cookie / privacy notices
        r"^(we use cookies|view our privacy policy)",
        r"^(accept( all)?|decline|manage) cookies\s*$",
        # Copyright / footer (entire segment)
        r"^©\s*\d{4}",
        r"^all rights reserved\s*$",
        # Newspaper section-header nav items (entire segment is a single nav word)
        r"^(news|sport|culture|lifestyle|opinion|business|health|politics|entertainment"
        r"|education|videos|home|international|search)\s*$",
        # Facebook / Meta footer links
        r"^\+\s*meta\s*©",
        # YouTube about-page boilerplate
        r"^about press copyright contact us",
        # Guardian section navigation
        r"^the guardian - back to home",
        r"^the guardian\s*$",
        # Hamburger / close icon residue
        r"^☰\s",
        r"^×\s",
    ]
]


def _is_boilerplate_segment(seg: str) -> bool:
    s = seg.strip()
    if not s:
        return True
    for pat in _BOILERPLATE_SEGMENT_PATTERNS:
        if pat.search(s):
            return True
    return False


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_content(text: str) -> tuple[str, dict[str, bool]]:
    """Return (cleaned_text, filter_flags).

    filter_flags keys
    -----------------
    error_detection   – entire content was an error / login / JS-required page
    nav_separator     – at least one nav-separator char split the content
    boilerplate       – at least one segment matched a boilerplate pattern
    empty_segment     – at least one segment was empty after stripping
    dedup             – at least one segment was an exact duplicate of an earlier one
    """
    flags: dict[str, bool] = {
        "error_detection": False,
        "nav_separator":   False,
        "boilerplate":     False,
        "empty_segment":   False,
        "dedup":           False,
    }

    if not text or not text.strip():
        return "", flags

    if _is_error_page(text):
        flags["error_detection"] = True
        return "", flags

    segments = _NAV_SPLIT.split(text)
    if len(segments) > 1:
        flags["nav_separator"] = True

    seen: set[str] = set()
    kept: list[str] = []
    for raw_seg in segments:
        seg = raw_seg.strip()
        if not seg:
            flags["empty_segment"] = True
            continue
        if _is_boilerplate_segment(seg):
            flags["boilerplate"] = True
            continue
        key = re.sub(r"\s+", " ", seg.lower())
        if key in seen:
            flags["dedup"] = True
            continue
        seen.add(key)
        kept.append(seg)

    return " ".join(kept), flags


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _derive_paths(input_path: str, output_path: str | None) -> tuple[str, str, str, str]:
    """Return (output_json, stats_csv, plot_png, out_dir)."""
    base, ext = os.path.splitext(input_path)
    out_json  = output_path or (base + "_cleaned" + ext)
    out_dir   = os.path.dirname(os.path.abspath(out_json))
    stem      = os.path.splitext(os.path.basename(out_json))[0]
    stats_csv = os.path.join(out_dir, stem + "_stats.csv")
    plot_png  = os.path.join(out_dir, stem + "_content_length_distribution.png")
    return out_json, stats_csv, plot_png, out_dir


def _percentile_rank(sorted_vals: list[int], v: int) -> float:
    """Percentile rank of *v* in a pre-sorted list (0–100, higher = longer)."""
    import bisect
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    # fraction of values strictly below v, mid-point for ties
    below = bisect.bisect_left(sorted_vals, v)
    equal = bisect.bisect_right(sorted_vals, v) - below
    return round(100.0 * (below + 0.5 * equal) / n, 1)


def _annotate_percentiles(rows: list[dict]) -> None:
    """Add percentile and decile-band columns to each row in-place."""
    import bisect

    orig_sorted    = sorted(r["original_length_chars"] for r in rows)
    cleaned_sorted = sorted(r["cleaned_length_chars"]  for r in rows)

    for r in rows:
        p_orig    = _percentile_rank(orig_sorted,    r["original_length_chars"])
        p_cleaned = _percentile_rank(cleaned_sorted, r["cleaned_length_chars"])

        r["percentile_original"] = p_orig
        r["percentile_cleaned"]  = p_cleaned
        r["top_10pct_original"]  = "Y" if p_orig    >= 90 else "N"
        r["top_10pct_cleaned"]   = "Y" if p_cleaned >= 90 else "N"
        r["btw_10_20pct_original"] = "Y" if 80 <= p_orig    < 90 else "N"
        r["btw_10_20pct_cleaned"]  = "Y" if 80 <= p_cleaned < 90 else "N"


def _write_stats(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Stats CSV → %s  (%d rows)", path, len(rows))


def _plot_distribution(rows: list[dict], path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cleaned_lens = [r["cleaned_length_chars"] for r in rows]
    orig_lens    = [r["original_length_chars"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("CONFACT HumC – Document Content Length Distribution After Cleanup",
                 fontsize=12, y=1.01)

    bins = 50

    ax0 = axes[0]
    ax0.hist(orig_lens, bins=bins, color="#4878CF", edgecolor="white", linewidth=0.4)
    ax0.set_title("Original content length")
    ax0.set_xlabel("Characters")
    ax0.set_ylabel("Documents")
    ax0.yaxis.grid(True, linewidth=0.4, color="0.85")
    ax0.set_axisbelow(True)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = axes[1]
    ax1.hist(cleaned_lens, bins=bins, color="#6ACC65", edgecolor="white", linewidth=0.4)
    ax1.set_title("Cleaned content length")
    ax1.set_xlabel("Characters")
    ax1.set_ylabel("Documents")
    ax1.yaxis.grid(True, linewidth=0.4, color="0.85")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    n_total    = len(rows)
    n_zeroed   = sum(1 for r in rows if r["filtered_error_detection"] == "Y")
    n_nonempty = sum(1 for r in rows if r["cleaned_length_chars"] > 0)
    med_clean  = int(np.median([c for c in cleaned_lens if c > 0])) if n_nonempty else 0
    mean_clean = int(np.mean([c for c in cleaned_lens if c > 0]))   if n_nonempty else 0

    summary = (
        f"N docs = {n_total}\n"
        f"Error pages zeroed = {n_zeroed}\n"
        f"Non-empty after clean = {n_nonempty}\n"
        f"Median cleaned length = {med_clean:,} chars\n"
        f"Mean cleaned length = {mean_clean:,} chars"
    )
    fig.text(0.5, -0.04, summary, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="0.7"))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot → %s", path)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _process_file(
    input_path: str,
    output_path: str | None,
    dry_run: bool,
    sample_n: int,
) -> None:
    out_json, stats_csv, plot_png, out_dir = _derive_paths(input_path, output_path)

    logger.info("Loading %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    stat_rows: list[dict] = []
    sampled:   list[tuple[str, str, str]] = []

    total_before = total_after = 0
    zeroed = 0

    for qi in data:
        for doc in qi["documents"]:
            raw: str = doc.get("content") or ""
            cleaned, flags = clean_content(raw)

            before_len = len(raw)
            after_len  = len(cleaned)
            total_before += before_len
            total_after  += after_len
            if flags["error_detection"]:
                zeroed += 1

            doc["metadata"]["original_content_length"] = before_len
            doc["metadata"]["cleaned_content_length"]  = after_len
            doc["metadata"]["content_available"]       = bool(cleaned)
            doc["content"] = cleaned

            reduction_chars = before_len - after_len
            reduction_pct   = round(100.0 * reduction_chars / before_len, 1) if before_len > 0 else 0.0

            stat_rows.append({
                "doc_id":                    doc["id"],
                "instance_id":               qi["id"],
                "source":                    doc.get("source", ""),
                "credibility":               doc.get("credibility", ""),
                "original_length_chars":     before_len,
                "cleaned_length_chars":      after_len,
                "reduction_chars":           reduction_chars,
                "reduction_pct":             reduction_pct,
                "filtered_error_detection":  "Y" if flags["error_detection"] else "N",
                "filtered_nav_separator":    "Y" if flags["nav_separator"]   else "N",
                "filtered_boilerplate":      "Y" if flags["boilerplate"]     else "N",
                "filtered_empty_segment":    "Y" if flags["empty_segment"]   else "N",
                "filtered_dedup":            "Y" if flags["dedup"]           else "N",
                "content_available":         "Y" if cleaned else "N",
                # percentile columns populated in second pass below
            })

            if sample_n > 0 and len(sampled) < sample_n and before_len > 300:
                sampled.append((doc["id"], raw[:600], cleaned[:600]))

    _annotate_percentiles(stat_rows)

    n_total   = len(stat_rows)
    reduction = 100.0 * (1 - total_after / max(total_before, 1))
    logger.info("Documents processed : %d", n_total)
    logger.info("Error pages zeroed  : %d  (%.1f%%)", zeroed,
                100 * zeroed / max(n_total, 1))
    logger.info("Chars before        : %d", total_before)
    logger.info("Chars after         : %d", total_after)
    logger.info("Reduction           : %.1f%%", reduction)
    logger.info("Avg chars before    : %.0f", total_before / max(n_total, 1))
    logger.info("Avg chars after     : %.0f", total_after / max(n_total, 1))

    if sample_n > 0:
        random.shuffle(sampled)
        for doc_id, before, after in sampled[:sample_n]:
            print(f"\n{'='*60}\nDOC: {doc_id}")
            print(f"--- BEFORE ({len(before)} shown) ---\n{before}")
            print(f"--- AFTER ({len(after)} shown) ---\n{after}")

    if dry_run:
        logger.info("Dry-run — no files written.")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Cleaned JSON → %s", out_json)

    _write_stats(stat_rows, stats_csv)
    _plot_distribution(stat_rows, plot_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean scraped web content in humc.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default="data/full/confact/humc.json")
    p.add_argument("--output", "-o", default=None,
                   help="Output JSON (default: <input>_cleaned.json).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats; do not write files.")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="Print before/after for N random docs.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    _process_file(args.input, args.output, dry_run=args.dry_run, sample_n=args.sample)


if __name__ == "__main__":
    main()
