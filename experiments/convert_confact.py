#!/usr/bin/env python3
"""Download CONFACT HumC and convert to the project's JSON schema.

Source
------
  Repository : https://github.com/zoeyyes/CONFACT
  File       : data/dataset/HumC.pkl.gz  (gzip-compressed pickle, ~26 MB)

Dataset summary
---------------
  * 287 records — each is a yes/no factual claim with multi-source evidence.
  * Labels: "Supported" / "Refuted" (ground truth for the claim).
  * 6–14 evidence pieces per record drawn from web sources.
  * ~25 % of evidence pieces have empty content (paywalled sources such as
    NYT, Reuters, etc.).  These are skipped by default; pass --keep-empty
    to retain them.
  * No per-evidence support/refute or credibility labels in the raw data —
    ``supports_answer`` is left empty and ``credibility`` is "unknown".
    Run an enrichment step (e.g. via media credibility data) to populate
    those fields if needed.

Output schema  (matches project QueryInstance / Document)
---------------------------------------------------------
  QueryInstance
    id              : "humc_<record_id>"
    question        : the yes/no question derived from the claim
    correct_answer  : "yes"  (Supported)  /  "no"  (Refuted)
    documents       : list of Document (one per evidence piece)
    metadata        : claim, raw_label, claim_date, review_date,
                      country, original_claim_url, fact_checking_article,
                      source_benchmark

  Document
    id              : "humc_<record_id>_<evidence_id>"
    content         : scraped web text  (empty string if not available)
    source          : domain extracted from original_link
    credibility     : "unknown"  (populate later via media credibility data)
    supports_answer : ""         (no per-evidence annotation in raw HumC)
    tone            : "original"
    metadata        : evidence_id, original_link, content_available

Usage
-----
  # Download + convert (default output path):
  python experiments/convert_confact.py

  # Custom output path:
  python experiments/convert_confact.py --output data/full/confact/humc.json

  # Skip download if file already exists locally:
  python experiments/convert_confact.py --skip-download \\
      --pkl-path data/raw/confact/HumC.pkl.gz

  # Keep evidence with empty content:
  python experiments/convert_confact.py --keep-empty

  # Limit to N records (for smoke tests):
  python experiments/convert_confact.py --limit 10
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import pickle
import re
import sys
import urllib.parse
import urllib.request
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loader import Document, QueryInstance, save_dataset

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/zoeyyes/CONFACT/main/data/dataset/HumC.pkl.gz"
)
_DEFAULT_PKL  = "data/raw/confact/HumC.pkl.gz"
_DEFAULT_OUT  = "data/full/confact/humc.json"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    def _progress(block_num: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, block_num * block_size / total * 100)
            print(f"\r  {pct:5.1f}%  ({block_num * block_size // 1_048_576} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar
    size_mb = os.path.getsize(dest) / 1_048_576
    logger.info("Downloaded %.1f MB → %s", size_mb, dest)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _domain(url: str) -> str:
    """Extract hostname from a URL (with www. preserved for storage)."""
    try:
        return urllib.parse.urlparse(url).netloc or url
    except Exception:
        return url


def _bare_domain(domain: str) -> str:
    """Strip leading www. for media-data lookup."""
    return domain[4:] if domain.startswith("www.") else domain


# ---------------------------------------------------------------------------
# Media credibility data
# ---------------------------------------------------------------------------

def _load_media(path: str) -> dict[str, dict]:
    """Load all_media_data.pkl.  Keys are bare domains, values have
    'credibility' ('high'/'medium'/'low') and 'details' (str)."""
    logger.info("Loading media credibility data: %s", path)
    with open(path, "rb") as f:
        media: dict[str, dict] = pickle.load(f)
    logger.info("Loaded %d media entries.", len(media))
    return media


_MEDIA_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/zoeyyes/CONFACT/main/data/dataset/all_media_data.pkl"
)
_DEFAULT_MEDIA_PKL = "data/raw/confact/all_media_data.pkl"


def _enrich_with_media(
    instances: list[QueryInstance],
    media: dict[str, dict],
) -> tuple[int, int]:
    """Populate credibility on every Document using the media lookup table.

    Returns (n_enriched, n_unknown) counts.
    """
    enriched = unknown = 0
    for qi in instances:
        for doc in qi.documents:
            entry = media.get(_bare_domain(doc.source))
            if entry:
                doc.credibility = entry["credibility"]
                doc.metadata["media_details"] = entry.get("details", "")
                enriched += 1
            else:
                unknown += 1
    return enriched, unknown


def _correct_answer(label: str) -> str:
    """Map stripped label to yes/no correct answer."""
    return "yes" if label.strip().lower() == "supported" else "no"


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def _convert_record(
    rec: dict[str, Any],
    keep_empty: bool = False,
) -> QueryInstance | None:
    """Convert one HumC record to a QueryInstance.

    Returns None if the record lacks a question or label.
    """
    question = _norm(rec.get("question", ""))
    label    = _norm(rec.get("label", ""))
    if not question or not label:
        return None

    rec_id   = rec["id"]
    c_answer = _correct_answer(label)

    documents: list[Document] = []
    for ev in rec.get("evidence_url", []):
        content = (ev.get("content") or "").strip()
        if not content and not keep_empty:
            continue
        ev_id = ev.get("evidence_id", f"{rec_id}_?")
        link  = ev.get("original_link", "")
        documents.append(
            Document(
                id              = f"humc_{rec_id}_{ev_id}",
                content         = content,
                source          = _domain(link),
                credibility     = "unknown",
                supports_answer = "",
                tone            = "original",
                metadata        = {
                    "evidence_id"      : ev_id,
                    "original_link"    : link,
                    "content_available": bool(content),
                },
            )
        )

    return QueryInstance(
        id            = f"humc_{rec_id}",
        question      = question,
        documents     = documents,
        correct_answer= c_answer,
        metadata      = {
            "source_benchmark"    : "confact_humc",
            "raw_label"           : label,
            "claim"               : _norm(rec.get("claim", "")),
            "claim_date"          : rec.get("claim_date"),
            "review_date"         : rec.get("review_date"),
            "country"             : _norm(rec.get("country", "")).upper(),
            "original_claim_url"  : rec.get("original_claim_url", ""),
            "fact_checking_article": rec.get("fact_checking_article", ""),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CONFACT HumC and convert to the project JSON schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUT,
        help="Output JSON path.",
    )
    p.add_argument(
        "--pkl-path",
        default=_DEFAULT_PKL,
        help="Local path for the downloaded / cached HumC.pkl.gz.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; use --pkl-path as-is (must already exist).",
    )
    p.add_argument(
        "--media-data",
        default=_DEFAULT_MEDIA_PKL,
        metavar="PATH",
        help="Path to all_media_data.pkl for source credibility enrichment.",
    )
    p.add_argument(
        "--skip-media-download",
        action="store_true",
        help="Skip media data download; use --media-data as-is (must already exist).",
    )
    p.add_argument(
        "--no-media",
        action="store_true",
        help="Disable media credibility enrichment entirely.",
    )
    p.add_argument(
        "--keep-empty",
        action="store_true",
        help="Include evidence pieces with empty content (paywalled sources).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Convert only the first N records (useful for smoke tests).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    # ── Download HumC ────────────────────────────────────────────────────────
    if args.skip_download:
        if not os.path.isfile(args.pkl_path):
            logger.error("--skip-download set but file not found: %s", args.pkl_path)
            sys.exit(1)
        logger.info("Using existing file: %s", args.pkl_path)
    else:
        if os.path.isfile(args.pkl_path):
            logger.info("Pickle already cached at %s — skipping download.", args.pkl_path)
        else:
            _download(_DOWNLOAD_URL, args.pkl_path)

    # ── Download media credibility data ─────────────────────────────────────
    media: dict[str, dict] | None = None
    if not args.no_media:
        if args.skip_media_download:
            if not os.path.isfile(args.media_data):
                logger.error("--skip-media-download set but file not found: %s", args.media_data)
                sys.exit(1)
        else:
            if os.path.isfile(args.media_data):
                logger.info("Media data already cached at %s — skipping download.", args.media_data)
            else:
                _download(_MEDIA_DOWNLOAD_URL, args.media_data)
        media = _load_media(args.media_data)

    # ── Load pickle ─────────────────────────────────────────────────────────
    logger.info("Loading pickle: %s", args.pkl_path)
    with gzip.open(args.pkl_path, "rb") as f:
        raw: list[dict] = pickle.load(f)
    logger.info("Loaded %d raw records.", len(raw))

    # ── Convert ─────────────────────────────────────────────────────────────
    instances: list[QueryInstance] = []
    skipped_records = 0
    empty_evidence_skipped = 0

    for i, rec in enumerate(raw):
        if args.limit is not None and i >= args.limit:
            break
        inst = _convert_record(rec, keep_empty=args.keep_empty)
        if inst is None:
            logger.warning("Skipping record id=%s (missing question or label)", rec.get("id"))
            skipped_records += 1
            continue
        if not args.keep_empty:
            total_ev   = len(rec.get("evidence_url", []))
            kept_ev    = len(inst.documents)
            empty_evidence_skipped += total_ev - kept_ev
        instances.append(inst)

    # ── Media credibility enrichment ────────────────────────────────────────
    if media is not None:
        enriched, unknown = _enrich_with_media(instances, media)
        logger.info(
            "Credibility : %d enriched  /  %d unknown  (%.1f%% coverage)",
            enriched, unknown, 100 * enriched / max(enriched + unknown, 1),
        )
        # Per-tier breakdown
        import collections
        tier_counts: dict[str, int] = collections.Counter(
            doc.credibility
            for qi in instances
            for doc in qi.documents
        )
        logger.info(
            "Tiers       : high=%d  medium=%d  low=%d  unknown=%d",
            tier_counts.get("high", 0),
            tier_counts.get("medium", 0),
            tier_counts.get("low", 0),
            tier_counts.get("unknown", 0),
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    n_yes = sum(1 for qi in instances if qi.correct_answer == "yes")
    n_no  = sum(1 for qi in instances if qi.correct_answer == "no")
    n_ev  = sum(len(qi.documents) for qi in instances)

    logger.info("Converted  : %d instances  (%d Supported / %d Refuted)", len(instances), n_yes, n_no)
    logger.info("Documents  : %d total evidence pieces kept", n_ev)
    if not args.keep_empty:
        logger.info("Skipped    : %d empty-content evidence pieces (paywalled)", empty_evidence_skipped)
    if skipped_records:
        logger.warning("Records skipped (bad data): %d", skipped_records)

    # ── Save ────────────────────────────────────────────────────────────────
    save_dataset(instances, args.output)
    logger.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
