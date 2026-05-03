#!/usr/bin/env python3
"""Re-scrape web pages and extract clean article text for CONFACT HumC.

For each document in humc.json where metadata["content_available"] is True,
fetches the original URL and extracts clean article text using trafilatura
(removes navigation, ads, headers, footers, images, links, HTML markup).
The extracted text replaces the document's "content" field.

Parallelism model
-----------------
  --workers N   spawns N OS processes, each running its own asyncio event loop
                and aiohttp session over its assigned slice of URLs.
                Total simultaneous connections = workers × concurrency.

  --concurrency C  max simultaneous HTTP connections per worker (default 10).

  For 2592 URLs a practical starting point is:
      --workers 4 --concurrency 10   (40 total connections, ~4 CPU cores)
  or simply increase concurrency in a single worker:
      --workers 1 --concurrency 40   (equivalent I/O throughput, 1 CPU core)

  Multiple workers add value when trafilatura CPU parsing is a bottleneck,
  or when you want fault isolation between slices.

Worker data flow
----------------
  Each worker receives a list of (doc_id, url, original_content) plain tuples
  (picklable). It returns a list of result dicts containing the retrieved text.
  The main process applies the content updates to the loaded data and writes
  the output files — no shared state between processes.

Dependencies
------------
  pip install trafilatura aiohttp

Outputs
-------
  <stem>_retrieved.json      – dataset with updated content fields
  <stem>_retrieved_log.csv   – per-doc: id, original_link, retrieval_status,
                                original_content, retrieved_content

Usage
-----
  # Full parallel run (4 workers × 10 connections each):
  python experiments/retrieve_confact_content.py --workers 4

  # Single-process, high-concurrency (equivalent for I/O-bound work):
  python experiments/retrieve_confact_content.py --concurrency 40

  # Smoke test — first 20 docs, 2 workers:
  python experiments/retrieve_confact_content.py --limit 20 --workers 2

  # Dry-run (no HTTP, no writes):
  python experiments/retrieve_confact_content.py --dry-run --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import json
import logging
import math
import os
import random
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB per page cap


# ---------------------------------------------------------------------------
# Fetch (async, runs inside each worker's event loop)
# ---------------------------------------------------------------------------

async def _fetch_html(
    session:   Any,
    url:       str,
    timeout_s: int,
    retries:   int = 2,
) -> tuple[str | None, str]:
    """Return (html_text, status).  status is 'OK' or a short error code."""
    import aiohttp

    last_err = "UNKNOWN"
    for attempt in range(retries + 1):
        if attempt > 0:
            await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
        try:
            async with session.get(
                url,
                headers=_BROWSER_HEADERS,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
                allow_redirects=True,
                max_redirects=5,
            ) as resp:
                if resp.status == 200:
                    raw  = await resp.read()
                    html = raw[:_MAX_RESPONSE_BYTES].decode(
                        resp.charset or "utf-8", errors="replace"
                    )
                    return html, "OK"
                elif resp.status in (429, 503) and attempt < retries:
                    last_err = f"HTTP_{resp.status}_RETRY"
                    continue
                else:
                    return None, f"HTTP_{resp.status}"
        except asyncio.TimeoutError:
            last_err = "TIMEOUT"
        except Exception as exc:
            last_err = type(exc).__name__

    return None, last_err


# ---------------------------------------------------------------------------
# Extract (CPU — trafilatura; called synchronously inside the async loop)
# ---------------------------------------------------------------------------

def _extract_text(html: str, url: str) -> str | None:
    import trafilatura
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )
    return text.strip() if text else None


# ---------------------------------------------------------------------------
# Per-URL coroutine
# ---------------------------------------------------------------------------

# A target is a plain 3-tuple so it is picklable across process boundaries.
Target = tuple[str, str, str]  # (doc_id, url, original_content)


async def _process_target(
    session:   Any,
    target:    Target,
    sem:       asyncio.Semaphore,
    timeout_s: int,
    delay_s:   float,
) -> dict[str, Any]:
    """Fetch + extract one URL.  Returns a result dict."""
    doc_id, url, original_content = target

    if not url:
        return dict(doc_id=doc_id, url=url, status="NO_URL",
                    original_content=original_content, retrieved_content="")

    async with sem:
        if delay_s > 0:
            await asyncio.sleep(delay_s + random.uniform(0, delay_s * 0.5))
        html, fetch_status = await _fetch_html(session, url, timeout_s)

    if html is None:
        return dict(doc_id=doc_id, url=url, status=fetch_status,
                    original_content=original_content, retrieved_content="")

    retrieved = _extract_text(html, url)
    if not retrieved:
        return dict(doc_id=doc_id, url=url, status="EXTRACTION_FAILED",
                    original_content=original_content, retrieved_content="")

    return dict(doc_id=doc_id, url=url, status="OK",
                original_content=original_content, retrieved_content=retrieved)


# ---------------------------------------------------------------------------
# Async batch runner (one event loop per worker process)
# ---------------------------------------------------------------------------

async def _run_batch_async(
    targets:     list[Target],
    concurrency: int,
    timeout_s:   int,
    delay_s:     float,
    worker_id:   int,
) -> list[dict[str, Any]]:
    import aiohttp

    sem       = asyncio.Semaphore(concurrency)
    results:  list[dict[str, Any]] = []
    n         = len(targets)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _process_target(session, t, sem, timeout_s, delay_s),
                name=t[0],
            )
            for t in targets
        ]
        done = 0
        t0   = time.monotonic()
        for coro in asyncio.as_completed(tasks):
            row = await coro
            results.append(row)
            done += 1
            if done % 50 == 0 or done == n:
                elapsed = time.monotonic() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                ok      = sum(1 for r in results if r["status"] == "OK")
                # Each worker logs independently; the prefix makes it easy to
                # identify which worker the line came from.
                logging.getLogger(__name__).info(
                    "[worker %d | %d/%d]  %.1f docs/s  OK=%d  fail=%d",
                    worker_id, done, n, rate, ok, done - ok,
                )

    return results


# ---------------------------------------------------------------------------
# Top-level picklable worker function
# (must be defined at module level to be picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_entry(args: tuple) -> list[dict[str, Any]]:
    """Entry point for each subprocess.  Runs its own asyncio event loop."""
    targets, concurrency, timeout_s, delay_s, worker_id, log_level = args
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s  W{worker_id}  %(levelname)-7s  %(message)s",
    )
    return asyncio.run(
        _run_batch_async(targets, concurrency, timeout_s, delay_s, worker_id)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(lst: list, n: int) -> list[list]:
    """Split *lst* into *n* roughly equal chunks."""
    size = math.ceil(len(lst) / n)
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _write_log_csv(rows: list[dict[str, Any]], path: str) -> None:
    fieldnames = ["doc_id", "original_link", "retrieval_status",
                  "original_content", "retrieved_content"]
    rename = {"url": "original_link", "status": "retrieval_status"}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({rename.get(k, k): v for k, v in r.items()})
    logger.info("Log CSV → %s  (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    in_base, in_ext = os.path.splitext(args.input)
    out_json = args.output or (in_base + "_retrieved" + in_ext)
    out_dir  = os.path.dirname(os.path.abspath(out_json))
    out_stem = os.path.splitext(os.path.basename(out_json))[0]
    out_csv  = os.path.join(out_dir, out_stem + "_log.csv")

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s", args.input)
    with open(args.input, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    # Build a flat lookup for applying results back to the dataset.
    doc_index: dict[str, dict[str, Any]] = {
        doc["id"]: doc
        for qi in data
        for doc in qi["documents"]
    }

    # Collect targets as plain picklable tuples.
    targets: list[Target] = [
        (doc["id"],
         doc["metadata"].get("original_link", ""),
         doc.get("content", ""))
        for doc in doc_index.values()
        if doc["metadata"].get("content_available", False)
    ]
    if args.limit:
        targets = targets[: args.limit]

    n_workers = min(args.workers, len(targets))
    logger.info(
        "Targets: %d  |  workers: %d  |  concurrency/worker: %d  "
        "|  total connections: %d  |  timeout: %ds",
        len(targets), n_workers, args.concurrency,
        n_workers * args.concurrency, args.timeout,
    )

    if args.dry_run:
        for doc_id, url, _ in targets[:5]:
            logger.info("  [dry-run] %s  →  %s", doc_id, url[:80])
        logger.info("Dry-run complete — no HTTP requests made.")
        return

    # ── Dispatch ─────────────────────────────────────────────────────────────
    chunks   = _chunk(targets, n_workers)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    worker_args = [
        (chunk, args.concurrency, args.timeout, args.delay, i + 1, log_level)
        for i, chunk in enumerate(chunks)
    ]

    t0 = time.monotonic()
    all_results: list[dict[str, Any]] = []

    if n_workers == 1:
        # Run in-process (avoids fork overhead for small jobs / smoke tests).
        all_results = _worker_entry(worker_args[0])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_entry, wa): i
                       for i, wa in enumerate(worker_args)}
            for fut in concurrent.futures.as_completed(futures):
                worker_idx = futures[fut]
                try:
                    all_results.extend(fut.result())
                except Exception as exc:
                    logger.error("Worker %d raised: %s", worker_idx + 1, exc)

    elapsed = time.monotonic() - t0

    # ── Apply results to data ─────────────────────────────────────────────────
    for row in all_results:
        if row["status"] == "OK" and row["doc_id"] in doc_index:
            doc = doc_index[row["doc_id"]]
            doc["content"] = row["retrieved_content"]
            doc["metadata"]["retrieved_content_length"] = len(row["retrieved_content"])

    # ── Summary ───────────────────────────────────────────────────────────────
    status_counts: dict[str, int] = {}
    for r in all_results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    ok     = status_counts.get("OK", 0)
    failed = len(all_results) - ok
    logger.info(
        "Done in %.1fs  |  %d OK  /  %d failed  (%.1f docs/s)",
        elapsed, ok, failed, len(all_results) / elapsed if elapsed > 0 else 0,
    )
    for status, count in sorted(status_counts.items()):
        if status != "OK":
            logger.info("  %-30s %d", status, count)

    # ── Write ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("JSON → %s", out_json)
    _write_log_csv(all_results, out_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-scrape and extract clean text for CONFACT HumC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default="data/full/confact/humc.json")
    p.add_argument("--output", "-o", default=None,
                   help="Output JSON (default: <input>_retrieved.json).")
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="Number of parallel worker processes. "
                        "Total connections = workers × concurrency.")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max simultaneous HTTP connections per worker.")
    p.add_argument("--timeout",     type=int,   default=30,
                   help="Per-request HTTP timeout in seconds.")
    p.add_argument("--delay",       type=float, default=0.3,
                   help="Base polite delay between requests per slot (seconds).")
    p.add_argument("--limit",       type=int,   default=None, metavar="N",
                   help="Process only the first N docs (smoke test).")
    p.add_argument("--dry-run",     action="store_true",
                   help="No HTTP requests, no file writes.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
