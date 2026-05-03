#!/usr/bin/env python3
"""Custom per-source content cleanup for CONFACT HumC.

Strategy
--------
1. Zero out documents from sources that only yield error/login/JS-wall pages
   (Facebook, Quora, YouTube, Twitter/X, LinkedIn, Instagram, Europol, Reddit).
2. Apply source-specific nav-strip rules for the top ~25 sources by doc count.
   Each extractor finds the article-body start anchor inside the scraped text
   and discards the nav/header blob that precedes it.
3. Fall back to the generic cleaner from clean_confact_content.py for all
   remaining sources.

Outputs
-------
  data/full/confact/humc_cleaned_v2.json         – cleaned dataset
  data/full/confact/humc_cleaned_v2_stats.csv    – per-doc stats
  data/full/confact/humc_cleaned_v2_distribution.png  – content-length histogram

Usage
-----
  python experiments/clean_confact_v2.py
  python experiments/clean_confact_v2.py --input data/full/confact/humc.json
  python experiments/clean_confact_v2.py --dry-run --sample 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Import generic cleaner from v1 ──────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from experiments.clean_confact_content import clean_content as _generic_clean
    from experiments.clean_confact_content import _is_error_page
except ImportError:
    # Allow running as top-level script from repo root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean_confact_content import clean_content as _generic_clean  # type: ignore
    from clean_confact_content import _is_error_page  # type: ignore


# ===========================================================================
# Zero-out domains  (always produce error / login / JS walls)
# ===========================================================================

_ZERO_EXACT: frozenset[str] = frozenset({
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "www.youtube.com",
    "m.youtube.com",
    "twitter.com",
    "x.com",
    "mobile.x.com",
    "www.linkedin.com",
    "www.europol.europa.eu",
    "www.instagram.com",
    "www.statista.com",       # paywalled stats database — no article text
    "www.amazon.in",          # e-commerce product pages
    "www.boat-lifestyle.com", # e-commerce product pages
    "inshorts.com",           # JS-only app, no text
    "www.premiumtimesng.com", # empty content
    "www.imf.org",            # empty / gated content
    "www.federalregister.gov",# captcha / IP-blocked
    "www.yahoo.com",          # finance nav only, no article
    "www.lemonde.fr",         # subscriber-only paywall
    # ── Junk content (sitemaps, email dumps, frequency tables) ──────────────
    "www.nps.gov",                       # E&E News email dump forwarded to NPS
    "www.dishadaily.com",                # XML sitemap — URL-only content
    "marble.nd.edu",                     # Digital collections sitemap
    "www.v6velugu.com",                  # XML sitemap — URL-only content
    "ntvtelugu.com",                     # XML sitemap — URL-only content
    "www.ntnews.com",                    # XML sitemap — URL-only content
    "telanganatoday.com",                # XML sitemap — URL-only content
    "www.yorku.ca",                      # Corpus word-frequency table, not article
    "odishapostepaper.com",              # Garbled print newspaper with CMYK markers
    "www.everycrsreport.com",            # CRS report topic index, no article text
    "www.greenlandsubhashnagar.com",     # School website nav / datesheets only
    "www.indiastat.com",                 # Stats subscription nav, no data tables
    # ── Data portals / dashboards (no article prose) ────────────────────────
    "www.oecd.org",                      # OECD data portal — nav menus + data tables
    "ukhsa-dashboard.data.gov.uk",       # UK Health dashboard — cookie consent + nav only
    # ── Video / media platforms ──────────────────────────────────────────────
    "www.dailymotion.com",               # Video player shell only
    "mobile.twitter.com",               # Twitter mobile (x.com already listed)
    # ── Stub-only pages (always return just a site name or nav shell) ────────
    "en.youturn.in",                     # Always "YouTurn YouTurn" (15 chars)
    "www.ptinews.com",                   # Always "Press Trust Of India" (20 chars)
    "library.municode.com",              # Always "Municode Library" (16 chars)
    # ── Document / template / JS-wall shells ─────────────────────────────────
    "www.scribd.com",                    # Document viewer — requires login; content is nav + AI stub
    "en.rattibha.com",                   # JS template shell with {{ logged_user.name }} placeholders
    "www.mygov.in",                      # Indian gov portal — JS wall, language nav only
    "dcist.com",                         # Scraped GDPR cookie-consent modal only (963 chars, no article)
    "www.deepayurveda.in",               # Ayurvedic e-commerce product/cart pages, no article prose
})

# Any netloc that IS or ends with one of these suffixes → zero out
_ZERO_SUFFIX: frozenset[str] = frozenset({
    "quora.com",      # quora.com and *.quora.com
    "reddit.com",     # all reddit domains — content is JS-gated / nav only
})

# Minimum cleaned-content length; shorter results are treated as nav/stub-only.
_MIN_CONTENT_LEN = 100


def _is_zero_out(netloc: str) -> bool:
    if netloc in _ZERO_EXACT:
        return True
    for s in _ZERO_SUFFIX:
        if netloc == s or netloc.endswith("." + s):
            return True
    return False


# ===========================================================================
# Shared utilities
# ===========================================================================

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


_LONG_DOC_CAP = 25_000


def _cap_text(text: str, limit: int = _LONG_DOC_CAP) -> str:
    """Truncate text at a clean paragraph/sentence boundary near limit."""
    if len(text) <= limit:
        return text
    chunk = text[:limit]
    pos = chunk.rfind("\n\n")
    if pos > limit * 0.75:
        return chunk[:pos].rstrip()
    pos = max(chunk.rfind(". "), chunk.rfind(".\n"))
    if pos > limit * 0.75:
        return chunk[:pos + 1].rstrip()
    return chunk.rstrip()


def _after(text: str, pattern: str | re.Pattern, flags: int = 0) -> str | None:
    """Return text after the first match of *pattern*, or None if not found."""
    if isinstance(pattern, str):
        m = re.search(pattern, text, flags)
    else:
        m = pattern.search(text)
    return text[m.end():].strip() if m else None


def _strip_trailing_nav(text: str) -> str:
    """Remove common trailing footer/nav blobs found after article body."""
    # Hard cutoffs: unambiguously nav, fire at 25% threshold
    hard_cutoffs = [
        r"\bSite Map\s*\|",  # footer link list (India TV News and similar)
    ]
    for pat in hard_cutoffs:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() > len(text) * 0.25:
            text = text[: m.start()].strip()

    cutoffs = [
        r"\bSign up for our newsletter\b",
        r"\bSubscribe to (our )?newsletter\b",
        r"\bMost (Popular|Read|Viewed)\b",
        r"\bRelated (Articles?|Stories?|Content)\b",
        r"\bRecommended (for you|Stories?)\b",
        r"\bYou (May Also Like|Might Also Like)\b",
        r"\bShare (this )?article\b",
        r"\bMore From\b",
        r"\bEditor's (Pick|Note)\b",
        # BBC trailing nav footer (appears after related-stories block)
        r"\bBritish Broadcasting Corporation\s+Home\b",
        r"\bBBC News Services\b",
    ]
    for pat in cutoffs:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.start() > len(text) * 0.4:
            text = text[: m.start()].strip()
            break
    return text


# ===========================================================================
# Per-source extractors
# ===========================================================================

# Each extractor receives the raw content string and returns cleaned text
# (or the original if the source-specific rule didn't fire).

# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

_WP_CITE      = re.compile(r"\[\s*\d+\s*\]")
_WP_EDIT      = re.compile(r"\s*\[\s*edit\s*(source)?\s*\]\s*")
_WP_HIDE_TOC  = re.compile(
    r"(?:move to sidebar hide|move to sidebar|hide)\s+", re.IGNORECASE
)


def _clean_wikipedia(text: str) -> str:
    anchor = "From Wikipedia, the free encyclopedia"
    idx = text.find(anchor)
    if idx >= 0:
        text = text[idx + len(anchor):]
    else:
        # Fallback: strip everything before first paragraph-length sentence
        # (heuristic: first segment ≥ 200 chars after generic clean)
        pass
    text = _WP_EDIT.sub(" ", text)
    text = _WP_CITE.sub("", text)
    text = _WP_HIDE_TOC.sub("", text)
    return _norm(text)


# ---------------------------------------------------------------------------
# BBC  (.co.uk and .com)
# ---------------------------------------------------------------------------

_BBC_ANCHOR = re.compile(
    r"About sharing\b|"
    r"By [A-Z][a-z].{3,60}BBC\b|"
    r"\bPublished\s+\d+\s+\w+\s+\d{4}\b",
    re.IGNORECASE,
)

# Anchors that mark the end of BBC nav; article content starts AFTER the match.
_BBC_NAV_END = re.compile(
    r"Getty Images\s+|"   # image credit after nav block
    r"\bNewsletters\s+",  # "Newsletters" is the last nav item before article re-title
    re.IGNORECASE,
)


def _clean_bbc(text: str) -> str:
    m = _BBC_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    m2 = _BBC_NAV_END.search(text)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    # BBC Pidgin and similar: title appears twice (once in nav, once at article start)
    m3 = _AP_TITLE_REPEAT.search(text)
    if m3:
        return _norm(_strip_trailing_nav(text[m3.start(1):]))
    # Nav-only page (e.g. Programme Information): no article content present
    if re.search(r"Skip to content.{0,60}BBC", text, re.IGNORECASE):
        return ""
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# AP News
# ---------------------------------------------------------------------------

_AP_ANCHOR = re.compile(
    r"Copyright \d{4} The Associated Press.*?\n|"
    r"(\w[\w ,'-]{10,80})\s+\1",  # title repeated twice
    re.IGNORECASE | re.DOTALL,
)
_AP_TITLE_REPEAT = re.compile(r"(.{20,120})\s+\1")


def _clean_apnews(text: str) -> str:
    m = _AP_TITLE_REPEAT.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start(1):]))
    m2 = re.search(
        r"Copyright \d{4} The Associated Press[^\n]*\n", text, re.IGNORECASE
    )
    if m2:
        after = text[m2.end():].strip()
        if len(after) > 200:
            return _norm(_strip_trailing_nav(after))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

_CNN_ANCHOR = re.compile(
    r"\d+\s+minute(?:s)?\s+read\b|"
    r"Updated\s+\d+:\d+\s+[AP]M\s+E[DS]T\b",
    re.IGNORECASE,
)
_CNN_TITLE_BEFORE = re.compile(r"([^\n]{10,120})\s*$")


def _clean_cnn(text: str) -> str:
    m = _CNN_ANCHOR.search(text)
    if m:
        # Include the article title that precedes the anchor (last ~200 chars)
        prefix = text[max(0, m.start() - 200): m.start()]
        title_m = _CNN_TITLE_BEFORE.search(prefix)
        start = title_m.start() + max(0, m.start() - 200) if title_m else m.start()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Snopes
# ---------------------------------------------------------------------------

_SNOPES_ANCHOR = re.compile(
    r"\bClaim\s*:|"
    r"\bRating\s*:|"
    r"\bWhat('s| is) (True|False|Undetermined)\b",
    re.IGNORECASE,
)


def _clean_snopes(text: str) -> str:
    m = _SNOPES_ANCHOR.search(text)
    if m:
        # Walk back to find the article title (last sentence before the anchor)
        pre = text[max(0, m.start() - 400): m.start()]
        # Take from the last sentence/heading in the pre-block
        lines = [l.strip() for l in pre.split("\n") if l.strip()]
        start_text = lines[-1] if lines else ""
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Guardian
# ---------------------------------------------------------------------------

_GUARDIAN_AGE_NOTICE = re.compile(
    r"This article is (?:more than \d+ years? old|just \w+)\.?\s*",
    re.IGNORECASE,
)
_GUARDIAN_PHOTO = re.compile(r"Photograph:\s+[^\n]{5,100}\n?", re.IGNORECASE)


def _clean_guardian(text: str) -> str:
    # Strategy 1: "This article is more than N year(s) old" appears just
    # before the article title + body.  Take everything after the notice.
    m = _GUARDIAN_AGE_NOTICE.search(text)
    if m and m.start() > 500:
        return _norm(_strip_trailing_nav(text[m.end():]))

    # Strategy 2: Find the LAST "Photograph: [credit]" line (the caption that
    # immediately precedes the article standfirst/title).
    matches = list(_GUARDIAN_PHOTO.finditer(text))
    if matches:
        last = matches[-1]
        if last.start() > 800:
            return _norm(_strip_trailing_nav(text[last.end():]))

    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Full Fact
# ---------------------------------------------------------------------------

_FULLFACT_ANCHOR = re.compile(
    r"\bWhat was claimed\b|"
    r"\bOur verdict\b|"
    r"\bOur Rating\b",
    re.IGNORECASE,
)


def _clean_fullfact(text: str) -> str:
    m = _FULLFACT_ANCHOR.search(text)
    if m:
        pre = text[max(0, m.start() - 300): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 20]
        start_text = lines[-1] if lines else None
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# NPR
# ---------------------------------------------------------------------------

_NPR_ANCHOR = re.compile(
    r"\d+:\d+\s+[AP]M\s+ET\b|"
    r"[A-Z][a-z]+ \d+, \d{4} \d+:\d+",
    re.IGNORECASE,
)


def _clean_npr(text: str) -> str:
    m = _NPR_ANCHOR.search(text)
    if m:
        pre = text[max(0, m.start() - 200): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 15]
        start_text = lines[-1] if lines else None
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# India Today
# ---------------------------------------------------------------------------

_INDIATODAY_ANCHOR = re.compile(r"\bListen to Story\b", re.IGNORECASE)
_INDIATODAY_TITLE  = re.compile(r"(.{20,120})\s+\1")


def _clean_indiatoday(text: str) -> str:
    m = _INDIATODAY_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    # Fallback: article title repeats after nav
    m2 = _INDIATODAY_TITLE.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Hindu
# ---------------------------------------------------------------------------

_THEHINDU_ANCHOR = re.compile(
    r"(?:Updated|Published)\s*[-–:]\s*\w+ \d+, \d{4}\s+\d+:\d+\s+(?:am|pm)\s*IST\b|"
    r"\w+ \d+, \d{4}\s+\d+:\d+\s+(?:am|pm)\s*IST\b",
    re.IGNORECASE,
)


_THEHINDU_PRINT = re.compile(
    r"(?:READ LATER\s+Remove\s+SEE ALL\s+PRINT|SEE ALL PRINT)\b",
    re.IGNORECASE,
)


def _clean_thehindu(text: str) -> str:
    m = _THEHINDU_ANCHOR.search(text)
    if m:
        # Skip the social-share blob (Copy link Email...Reddit READ LATER PRINT)
        # and start from the actual article body.
        skip = _THEHINDU_PRINT.search(text, m.start())
        if skip:
            return _norm(_strip_trailing_nav(text[skip.end():]))
        # Fallback: take from after anchor directly
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Times of India
# ---------------------------------------------------------------------------

_TOI_ANCHOR = re.compile(
    r"Updated:\s+\w+ \d+, \d{4}(?:,\s*\d+:\d+)?\s*IST\b|"
    r"(?:TNN|PTI|AFP|ANI|ET Bureau)\s*/?\s+\w+ \d+, \d{4}(?:,\s*\d+:\d+)?\s*IST\b",
    re.IGNORECASE,
)
_TOI_FOLLOW  = re.compile(r"\bFollow us\b", re.IGNORECASE)


def _clean_toi(text: str) -> str:
    m = _TOI_ANCHOR.search(text)
    if m:
        # Article body starts after "Follow us" or "Share AA" metadata line
        after = _TOI_FOLLOW.search(text, m.end())
        start = after.end() if after else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Economic Times (mobile)
# ---------------------------------------------------------------------------

_ET_ANCHOR = re.compile(
    r"Last Updated:\s+\w+ \d+, \d{4}(?:, \d+:\d+:\d+ [AP]M IST)?",
    re.IGNORECASE,
)
_ET_FOLLOW = re.compile(r"\bFollow us\b|\bFont Size\b", re.IGNORECASE)


def _clean_economictimes(text: str) -> str:
    m = _ET_ANCHOR.search(text)
    if m:
        after = _ET_FOLLOW.search(text, m.end())
        start = after.end() if after else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# NDTV
# ---------------------------------------------------------------------------

_NDTV_ANCHOR = re.compile(
    r"Updated:\s+\w+ \d+, \d{4}\s+\d+:\d+\s+(?:am|pm)\s+IST\b|"
    r"This Article is From\b",
    re.IGNORECASE,
)


def _clean_ndtv(text: str) -> str:
    m = _NDTV_ANCHOR.search(text)
    if m:
        pre = text[max(0, m.start() - 400): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 20]
        start_text = lines[-1] if lines else None
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Punch Nigeria
# ---------------------------------------------------------------------------

_PUNCH_SHARE = re.compile(r"Kindly share this story\s*:\s*", re.IGNORECASE)
_PUNCH_BY    = re.compile(r"\bBy\s+\w[\w ]{2,60}\s+Kindly share this story", re.IGNORECASE)


def _clean_punch(text: str) -> str:
    # Article body starts after "Kindly share this story: "
    m = _PUNCH_SHARE.search(text)
    if m and m.start() > 100:
        # Include the byline (By Author Date) that precedes "Kindly share..."
        pre = text[max(0, m.start() - 200): m.start()]
        by_m = re.search(r"\bBy\s+\w+[\w ]{1,50}\d{1,2}\w+ \w+ \d{4}", pre, re.IGNORECASE)
        if not by_m:
            by_m = re.search(r"\bBy\s+\w[\w ]{3,40}", pre, re.IGNORECASE)
        start = (m.start() - len(pre) + by_m.start()) if by_m else m.start()
        return _norm(_strip_trailing_nav(text[start:m.end()] + text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# USA Today
# ---------------------------------------------------------------------------

_USATODAY_ANCHOR = re.compile(r"\bUSA TODAY\s+[A-Z]{2,}", re.IGNORECASE)


def _clean_usatoday(text: str) -> str:
    m = _USATODAY_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# PolitiFact
# ---------------------------------------------------------------------------

_POLITI_ANCHOR = re.compile(
    r"\w[\w ,'-]{10,100}\bstated on\b|"
    r"By\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\w+ \d+, \d{4}",
    re.IGNORECASE,
)


def _clean_politifact(text: str) -> str:
    m = _POLITI_ANCHOR.search(text)
    if m:
        pre = text[max(0, m.start() - 200): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 15]
        start_text = lines[-1] if lines else None
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# FactCheck.org
# ---------------------------------------------------------------------------

def _clean_factcheck(text: str) -> str:
    # "Mailbag" is the last item of the FactCheck.org top nav bar.
    # For article pages, DVC ("Debunking Viral Claims") follows immediately,
    # then the article title + byline.  For archive/tag pages, the tag label
    # follows directly (no DVC) and is already the content listing.
    m_mb = re.search(r"\bMailbag\s+", text, re.IGNORECASE)
    if m_mb and m_mb.start() < 1200:
        rest = text[m_mb.end():]
        last_dvc = None
        for mm in re.finditer(r"\bDebunking Viral Claims\s+", rest[:300], re.IGNORECASE):
            last_dvc = mm
        if last_dvc:
            return _norm(_strip_trailing_nav(rest[last_dvc.end():]))
        return _norm(_strip_trailing_nav(rest))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# GOV.UK
# ---------------------------------------------------------------------------

_GOVUK_ANCHOR = re.compile(
    r"Search GOV\.UK\s+(?:Search\s+)?Home\b",
    re.IGNORECASE,
)


def _clean_govuk(text: str) -> str:
    m = _GOVUK_ANCHOR.search(text)
    if m:
        # After "Search GOV.UK Search Home" comes breadcrumb then article
        after = text[m.end():].strip()
        # Skip one or two breadcrumb items (short lines)
        lines = after.split("\n")
        start_idx = 0
        for i, line in enumerate(lines[:5]):
            if len(line.strip()) > 80:
                start_idx = i
                break
        return _norm(_strip_trailing_nav("\n".join(lines[start_idx:])))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# ONS.gov.uk  (Office for National Statistics)
# ---------------------------------------------------------------------------

_ONS_ANCHOR = re.compile(
    r"Skip to main content\b",
    re.IGNORECASE,
)


def _clean_ons(text: str) -> str:
    # ONS pages: find the second "Skip to main content" or deep anchor
    matches = list(_ONS_ANCHOR.finditer(text))
    if matches:
        m = matches[-1]
        after = text[m.end():].strip()
        # Strip the breadcrumb navigation (short lines before article body)
        lines = [l.strip() for l in after.split("\n")]
        body_lines = []
        in_body = False
        for line in lines:
            if not in_body and len(line) > 100:
                in_body = True
            if in_body:
                body_lines.append(line)
        if body_lines:
            return _norm(_strip_trailing_nav("\n".join(body_lines)))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Rappler
# ---------------------------------------------------------------------------

_RAPPLER_ANCHOR = re.compile(
    r"Sign in\s+Donate\s+Search\s+RAPPLER\b",
    re.IGNORECASE,
)
_RAPPLER_TITLE_REPEAT = re.compile(r"(.{20,120})\s+\1")


def _clean_rappler(text: str) -> str:
    m = _RAPPLER_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _RAPPLER_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 500:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Newsmeter
# ---------------------------------------------------------------------------

_NEWSMETER_ANCHOR = re.compile(
    r"By\s+\w+[^\n]{5,60}Published on\b",
    re.IGNORECASE,
)


def _clean_newsmeter(text: str) -> str:
    m = _NEWSMETER_ANCHOR.search(text)
    if m:
        pre = text[max(0, m.start() - 300): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 20]
        start_text = lines[-1] if lines else None
        if start_text:
            idx = text.rfind(start_text, 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# CDC.gov
# ---------------------------------------------------------------------------

_CDC_ANCHOR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d+,?\s+\d{4}\b",
    re.IGNORECASE,
)


def _clean_cdc(text: str) -> str:
    # Find first date that's followed by substantial article text
    for m in _CDC_ANCHOR.finditer(text):
        after = text[m.start():]
        if len(after) > 300:
            return _norm(_strip_trailing_nav(after))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# NBC News
# ---------------------------------------------------------------------------

_NBC_ANCHOR = re.compile(
    r"\w+ \d{1,2}, \d{4},\s+\d+:\d+\s+[AP]M\s+UTC\b",
    re.IGNORECASE,
)


def _clean_nbcnews(text: str) -> str:
    m = _NBC_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Quint
# ---------------------------------------------------------------------------

_QUINT_ANCHOR = re.compile(r"\bREMOVE AD\b", re.IGNORECASE)


def _clean_thequint(text: str) -> str:
    m = _QUINT_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# World Bank
# ---------------------------------------------------------------------------

_WB_ANCHOR = re.compile(r"\bThe World Bank in [A-Z]\w+\b")


def _clean_worldbank(text: str) -> str:
    m = _WB_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Forbes
# ---------------------------------------------------------------------------

_FORBES_ANCHOR = re.compile(r"\bEdit Story\b", re.IGNORECASE)


def _clean_forbes(text: str) -> str:
    m = _FORBES_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Livemint
# ---------------------------------------------------------------------------

_MINT_ANCHOR = re.compile(
    r"Published\s+\d+ \w+ \d{4},\s+\d+:\d+\s+(?:AM|PM)\s+IST\b",
    re.IGNORECASE,
)


def _clean_livemint(text: str) -> str:
    m = _MINT_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Vishvas News
# ---------------------------------------------------------------------------

_VISHVAS_DATE  = re.compile(
    r"Published:\s+\w+ \d+,\s+\d{4}\s+at\s+\d+:\d+\s+[AP]M",
    re.IGNORECASE,
)
_VISHVAS_SHARE = re.compile(r"\bShare\b", re.IGNORECASE)


def _clean_vishvasnews(text: str) -> str:
    m = _VISHVAS_DATE.search(text)
    if m:
        share = _VISHVAS_SHARE.search(text, m.end())
        start = share.end() if share and share.start() - m.end() < 30 else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Daily Mail
# ---------------------------------------------------------------------------

_MAIL_DATE     = re.compile(
    r"Published:\s+\d+:\d+\s+BST,\s+\d+ \w+ \d{4}", re.IGNORECASE
)
_MAIL_COMMENTS = re.compile(r"\bView comments\b", re.IGNORECASE)


def _clean_dailymail(text: str) -> str:
    m = _MAIL_DATE.search(text)
    if m:
        vc = _MAIL_COMMENTS.search(text, m.end())
        start = vc.end() if vc else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Time magazine
# ---------------------------------------------------------------------------

_TIME_ANCHOR = re.compile(
    r"\d+ minute read\s+By\s+[\w ,]+\s+\w+ \d+,\s+\d{4}\s+\d+:\d+\s+[AP]M\s+E[DS]T\b",
    re.IGNORECASE,
)


def _clean_time(text: str) -> str:
    m = _TIME_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# OpIndia
# ---------------------------------------------------------------------------

_OPINDIA_ANCHOR = re.compile(r"Updated:\s+\d+ \w+,\s+\d{4}\b", re.IGNORECASE)


def _clean_opindia(text: str) -> str:
    # Nav is ~9500 chars; article marker appears late
    matches = list(_OPINDIA_ANCHOR.finditer(text))
    if matches:
        m = matches[-1]
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# ThePrint
# ---------------------------------------------------------------------------

_PRINT_IST       = re.compile(r"\d+:\d+\s+(?:am|pm)\s+IST\b", re.IGNORECASE)
_PRINT_TEXTSIZE  = re.compile(r"\bText Size\s*:\s*A[-+]\s*A[-+]\b", re.IGNORECASE)
_PRINT_LINKEDIN  = re.compile(r"\bLinkedin\b", re.IGNORECASE)


def _clean_theprint(text: str) -> str:
    m = _PRINT_IST.search(text)
    if m:
        # Skip "Follow Us : Text Size: A- A+" social-share widget
        ts = _PRINT_TEXTSIZE.search(text, m.end())
        if ts and ts.start() - m.end() < 100:
            return _norm(_strip_trailing_nav(text[ts.end():]))
        li = _PRINT_LINKEDIN.search(text, m.end())
        start = li.end() if li and li.start() - m.end() < 80 else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Philstar
# ---------------------------------------------------------------------------

_PHILSTAR_ANCHOR = re.compile(
    r"-\s*The Philippine Star\s+\w+ \d+,\s+\d{4}", re.IGNORECASE
)


def _clean_philstar(text: str) -> str:
    m = _PHILSTAR_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Washington Post
# ---------------------------------------------------------------------------

_WAPO_ANCHOR = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r" \d{1,2},\s+\d{4}\s+at\s+\d+:\d+\s+[ap]\.m\.",
    re.IGNORECASE,
)


def _clean_washingtonpost(text: str) -> str:
    m = _WAPO_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# New Indian Express
# ---------------------------------------------------------------------------

_NIE_DATE   = re.compile(
    r"Updated on\s*:\s*\d+ \w+ \d{4},\s*\d+:\d+ (?:am|pm)\b", re.IGNORECASE
)
_NIE_COPIED = re.compile(r"\bCopied\b", re.IGNORECASE)


def _clean_newindianexpress(text: str) -> str:
    m = _NIE_DATE.search(text)
    if m:
        copied = _NIE_COPIED.search(text, m.end())
        start = copied.end() if copied and copied.start() - m.end() < 60 else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# CBS News
# ---------------------------------------------------------------------------

_CBS_ANCHOR = re.compile(
    r"\w+ \d+, \d{4}\s*/\s*\d+:\d+\s+[AP]M\s+\w+\b",
    re.IGNORECASE,
)


def _clean_cbsnews(text: str) -> str:
    m = _CBS_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# IFS (Institute for Fiscal Studies)
# ---------------------------------------------------------------------------

_IFS_ANCHOR = re.compile(r"\bTaxlab Key Questions\b", re.IGNORECASE)


def _clean_ifs(text: str) -> str:
    # Taxlab-specific pages have a reliable anchor
    matches = list(_IFS_ANCHOR.finditer(text))
    if matches:
        m = matches[-1]
        return _norm(_strip_trailing_nav(text[m.end():]))
    # Other IFS pages: title appears at pos 0 before "| Institute for Fiscal Studies",
    # then a ~3 KB nav block, then the title again just before the article body.
    title_sep = re.search(r"\s*\|?\s*Institute for Fiscal Studies\s*", text[:300], re.I)
    if title_sep and title_sep.start() > 10:
        title = text[:title_sep.start()].strip()
        if len(title) >= 20:
            title_pos = text.find(title, 2500)
            if title_pos > 0:
                return _norm(_cap_text(_strip_trailing_nav(text[title_pos:])))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Jagran Josh
# ---------------------------------------------------------------------------

_JAGRAN_ANCHOR = re.compile(
    r"By\s+\w+\s+\w+\s+\w+ \d+,\s+\d{4},\s+\d+:\d+\s+IST\b",
    re.IGNORECASE,
)


def _clean_jagranjosh(text: str) -> str:
    m = _JAGRAN_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# ABP Live
# ---------------------------------------------------------------------------

_ABP_ANCHOR = re.compile(
    r"Updated at\s*:\s*\d+ \w+ \d{4}\s+\d+:\d+\s+[AP]M\s*\(IST\)",
    re.IGNORECASE,
)


def _clean_abplive(text: str) -> str:
    m = _ABP_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Al Jazeera
# ---------------------------------------------------------------------------

_AJ_ANCHOR = re.compile(
    r"Published On\s+\d+ \w+ \d{4}\b", re.IGNORECASE
)


def _clean_aljazeera(text: str) -> str:
    m = _AJ_ANCHOR.search(text)
    if m:
        # Date is often repeated once; skip the duplicate and any Correction notice
        rest = text[m.end():]
        m2 = re.match(r"\s+\d+ \w+ \d{4}\b", rest)
        start = m2.end() if m2 else 0
        return _norm(_strip_trailing_nav(rest[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Britannica
# ---------------------------------------------------------------------------

_BRIT_ANCHOR = re.compile(
    r"Ask the Chatbot a Question\s+Ask the Chatbot a Question\b",
    re.IGNORECASE,
)


def _clean_britannica(text: str) -> str:
    m = _BRIT_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    # Fallback: "Article History" is earlier but still before the body
    m2 = re.search(r"\bArticle History\b", text, re.IGNORECASE)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# OECD
# ---------------------------------------------------------------------------

_OECD_ANCHOR = re.compile(r"\bClose menu\s+OECD Topics\b", re.IGNORECASE)


def _clean_oecd(text: str) -> str:
    m = _OECD_ANCHOR.search(text)
    if m:
        # Skip the repeated topic-title header (2x repetitions before body)
        rest = text[m.end():]
        title_m = re.search(r"([A-Z][\w ()/-]{15,100})\s+\1\s+", rest)
        start = title_m.end() if title_m else 0
        return _norm(_strip_trailing_nav(rest[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# BoomLive
# ---------------------------------------------------------------------------

_BOOM_ANCHOR = re.compile(
    r"By\s*-\s*[\w ]+\|\s*\d+ \w+ \d{4}\s+\d+:\d+\s+[AP]M\s+GMT\b",
    re.IGNORECASE,
)


def _clean_boomlive(text: str) -> str:
    m = _BOOM_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# News18
# ---------------------------------------------------------------------------

_NEWS18_ANCHOR  = re.compile(
    r"Last Updated:\s+\w+ \d+,\s+\d{4},\s+\d+:\d+\s+IST\b", re.IGNORECASE
)
_NEWS18_FOLLOW  = re.compile(r"\bFollow us\b", re.IGNORECASE)


def _clean_news18(text: str) -> str:
    m = _NEWS18_ANCHOR.search(text)
    if m:
        fu = _NEWS18_FOLLOW.search(text, m.end())
        start = fu.end() if fu and fu.start() - m.end() < 150 else m.end()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Testbook
# ---------------------------------------------------------------------------

_TESTBOOK_ANCHOR = re.compile(r"Last Updated on\s+\w+ \d+,\s+\d{4}\b", re.IGNORECASE)


def _clean_testbook(text: str) -> str:
    m = _TESTBOOK_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# ABC News
# ---------------------------------------------------------------------------

_ABC_ANCHOR = re.compile(
    r"By\s+[\w ,]+\w+\s+\w+ \d+,\s+\d{4},\s+\d+:\d+\s+[AP]M\b",
    re.IGNORECASE,
)


def _clean_abcnews(text: str) -> str:
    m = _ABC_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Healthline
# ---------------------------------------------------------------------------

_HEALTH_ANCHOR = re.compile(
    r"(?:Written by|Medically reviewed by)\b", re.IGNORECASE
)


def _clean_healthline(text: str) -> str:
    m = _HEALTH_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Sky News
# ---------------------------------------------------------------------------

_SKY_ANCHOR = re.compile(r"\bWhy you can trust Sky News\b", re.IGNORECASE)


def _clean_skynews(text: str) -> str:
    m = _SKY_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Scroll.in
# ---------------------------------------------------------------------------

_SCROLL_ANCHOR = re.compile(r"\bRead in App\b", re.IGNORECASE)


def _clean_scroll(text: str) -> str:
    m = _SCROLL_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Newschecker
# ---------------------------------------------------------------------------

_NEWSCHECKER_ANCHOR = re.compile(r"\bClaim\s+[A-Z]")


def _clean_newschecker(text: str) -> str:
    m = _NEWSCHECKER_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Republic World
# ---------------------------------------------------------------------------

_REPUBLIC_ANCHOR = re.compile(
    r"Published\s+\d+:\d+\s+IST,\s+\w+ \d+\w*\s+\d{4}", re.IGNORECASE
)


def _clean_republicworld(text: str) -> str:
    m = _REPUBLIC_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# GMA Network
# ---------------------------------------------------------------------------

_GMA_ANCHOR = re.compile(r"\bFiltered By:\s+Topstories\b", re.IGNORECASE)


def _clean_gmanetwork(text: str) -> str:
    m = _GMA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Fox News
# ---------------------------------------------------------------------------

_FOX_ANCHOR = re.compile(
    r"Fox News Published\s+\w+ \d+,?\s+\d{4}\s+\d+:\d+(?:am|pm)\s+EDT\b",
    re.IGNORECASE,
)


def _clean_foxnews(text: str) -> str:
    m = _FOX_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# CNBC
# ---------------------------------------------------------------------------

_CNBC_ANCHOR = re.compile(
    r"Published \w+,\s+\w+ \d+ \d{4}\s+\d+:\d+\s+[AP]M\s+EDT\b",
    re.IGNORECASE,
)


def _clean_cnbc(text: str) -> str:
    m = _CNBC_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Euronews
# ---------------------------------------------------------------------------

_EURO_ANCHOR = re.compile(
    r"Published on \d{2}/\d{2}/\d{4}\s*[-–]\s*\d{1,2}:\d{2}\s+GMT",
    re.IGNORECASE,
)


def _clean_euronews(text: str) -> str:
    m = _EURO_ANCHOR.search(text)
    if m:
        # Walk back to include "Copyright By [author]" line if present
        pre = text[max(0, m.start() - 80): m.start()]
        copy_m = re.search(r"\bCopyright\s+By\b", pre, re.IGNORECASE)
        start = (m.start() - len(pre) + copy_m.start()) if copy_m else m.start()
        return _norm(_strip_trailing_nav(text[start:]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Moneycontrol
# ---------------------------------------------------------------------------

_MC_ANCHOR = re.compile(
    r"Moneycontrol News \w+ \d+,\s+\d{4}\s*/\s*\d+:\d+\s+IST\b",
    re.IGNORECASE,
)


def _clean_moneycontrol(text: str) -> str:
    m = _MC_ANCHOR.search(text)
    if m:
        # Article title appears just before the timestamp
        pre = text[max(0, m.start() - 300): m.start()]
        lines = [l.strip() for l in pre.split("\n") if len(l.strip()) > 20]
        if lines:
            idx = text.rfind(lines[-1], 0, m.start())
            if idx >= 0:
                return _norm(_strip_trailing_nav(text[idx:]))
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# BusinessToday.in
# ---------------------------------------------------------------------------

_BT_ANCHOR = re.compile(
    r"BusinessToday\.In Updated \w+ \d+,\s+\d{4},\s+\d+:\d+\s+[AP]M\s+IST\b",
    re.IGNORECASE,
)


def _clean_businesstoday(text: str) -> str:
    m = _BT_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# WHO
# ---------------------------------------------------------------------------

_WHO_ANCHOR = re.compile(
    r"Home\s*/\s*Newsroom\s*/\s*(?:Fact sheets|News releases|Detail)\s*/",
    re.IGNORECASE,
)
# "Member States Portal Home /" marks the transition from the nav sidebar to
# the breadcrumb trail that precedes every WHO article body.
_WHO_PORTAL_ANCHOR = re.compile(
    r"Member States Portal\s+Home\s*/",
    re.IGNORECASE,
)


def _clean_who(text: str) -> str:
    m = _WHO_PORTAL_ANCHOR.search(text)
    if m:
        rest = text[m.end():]
        # Breadcrumb path segments are separated by " / " (with spaces).
        # Find the last such separator to skip the full breadcrumb.
        last_sep = None
        for mm in re.finditer(r"\s/\s", rest[:300]):
            last_sep = mm
        if last_sep:
            return _norm(_strip_trailing_nav(rest[last_sep.end():]))
        return _norm(_strip_trailing_nav(rest))
    m2 = _WHO_ANCHOR.search(text)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Whitehouse.gov
# ---------------------------------------------------------------------------

_WH_ANCHOR = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December) \d{1,2}, \d{4}\s+"
    r"(?:FACT SHEET|STATEMENT|PROCLAMATION|EXECUTIVE ORDER|REMARKS|NOTICE|"
    r"PRESS BRIEFING|BY THE PRESIDENT|PRESIDENTIAL)\b",
    re.IGNORECASE,
)


_WH_SEARCH_ANCHOR = re.compile(
    r"Washington, DC \d{4,5}\s+To search this site, enter a search term Search\s+",
    re.IGNORECASE,
)


def _clean_whitehouse(text: str) -> str:
    m = _WH_ANCHOR.search(text)
    if m and m.start() > 200:
        return _norm(_strip_trailing_nav(text[m.start():]))
    # Blog posts and non-standard WH pages: nav ends at the site-search bar
    m2 = _WH_SEARCH_ANCHOR.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# PBS NewsHour
# ---------------------------------------------------------------------------

_PBS_ANCHOR = re.compile(r"Leave your feedback\b", re.IGNORECASE)


def _clean_pbs(text: str) -> str:
    m = _PBS_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Dubawa
# ---------------------------------------------------------------------------

_DUBAWA_ANCHOR = re.compile(
    r"Send an email\s+\w+ \d+,\s+\d{4}\b", re.IGNORECASE
)


def _clean_dubawa(text: str) -> str:
    m = _DUBAWA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Ferret (Scotland)
# ---------------------------------------------------------------------------

_FERRET_ANCHOR = re.compile(r"\bby Ferret Journalists\b", re.IGNORECASE)


def _clean_theferret(text: str) -> str:
    m = _FERRET_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Florida Health
# ---------------------------------------------------------------------------

_FLHEALTH_ANCHOR = re.compile(
    r"Tallahassee,\s+Fla\.\s*[–—-]\s*", re.IGNORECASE
)


def _clean_floridahealth(text: str) -> str:
    m = _FLHEALTH_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Brookings Institution
# ---------------------------------------------------------------------------

_BROOK_ANCHOR = re.compile(r"Sign Up Commentary\b", re.IGNORECASE)


def _clean_brookings(text: str) -> str:
    m = _BROOK_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    # Fallback: title repeats before article body
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 500:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Pew Research Center
# ---------------------------------------------------------------------------

_PEW_ANCHOR = re.compile(
    r"Report \| \w+ \d+, \d{4}\b", re.IGNORECASE
)


def _clean_pewresearch(text: str) -> str:
    m = _PEW_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text, 1000)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Sentencing Project
# ---------------------------------------------------------------------------

_SP_ANCHOR = re.compile(
    r"Download PDF\s+Table of Contents\b|Download\s+Download PDF\b",
    re.IGNORECASE,
)


def _clean_sentencingproject(text: str) -> str:
    m = _SP_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 500:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Resolution Foundation
# ---------------------------------------------------------------------------

_RESFOUND_ANCHOR = re.compile(r"\bSkip to main content\b", re.IGNORECASE)


def _clean_resolutionfoundation(text: str) -> str:
    m = _RESFOUND_ANCHOR.search(text)
    if m:
        after = text[m.end():]
        # Skip search-box lines (short lines) before the article paragraph
        lines = after.split("\n")
        body: list[str] = []
        for line in lines:
            if not body and len(line.strip()) < 60:
                continue
            body.append(line)
        return _norm(_strip_trailing_nav("\n".join(body))) if body else _generic_clean(text)[0]
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Alzheimer's Society
# ---------------------------------------------------------------------------

_ALZ_ANCHOR = re.compile(
    r"Dementia professionals\s+Information\b|"
    r"Save this information\s+Get a copy\b",
    re.IGNORECASE,
)


def _clean_alzheimers(text: str) -> str:
    m = _ALZ_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The BMJ
# ---------------------------------------------------------------------------

_BMJ_ANCHOR = re.compile(r"doi: https://doi\.org/10\.", re.IGNORECASE)


def _clean_bmj(text: str) -> str:
    m = _BMJ_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Schools Week
# ---------------------------------------------------------------------------

_SW_ANCHOR = re.compile(r"Home\s*[›>]\s*All news\s+News\b", re.IGNORECASE)


def _clean_schoolsweek(text: str) -> str:
    m = _SW_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Human Rights Watch
# ---------------------------------------------------------------------------

_HRW_ANCHOR = re.compile(r"\bShare this via Printer\b", re.IGNORECASE)


_HRW_FOOTER = re.compile(r"Protecting Rights, Saving Lives", re.IGNORECASE)


def _clean_hrw(text: str) -> str:
    matches = list(_HRW_ANCHOR.finditer(text))
    chosen = None
    for m in reversed(matches):
        after50 = text[m.end(): m.end() + 60]
        if not _HRW_FOOTER.search(after50):
            chosen = m
            break
    if chosen is None and matches:
        chosen = matches[0]
    if chosen:
        return _norm(_strip_trailing_nav(text[chosen.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Drishti IAS
# ---------------------------------------------------------------------------

_DRISHTI_ANCHOR = re.compile(r"\bfilter Hide Menu\b", re.IGNORECASE)


def _clean_drishtiias(text: str) -> str:
    m = _DRISHTI_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# MDPI (open-access journals)
# ---------------------------------------------------------------------------

_MDPI_ANCHOR = re.compile(r"\bAbstract\b", re.IGNORECASE)


def _clean_mdpi(text: str) -> str:
    # The journal-list nav fills thousands of chars; Abstract appears deep
    for m in _MDPI_ANCHOR.finditer(text):
        if m.start() > 5000:
            return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# EUAA (EU Agency for Asylum)
# ---------------------------------------------------------------------------

_EUAA_ANCHOR = re.compile(
    r"\bShare\s+Print\s+",
    re.IGNORECASE,
)


def _clean_euaa(text: str) -> str:
    m = _EUAA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Alcohol Change UK
# ---------------------------------------------------------------------------

_ALCC_ANCHOR = re.compile(r"\bAlcohol in the UK\b", re.IGNORECASE)


def _clean_alcoholchange(text: str) -> str:
    # Take the LAST occurrence — the nav contains it once, the article body after it
    matches = list(_ALCC_ANCHOR.finditer(text))
    if matches:
        m = matches[-1]
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Scottish Government (gov.scot)
# ---------------------------------------------------------------------------

_GOVSCOT_ANCHOR = re.compile(r"\bSkip to main content\b", re.IGNORECASE)


def _clean_govscot(text: str) -> str:
    m = _GOVSCOT_ANCHOR.search(text)
    if m:
        after = text[m.end():].strip()
        # Skip short breadcrumb / cookie-notice lines
        lines = after.split("\n")
        body: list[str] = []
        for line in lines:
            if not body and len(line.strip()) < 80:
                continue
            body.append(line)
        return _norm(_strip_trailing_nav("\n".join(body))) if body else _generic_clean(text)[0]
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Skeptics Stack Exchange
# ---------------------------------------------------------------------------

_SE_ANCHOR = re.compile(r"\bAsk Question\s+Asked\s+\d+", re.IGNORECASE)


def _clean_skepticsse(text: str) -> str:
    m = _SE_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Mayo Clinic
# ---------------------------------------------------------------------------

_MAYO_ANCHOR = re.compile(
    r"\bAbout\s+(?:Doctors\s+&\s+Departments\s+)?Print\s+Overview\b|"
    r"\bPrint\s+Sections\b",
    re.IGNORECASE,
)


def _clean_mayoclinic(text: str) -> str:
    m = _MAYO_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Nature (academic journal)
# ---------------------------------------------------------------------------

_NATURE_ANCHOR = re.compile(r"\bAbstract\b", re.IGNORECASE)
_NATURE_NEWS_ANCHOR = re.compile(r"\bTwitter\s+Facebook\s+Email\b", re.IGNORECASE)


def _clean_nature(text: str) -> str:
    m = _NATURE_ANCHOR.search(text)
    if m and m.start() > 300:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _NATURE_NEWS_ANCHOR.search(text)
    if m2 and m2.start() > 300:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# IFLScience
# ---------------------------------------------------------------------------

_IFLSCI_ANCHOR = re.compile(r"\bDOWNLOAD PDF VERSION\b", re.IGNORECASE)


def _clean_iflscience(text: str) -> str:
    # Second occurrence of "DOWNLOAD PDF VERSION" is right before the article
    matches = list(_IFLSCI_ANCHOR.finditer(text))
    if len(matches) >= 2:
        return _norm(_strip_trailing_nav(text[matches[1].end():]))
    if matches:
        return _norm(_strip_trailing_nav(text[matches[0].end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# USAID
# ---------------------------------------------------------------------------

_USAID_ANCHOR = re.compile(r"\bOVERVIEW\b")


def _clean_usaid(text: str) -> str:
    m = _USAID_ANCHOR.search(text)
    if m and m.start() > 500:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 300:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# EPA
# ---------------------------------------------------------------------------

_EPA_PRESS_ANCHOR = re.compile(
    r"Contact Information\s+EPA Press Office\s*\([^)]+\)\s*", re.IGNORECASE
)
_EPA_NAV_END = re.compile(r"\bContact Us\b")


def _clean_epa(text: str) -> str:
    m = _EPA_PRESS_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    for m2 in _EPA_NAV_END.finditer(text):
        if m2.start() < 2000:
            return _norm(_strip_trailing_nav(text[m2.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Mail & Guardian (South Africa)
# ---------------------------------------------------------------------------

_MG_ANCHOR = re.compile(
    r"Article\s*/\s*\d+ \w+ \d{4}\s+[A-Z]|"
    r"\bBy\s+Sapa\b|"
    r"\bBy\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
    re.IGNORECASE,
)


def _clean_mgza(text: str) -> str:
    for m in _MG_ANCHOR.finditer(text):
        if m.start() > 300:
            return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# PubMed / NCBI
# ---------------------------------------------------------------------------

_PUBMED_ANCHOR = re.compile(r"\. doi: 10\.\d{4}/", re.IGNORECASE)


def _clean_pubmed(text: str) -> str:
    m = _PUBMED_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# DNA India
# ---------------------------------------------------------------------------

_DNA_ANCHOR = re.compile(
    r"\bLATEST WEBSTORY TRENDING\b|\bTRENDING NOW\b",
    re.IGNORECASE,
)


def _clean_dnaindia(text: str) -> str:
    # Article body is buried after a sidebar of headlines; find last "TRENDING" marker
    matches = list(_DNA_ANCHOR.finditer(text))
    if matches:
        m = matches[-1]
        # Article body follows the last trending section
        rest = text[m.end():]
        # Skip headlines (short lines) until a long paragraph appears
        lines = rest.split("\n")
        body: list[str] = []
        for line in lines:
            if not body and len(line.strip()) < 80:
                continue
            body.append(line)
        if body:
            return _norm(_strip_trailing_nav("\n".join(body)))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 300:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# India.com
# ---------------------------------------------------------------------------

_INDIA_COM_ANCHOR = re.compile(
    r"Published:\s+\w+ \d+,\s+\d{4}\s+\d+:\d+\s+[AP]M\s+IST\s+By\b",
    re.IGNORECASE,
)


def _clean_indiacom(text: str) -> str:
    m = _INDIA_COM_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Logically Facts
# ---------------------------------------------------------------------------

_LOGICALLY_ANCHOR = re.compile(
    r"\bThe Verdict\s+(?:True|False|Mostly True|Misleading|Partly True|Unverified)\b",
    re.IGNORECASE,
)


def _clean_logically(text: str) -> str:
    m = _LOGICALLY_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# OnManorama
# ---------------------------------------------------------------------------

_MANORAMA_ANCHOR = re.compile(r"\bEmail sent successfully\b", re.IGNORECASE)


def _clean_onmanorama(text: str) -> str:
    m = _MANORAMA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# India TV News
# ---------------------------------------------------------------------------

_ITVN_ANCHOR = re.compile(r"\bURL copied Follow us on\b", re.IGNORECASE)


def _clean_indiatvnews(text: str) -> str:
    m = _ITVN_ANCHOR.search(text)
    if m:
        # After the share bar comes image caption then article title + body
        rest = text[m.end():]
        # Skip "Image Source : INSTAGRAM" type captions
        rest = re.sub(r"^Image Source\s*:[^\n]{0,60}\n?", "", rest.lstrip(), flags=re.IGNORECASE)
        return _norm(_strip_trailing_nav(rest))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# The Sun (UK)
# ---------------------------------------------------------------------------

_SUN_ANCHOR = re.compile(
    r"Published\s*:\s*\d+:\d+,\s+\d+ \w+ \d{4}", re.IGNORECASE
)


def _clean_thesun(text: str) -> str:
    m = _SUN_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Trump White House Archives
# ---------------------------------------------------------------------------

_TRUMPWH_ISSUES_ANCHOR = re.compile(r"\bAll Issues\b")


def _clean_trumpwh(text: str) -> str:
    m = _TRUMPWH_ISSUES_ANCHOR.search(text)
    if m and m.start() < 1000:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# PIB (Press Information Bureau India)
# ---------------------------------------------------------------------------

_PIB_ANCHOR = re.compile(
    r"\d{2}-\w+-\d{4}\s+\d+:\d+\s+IST\b", re.IGNORECASE
)


def _clean_pib(text: str) -> str:
    m = _PIB_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Deccan Herald
# ---------------------------------------------------------------------------

_DH_ANCHOR = re.compile(
    r"Last Updated\s*:\s*\d+ \w+ \d{4},\s*\d+:\d+\s+IST\b", re.IGNORECASE
)


def _clean_deccanherald(text: str) -> str:
    m = _DH_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Africa Check
# ---------------------------------------------------------------------------

_AC_ANCHOR = re.compile(
    r"Published on \d{2} \w+ \d{4}\b", re.IGNORECASE
)


def _clean_africacheck(text: str) -> str:
    m = _AC_ANCHOR.search(text)
    if m:
        # "This article is more than N years old" may follow before body
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Vice
# ---------------------------------------------------------------------------

_VICE_ANCHOR = re.compile(
    r"By\s+\w+[\w ]+\w+\s+\w+ \d+,\s+\d{4},\s+\d+:\d+(?:am|pm)\b",
    re.IGNORECASE,
)


def _clean_vice(text: str) -> str:
    m = _VICE_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# National Geographic
# ---------------------------------------------------------------------------

_NATGEO_ANCHOR = re.compile(
    r"By\s+[\w ]+\s+\w+ \d+,\s+\d{4}\s+[•·]\s+\d+ min read\b",
    re.IGNORECASE,
)


def _clean_nationalgeographic(text: str) -> str:
    m = _NATGEO_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Medical News Today
# ---------------------------------------------------------------------------

def _clean_medicalnewstoday(text: str) -> str:
    m = _HEALTH_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# Swarajya Magazine
# ---------------------------------------------------------------------------

_SWARA_ANCHOR = re.compile(
    r"\w+ \d{1,2}, \d{4},\s+\d+:\d+\s+[AP]M\s*\|\s*Updated\b", re.IGNORECASE
)


def _clean_swarajyamag(text: str) -> str:
    m = _SWARA_ANCHOR.search(text)
    if m:
        after = text[m.end():]
        sign_in = re.search(r"\bSign In\b", after[:400])
        if sign_in:
            after = after[sign_in.end():]
        return _norm(_strip_trailing_nav(after))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# AOL
# ---------------------------------------------------------------------------

_AOL_ANCHOR = re.compile(
    r"Updated\s+\w+ \d+,\s+\d{4}\s+at\s+\d+:\d+\s+[AP]M\b", re.IGNORECASE
)


def _clean_aol(text: str) -> str:
    m = _AOL_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ---------------------------------------------------------------------------
# FactChecker.in
# ---------------------------------------------------------------------------

_FCINDIA_ANCHOR = re.compile(r"\|\s+\d+ \w+ \d{4}\b", re.IGNORECASE)


def _clean_factcheckerindia(text: str) -> str:
    m = _FCINDIA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ===========================================================================
# Round-3 extractors: remaining 3-5 doc sources
# ===========================================================================

# kippra.or.ke / www.ojp.gov / www.indiabudget.gov.in / sansad.in
# All are PDF documents that start directly with content.
def _clean_kippra(text: str) -> str:
    return _norm(_strip_trailing_nav(text))

def _clean_ojp(text: str) -> str:
    return _norm(_strip_trailing_nav(text))

def _clean_indiabudget(text: str) -> str:
    return _norm(_strip_trailing_nav(text))

def _clean_sansad(text: str) -> str:
    return _norm(_strip_trailing_nav(text))


# www.cambridge.org: PDFs start at content; web pages use "Published online by Cambridge"
_CAMB_ANCHOR = re.compile(r"Published online by Cambridge University Press", re.IGNORECASE)

def _clean_cambridge(text: str) -> str:
    m = _CAMB_ANCHOR.search(text)
    if m and m.start() > 300:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _norm(_strip_trailing_nav(text))


# www.ageuk.org.uk: PDFs start at content; web page has 12K+ chars of nav
_AGEUK_WEB_ANCHOR = re.compile(r"\bIf you'?re\s+\d{2}\s+or\s+(?:over|above|older)\b", re.IGNORECASE)

def _clean_ageuk(text: str) -> str:
    text = text.lstrip("﻿")
    if not re.search(r"Skip to content", text[:100]):
        return _norm(_strip_trailing_nav(text))
    m = _AGEUK_WEB_ANCHOR.search(text)
    if m and m.start() > 5000:
        return _norm(_strip_trailing_nav(text[m.start():]))
    m2 = _AP_TITLE_REPEAT.search(text, 5000)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# www.amnesty.org.uk: PDFs start at content; web page ends nav at last "Donate"
_AMNESTY_DONATE = re.compile(r"\bDonate\b")

def _clean_amnesty(text: str) -> str:
    if not re.search(r"Skip to main content", text[:100]):
        return _norm(_strip_trailing_nav(text))
    matches = [m for m in _AMNESTY_DONATE.finditer(text[:600])]
    if matches:
        return _norm(_strip_trailing_nav(text[matches[-1].end():]))
    m2 = _AP_TITLE_REPEAT.search(text, 100)
    if m2:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# www.refugeecouncil.org.uk: PDFs start at content; web pages use breadcrumb + "Ways to get involved"
def _clean_refugeecouncil(text: str) -> str:
    m_bc = re.search(r"Home\s*/\s*(?:Information|Latest|News)\s*/", text)
    if m_bc and m_bc.start() > 1000:
        m2 = re.search(r"\bWays to get involved\s+", text[m_bc.start():])
        if m2:
            return _norm(_strip_trailing_nav(text[m_bc.start() + m2.end():]))
        return _norm(_strip_trailing_nav(text[m_bc.end():]))
    return _norm(_strip_trailing_nav(text))


# www.justice.gov: PDFs start at content; web press release uses "For Immediate Release"
_DOJ_ANCHOR = re.compile(r"For Immediate Release\s+Office of Public Affairs\s+", re.IGNORECASE)

def _clean_justice(text: str) -> str:
    m = _DOJ_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _norm(_strip_trailing_nav(text))


# www.dailypioneer.com: web articles have "Weekday, DD Month YYYY | " before byline
_PIONEER_ANCHOR = re.compile(r"[A-Z][a-z]+day, \d{2} \w+ \d{4} \| ")

def _clean_dailypioneer(text: str) -> str:
    m = _PIONEER_ANCHOR.search(text)
    if m and m.start() > 300:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _norm(_strip_trailing_nav(text))


# ltoportal.ph: nav ends at "LTMS Portal Offices"
_LTO_ANCHOR = re.compile(r"\bLTMS Portal Offices\s+", re.IGNORECASE)

def _clean_ltoportal(text: str) -> str:
    m = _LTO_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# www.knbs.or.ke: homepage nav ends near "Our mission is clear"; PDFs start clean
_KNBS_ANCHOR = re.compile(r"Our mission is clear: We exist\b", re.IGNORECASE)

def _clean_knbs(text: str) -> str:
    m = _KNBS_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _norm(_strip_trailing_nav(text))


# dailytrust.com: Nigerian news; WAT timestamp + byline marks article start
_DTRUST_ANCHOR = re.compile(r"\d{1,2}:\d{2}(?::\d{2})? WAT\s+(?:By [^\n]{3,60}\s+)?")

def _clean_dailytrust(text: str) -> str:
    m = _DTRUST_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 200:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# www.thisdaylive.com: "Section | N months/years ago" tag precedes article body
_THISDAY_ANCHOR = re.compile(
    r"(?:Politics|Business|Life\s*&\s*Style|Health|Sport|Education|International|"
    r"Nigeria|Africa|Entertainment)\s*\|\s*\d+\s+(?:month|year)s?\s+ago\s+",
    re.IGNORECASE,
)

def _clean_thisdaylive(text: str) -> str:
    m = _THISDAY_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# www.thelallantop.com: Hindi articles; "IST )" closes timestamp metadata
_LALLANTOP_ANCHOR = re.compile(r"IST\s*\)\s*(?:Small\s+Medium\s+Large\s+){1,3}(?:Advertisement\s+)?", re.IGNORECASE)

def _clean_thelallantop(text: str) -> str:
    m = _LALLANTOP_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# www.ssa.gov: PDFs start clean; web page ends nav at "Call Resolution Site"
_SSA_ANCHOR = re.compile(r"Call Resolution Site\s+", re.IGNORECASE)

def _clean_ssa(text: str) -> str:
    m = _SSA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _norm(_strip_trailing_nav(text))


# www.freep.com: Detroit Free Press byline marks article start
_FREEP_ANCHOR = re.compile(r"\bDetroit Free Press\b", re.IGNORECASE)

def _clean_freep(text: str) -> str:
    m = _FREEP_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# www.europarl.europa.eu: Q&A pages or Fact Sheets; PDFs start clean
_EP_QA_ANCHOR = re.compile(r"Parliamentary question - E-\d+/\d{4}", re.IGNORECASE)
_EP_FS_ANCHOR = re.compile(r"Load fact sheet in pdf format\s+", re.IGNORECASE)

def _clean_europarl(text: str) -> str:
    m = _EP_QA_ANCHOR.search(text)
    if m and m.start() > 300:
        return _norm(_strip_trailing_nav(text[m.start():]))
    m2 = _EP_FS_ANCHOR.search(text)
    if m2 and m2.start() > 300:
        return _norm(_strip_trailing_nav(text[m2.end():]))
    return _norm(_strip_trailing_nav(text))


# www.coca-cola.com: "menu" (mobile nav close) is last nav item before article
_COCACOLA_MENU = re.compile(r"\bmenu\s+", re.IGNORECASE)

def _clean_cocacola(text: str) -> str:
    matches = list(_COCACOLA_MENU.finditer(text))
    if matches:
        return _norm(_strip_trailing_nav(text[matches[-1].end():]))
    return _generic_clean(text)[0]


# www.coca-colacompany.com: "Click to search" precedes FAQ/article content
_CCCORP_ANCHOR = re.compile(r"Click to search\s+", re.IGNORECASE)

def _clean_cocacolacompany(text: str) -> str:
    matches = list(_CCCORP_ANCHOR.finditer(text))
    if matches:
        return _norm(_strip_trailing_nav(text[matches[-1].end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 500:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# www.cbse.gov.in: strip BOM + "| Central Board of Secondary Education" title suffix
def _clean_cbse(text: str) -> str:
    text = text.lstrip("﻿")
    m = re.search(r"\| Central Board of Secondary Education\s+", text[:300])
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _norm(_strip_trailing_nav(text))


# www.businessdailyafrica.com: "- N min read" marker precedes article body
_BDA_ANCHOR = re.compile(r"- \d+ min read\s+", re.IGNORECASE)

def _clean_businessdailyafrica(text: str) -> str:
    m = _BDA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# worldpopulationreview.com: map color config ends nav; "Reverse Colors" is last item
_WPR_ANCHOR = re.compile(r"Click on a country for details\.\s*X\s+", re.IGNORECASE)

def _clean_worldpopulationreview(text: str) -> str:
    m = _WPR_ANCHOR.search(text)
    if m and m.start() < 1000:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = _AP_TITLE_REPEAT.search(text)
    if m2 and m2.start() > 100:
        return _norm(_strip_trailing_nav(text[m2.start(1):]))
    return _generic_clean(text)[0]


# theweek.com: "By [Author] published/last updated [date]" byline
_THEWEEK_ANCHOR = re.compile(
    r"By [A-Z][a-z]+(?: [A-Z][a-z]+)* (?:published|last updated) \d{1,2} \w+ \d{4}\b",
    re.IGNORECASE,
)

def _clean_theweek(text: str) -> str:
    m = _THEWEEK_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# prsindia.org: "Firstname Lastname - Month DD, YYYY" author-date line
_PRS_ANCHOR = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+ - \w+ \d{1,2}, \d{4}\s+", re.IGNORECASE)

def _clean_prsindia(text: str) -> str:
    m = _PRS_ANCHOR.search(text, 500)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    m2 = re.search(r"\bThe PRS Blog\b", text)
    if m2:
        after = text[m2.end():]
        lines = after.split("\n")
        body: list[str] = []
        for ln in lines:
            if not body and len(ln.strip()) < 80:
                continue
            body.append(ln)
        if body:
            return _norm(_strip_trailing_nav("\n".join(body)))
    return _generic_clean(text)[0]


# newsroom.heart.org: AHA press releases open with "DALLAS, Month DD, YYYY —"
_AHA_ANCHOR = re.compile(r"\bDALLAS,\s+\w+ \d{1,2},\s+\d{4}\s*[—–-]\s*", re.IGNORECASE)

def _clean_newsroomheart(text: str) -> str:
    m = _AHA_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# fivethirtyeight.com: archived articles; date "Mon. DD, YYYY , at H:MM AM/PM"
_F38_ANCHOR = re.compile(r"[A-Z][a-z]{2}\. \d{1,2}, \d{4} , at \d{1,2}:\d{2} [AP]M\b")

def _clean_fivethirtyeight(text: str) -> str:
    m = _F38_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.start():]))
    return _generic_clean(text)[0]


# explore-education-statistics.service.gov.uk: GOV.UK stats pages
_EES_ANCHOR = re.compile(r"Home Find statistics and data\s+", re.IGNORECASE)

def _clean_exploregovuk(text: str) -> str:
    m = _EES_ANCHOR.search(text)
    if m:
        return _norm(_strip_trailing_nav(text[m.end():]))
    return _generic_clean(text)[0]


# ===========================================================================
# Round 4: long-doc web-page extractors (nav-strip + 25K cap)
# ===========================================================================

# elibrary.judiciary.gov.ph — PH Supreme Court E-Library; nav ~800 chars before decision
_ELIBRARY_PH_ANCHOR = re.compile(r"\bDECISION\s+[A-Z]+,\s+J\.\s*:", re.IGNORECASE)

def _clean_elibrary_ph(text: str) -> str:
    m = _ELIBRARY_PH_ANCHOR.search(text[:3000])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# news.ycombinator.com — Hacker News; skip title + nav (~200 chars) before comments
_HN_ANCHOR = re.compile(r"\bsubmit login\s+", re.IGNORECASE)

def _clean_hackernews(text: str) -> str:
    m = _HN_ANCHOR.search(text[:600])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# obamawhitehouse.archives.gov — Obama WH archive; "Search form Search" marks nav end
_OBAMAWH_ANCHOR = re.compile(r"Search form Search\s+", re.IGNORECASE)

def _clean_obamawhitehouse(text: str) -> str:
    m = _OBAMAWH_ANCHOR.search(text[:4000])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 500)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# www.jacksoncountyprosecutor.com — press-release archive; first "Posted on:" is first story
_JCP_ANCHOR = re.compile(r"Posted on:\s+\w+ \d{1,2},\s+\d{4}")

def _clean_jacksoncounty(text: str) -> str:
    m = _JCP_ANCHOR.search(text[:800])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# kenyalaw.org — Kenya Law legal database; "Case Summary:" marks start of case content
_KENYALAW_ANCHOR = re.compile(r"Case Summary:\s+", re.IGNORECASE)

def _clean_kenyalaw(text: str) -> str:
    m = _KENYALAW_ANCHOR.search(text[:3000])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 200)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# www.mackinac.org — Mackinac Center policy studies; title repeats after "Download PDF"
_MACKINAC_ANCHOR = re.compile(r"\bDownload PDF\s+", re.IGNORECASE)

def _clean_mackinac(text: str) -> str:
    m = _MACKINAC_ANCHOR.search(text[:600])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# www.newageislam.com — Islamic news/opinion; "(date, NewAgeIslam.Com) Comment Comment" ends nav
_NEWAGEISLAM_ANCHOR = re.compile(r"NewAgeIslam\.Com\)\s+Comment\s+Comment\s+", re.IGNORECASE)

def _clean_newageislam(text: str) -> str:
    m = _NEWAGEISLAM_ANCHOR.search(text[:4000])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 500)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# www.presidency.ucsb.edu — American Presidency Project; APP disclaimer precedes document
_PRESIDENCY_ANCHOR = re.compile(
    r"This is not the analysis or work product of the American Presidency Project[^.]+\.\s*\]?\s*|"
    r"\[APP Note:[^\]]+\]\s*",
    re.IGNORECASE,
)

def _clean_presidency(text: str) -> str:
    m = _PRESIDENCY_ANCHOR.search(text[:3000])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 200)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# sc.judiciary.gov.ph — PH Supreme Court news; nav repeats heavily; content after offset 1000
_SCPHIL_ANCHOR = re.compile(r"The Supreme Court (?:has|ruled|declared|held)\b", re.IGNORECASE)

def _clean_scjudiciary_ph(text: str) -> str:
    m = _SCPHIL_ANCHOR.search(text, 1000)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start():])))
    m2 = _AP_TITLE_REPEAT.search(text, 200)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# www.airtel.in / www.acko.com / www.civilsdaily.com / www.jordanharbinger.com /
# www.yalelawjournal.org / jacquithurlowlippisch.com
# — all have moderate nav; title repeats after nav end

def _clean_airtel(text: str) -> str:
    m = _AP_TITLE_REPEAT.search(text, 500)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


def _clean_acko(text: str) -> str:
    m = _AP_TITLE_REPEAT.search(text, 800)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


_CIVILSDAILY_NEWSCARD = re.compile(r"\bNewsCards\s+Backgrounder\s+", re.IGNORECASE)


def _clean_civilsdaily(text: str) -> str:
    # "NewsCards Backgrounder" is the content-type tag that appears between
    # the first (nav) and second (article-heading) occurrences of the title.
    m = _CIVILSDAILY_NEWSCARD.search(text, 400)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 400)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


_JH_ANCHOR = re.compile(r"\bReading Time:\s*\d+\s+minutes?\s+", re.IGNORECASE)


def _clean_jordanharbinger(text: str) -> str:
    m = _JH_ANCHOR.search(text)
    if m and m.start() < 1500:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    m2 = _AP_TITLE_REPEAT.search(text, 200)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


def _clean_yalelawjournal(text: str) -> str:
    m = _AP_TITLE_REPEAT.search(text, 100)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start(1):])))
    m2 = re.search(r"VOLUME \d{3}\s+\d{4}-\d{4}\s+Forum\s+", text[:600])
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.end():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


def _clean_jacquithurlowlippisch(text: str) -> str:
    m2 = re.search(r"Menu Skip to content\s+", text[:400], re.IGNORECASE)
    if m2:
        return _norm(_cap_text(_strip_trailing_nav(text[m2.end():])))
    m = _AP_TITLE_REPEAT.search(text, 100)
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.start(1):])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# threadreaderapp.com — Twitter/X thread unroller; content follows view count
_THREADREADER_ANCHOR = re.compile(r"\d[\d,]+ views\s+", re.IGNORECASE)

def _clean_threadreaderapp(text: str) -> str:
    m = _THREADREADER_ANCHOR.search(text[:800])
    if m:
        return _norm(_cap_text(_strip_trailing_nav(text[m.end():])))
    return _norm(_cap_text(_strip_trailing_nav(text)))


# ===========================================================================
# Dispatch table
# ===========================================================================

_EXTRACTORS: dict[str, Any] = {
    # ── Original 23 ────────────────────────────────────────────────────────
    "en.wikipedia.org":             _clean_wikipedia,
    "www.bbc.co.uk":                _clean_bbc,
    "www.bbc.com":                  _clean_bbc,
    "apnews.com":                   _clean_apnews,
    "www.cnn.com":                  _clean_cnn,
    "www.snopes.com":               _clean_snopes,
    "www.theguardian.com":          _clean_guardian,
    "fullfact.org":                 _clean_fullfact,
    "www.npr.org":                  _clean_npr,
    "www.indiatoday.in":            _clean_indiatoday,
    "www.thehindu.com":             _clean_thehindu,
    "timesofindia.indiatimes.com":  _clean_toi,
    "m.economictimes.com":          _clean_economictimes,
    "www.ndtv.com":                 _clean_ndtv,
    "punchng.com":                  _clean_punch,
    "www.usatoday.com":             _clean_usatoday,
    "www.politifact.com":           _clean_politifact,
    "www.factcheck.org":            _clean_factcheck,
    "www.gov.uk":                   _clean_govuk,
    "www.ons.gov.uk":               _clean_ons,
    "www.rappler.com":              _clean_rappler,
    "newsmeter.in":                 _clean_newsmeter,
    "www.cdc.gov":                  _clean_cdc,
    # ── New 31 (≥5 docs, not already zeroed) ───────────────────────────────
    "www.nbcnews.com":              _clean_nbcnews,
    "www.thequint.com":             _clean_thequint,
    "www.worldbank.org":            _clean_worldbank,
    "www.forbes.com":               _clean_forbes,
    "www.livemint.com":             _clean_livemint,
    "www.vishvasnews.com":          _clean_vishvasnews,
    "www.dailymail.co.uk":          _clean_dailymail,
    "time.com":                     _clean_time,
    "www.opindia.com":              _clean_opindia,
    "theprint.in":                  _clean_theprint,
    "www.philstar.com":             _clean_philstar,
    "www.washingtonpost.com":       _clean_washingtonpost,
    "www.newindianexpress.com":     _clean_newindianexpress,
    "www.cbsnews.com":              _clean_cbsnews,
    "ifs.org.uk":                   _clean_ifs,
    "www.jagranjosh.com":           _clean_jagranjosh,
    "news.abplive.com":             _clean_abplive,
    "www.aljazeera.com":            _clean_aljazeera,
    "www.britannica.com":           _clean_britannica,
    "www.oecd.org":                 _clean_oecd,
    "www.boomlive.in":              _clean_boomlive,
    "www.news18.com":               _clean_news18,
    "testbook.com":                 _clean_testbook,
    "abcnews.go.com":               _clean_abcnews,
    "www.healthline.com":           _clean_healthline,
    "news.sky.com":                 _clean_skynews,
    "scroll.in":                    _clean_scroll,
    "newschecker.in":               _clean_newschecker,
    "www.republicworld.com":        _clean_republicworld,
    "www.gmanetwork.com":           _clean_gmanetwork,
    # ── New batch (3–5 docs, not already zeroed) ────────────────────────────
    "www.foxnews.com":              _clean_foxnews,
    "www.cnbc.com":                 _clean_cnbc,
    "www.euronews.com":             _clean_euronews,
    "www.moneycontrol.com":         _clean_moneycontrol,
    "www.businesstoday.in":         _clean_businesstoday,
    "www.who.int":                  _clean_who,
    "www.whitehouse.gov":           _clean_whitehouse,
    "www.pbs.org":                  _clean_pbs,
    "dubawa.org":                   _clean_dubawa,
    "theferret.scot":               _clean_theferret,
    "www.floridahealth.gov":        _clean_floridahealth,
    "www.brookings.edu":            _clean_brookings,
    "www.pewresearch.org":          _clean_pewresearch,
    "www.sentencingproject.org":    _clean_sentencingproject,
    "www.resolutionfoundation.org": _clean_resolutionfoundation,
    "www.alzheimers.org.uk":        _clean_alzheimers,
    "www.bmj.com":                  _clean_bmj,
    "schoolsweek.co.uk":            _clean_schoolsweek,
    "www.hrw.org":                  _clean_hrw,
    "www.drishtiias.com":           _clean_drishtiias,
    "www.mdpi.com":                 _clean_mdpi,
    "euaa.europa.eu":               _clean_euaa,
    "alcoholchange.org.uk":         _clean_alcoholchange,
    "www.gov.scot":                 _clean_govscot,
    "skeptics.stackexchange.com":   _clean_skepticsse,
    "www.mayoclinic.org":           _clean_mayoclinic,
    "www.nature.com":               _clean_nature,
    "www.iflscience.com":           _clean_iflscience,
    "www.usaid.gov":                _clean_usaid,
    "www.epa.gov":                  _clean_epa,
    "mg.co.za":                     _clean_mgza,
    "pubmed.ncbi.nlm.nih.gov":      _clean_pubmed,
    "www.dnaindia.com":             _clean_dnaindia,
    "www.india.com":                _clean_indiacom,
    "www.logicallyfacts.com":       _clean_logically,
    "www.onmanorama.com":           _clean_onmanorama,
    "www.indiatvnews.com":          _clean_indiatvnews,
    "www.thesun.co.uk":             _clean_thesun,
    "trumpwhitehouse.archives.gov": _clean_trumpwh,
    "pib.gov.in":                   _clean_pib,
    "www.deccanherald.com":         _clean_deccanherald,
    "africacheck.org":              _clean_africacheck,
    "www.vice.com":                 _clean_vice,
    "www.nationalgeographic.com":   _clean_nationalgeographic,
    "www.medicalnewstoday.com":     _clean_medicalnewstoday,
    "swarajyamag.com":              _clean_swarajyamag,
    "www.aol.com":                  _clean_aol,
    "www.factchecker.in":           _clean_factcheckerindia,
    # ── Round 3: remaining 3-5 doc sources ─────────────────────────────────
    "kippra.or.ke":                                        _clean_kippra,
    "www.ojp.gov":                                         _clean_ojp,
    "www.indiabudget.gov.in":                              _clean_indiabudget,
    "sansad.in":                                           _clean_sansad,
    "www.cambridge.org":                                   _clean_cambridge,
    "www.ageuk.org.uk":                                    _clean_ageuk,
    "www.amnesty.org.uk":                                  _clean_amnesty,
    "www.refugeecouncil.org.uk":                           _clean_refugeecouncil,
    "www.justice.gov":                                     _clean_justice,
    "www.dailypioneer.com":                                _clean_dailypioneer,
    "ltoportal.ph":                                        _clean_ltoportal,
    "www.knbs.or.ke":                                      _clean_knbs,
    "dailytrust.com":                                      _clean_dailytrust,
    "www.thisdaylive.com":                                 _clean_thisdaylive,
    "www.thelallantop.com":                                _clean_thelallantop,
    "www.ssa.gov":                                         _clean_ssa,
    "www.freep.com":                                       _clean_freep,
    "www.europarl.europa.eu":                              _clean_europarl,
    "www.coca-cola.com":                                   _clean_cocacola,
    "www.coca-colacompany.com":                            _clean_cocacolacompany,
    "www.cbse.gov.in":                                     _clean_cbse,
    "www.businessdailyafrica.com":                         _clean_businessdailyafrica,
    "worldpopulationreview.com":                           _clean_worldpopulationreview,
    "theweek.com":                                         _clean_theweek,
    "prsindia.org":                                        _clean_prsindia,
    "newsroom.heart.org":                                  _clean_newsroomheart,
    "fivethirtyeight.com":                                 _clean_fivethirtyeight,
    "explore-education-statistics.service.gov.uk":         _clean_exploregovuk,
    # ── Round 4: long-doc web-page sources (nav-strip + 25K cap) ───────────
    "elibrary.judiciary.gov.ph":                           _clean_elibrary_ph,
    "news.ycombinator.com":                                _clean_hackernews,
    "obamawhitehouse.archives.gov":                        _clean_obamawhitehouse,
    "www.jacksoncountyprosecutor.com":                     _clean_jacksoncounty,
    "kenyalaw.org":                                        _clean_kenyalaw,
    "www.mackinac.org":                                    _clean_mackinac,
    "www.newageislam.com":                                 _clean_newageislam,
    "www.presidency.ucsb.edu":                             _clean_presidency,
    "sc.judiciary.gov.ph":                                 _clean_scjudiciary_ph,
    "www.airtel.in":                                       _clean_airtel,
    "www.acko.com":                                        _clean_acko,
    "www.civilsdaily.com":                                 _clean_civilsdaily,
    "www.jordanharbinger.com":                             _clean_jordanharbinger,
    "www.yalelawjournal.org":                              _clean_yalelawjournal,
    "jacquithurlowlippisch.com":                           _clean_jacquithurlowlippisch,
    "threadreaderapp.com":                                 _clean_threadreaderapp,
}


# ===========================================================================
# Main per-document cleaning function
# ===========================================================================

def clean_document(text: str, netloc: str) -> tuple[str, str]:
    """Return (cleaned_text, method_used).

    method_used is one of: 'zero_out', 'error_page', 'custom', 'generic'.
    """
    if not text or not text.strip():
        return "", "zero_out"

    # 1. Zero-out: known error / login / JS-wall domains
    if _is_zero_out(netloc):
        return "", "zero_out"

    # 2. Generic error-page fingerprint catch-all
    if _is_error_page(text):
        return "", "error_page"

    # 3. Source-specific extractor — cap at 25K
    extractor = _EXTRACTORS.get(netloc)
    if extractor:
        cleaned = extractor(text)
        if len(cleaned) > _LONG_DOC_CAP:
            cleaned = _cap_text(cleaned)
        if len(cleaned) < _MIN_CONTENT_LEN:
            return "", "error_page"
        return cleaned, "custom"

    # 4. Generic cleaner (v1) — cap oversized docs at 25K chars
    cleaned, _ = _generic_clean(text)
    if len(cleaned) > _LONG_DOC_CAP:
        cleaned = _cap_text(cleaned)
    if len(cleaned) < _MIN_CONTENT_LEN:
        return "", "error_page"
    return cleaned, "generic"


# ===========================================================================
# I/O helpers
# ===========================================================================

def _percentile_rank(sorted_vals: list[int], v: int) -> float:
    import bisect
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    below = bisect.bisect_left(sorted_vals, v)
    equal = bisect.bisect_right(sorted_vals, v) - below
    return round(100.0 * (below + 0.5 * equal) / n, 1)


def _annotate_percentiles(rows: list[dict]) -> None:
    orig    = sorted(r["original_len"] for r in rows)
    cleaned = sorted(r["cleaned_len"]  for r in rows)
    for r in rows:
        po = _percentile_rank(orig,    r["original_len"])
        pc = _percentile_rank(cleaned, r["cleaned_len"])
        r["pct_original"] = po
        r["pct_cleaned"]  = pc
        r["top10_original"]  = "Y" if po >= 90 else "N"
        r["top10_cleaned"]   = "Y" if pc >= 90 else "N"
        r["b10_20_original"] = "Y" if 80 <= po < 90 else "N"
        r["b10_20_cleaned"]  = "Y" if 80 <= pc < 90 else "N"


def _write_stats(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Stats → %s  (%d rows)", path, len(rows))


def _plot(rows: list[dict], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    orig_lens    = [r["original_len"] for r in rows]
    cleaned_lens = [r["cleaned_len"]  for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CONFACT HumC v2 – Content Length Distribution",
                 fontsize=12, y=1.01)

    for ax, vals, title, color in [
        (axes[0], orig_lens,    "Original",   "#4878CF"),
        (axes[1], cleaned_lens, "Cleaned v2", "#6ACC65"),
    ]:
        ax.hist(vals, bins=50, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Documents")
        ax.yaxis.grid(True, linewidth=0.4, color="0.85")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    non_empty = [c for c in cleaned_lens if c > 0]
    summary = (
        f"N docs = {len(rows)}\n"
        f"Zeroed = {sum(1 for r in rows if r['method'] in ('zero_out','error_page'))}\n"
        f"Non-empty after clean = {len(non_empty)}\n"
        f"Median cleaned = {int(np.median(non_empty)):,} chars\n"
        f"Mean cleaned = {int(np.mean(non_empty)):,} chars"
    ) if non_empty else f"N docs = {len(rows)}\nAll zeroed"
    fig.text(0.5, -0.05, summary, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="0.7"))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot → %s", path)


# ===========================================================================
# Main processing
# ===========================================================================

def _process(
    input_path: str,
    output_path: str | None,
    dry_run: bool,
    sample_n: int,
) -> None:
    in_base, in_ext = os.path.splitext(input_path)
    out_json  = output_path or (in_base + "_cleaned_v2" + in_ext)
    out_dir   = os.path.dirname(os.path.abspath(out_json))
    stem      = os.path.splitext(os.path.basename(out_json))[0]
    stats_csv = os.path.join(out_dir, stem + "_stats.csv")
    plot_png  = os.path.join(out_dir, stem + "_distribution.png")

    logger.info("Loading %s", input_path)
    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    stat_rows: list[dict] = []
    samples: list[tuple[str, str, str, str]] = []

    method_counts: dict[str, int] = {}
    total_before = total_after = 0

    for qi in data:
        for doc in qi["documents"]:
            raw: str = doc.get("content") or ""
            link     = doc.get("metadata", {}).get("original_link", "")
            netloc   = urlparse(link).netloc if link else ""

            cleaned, method = clean_document(raw, netloc)

            before_len = len(raw)
            after_len  = len(cleaned)
            total_before += before_len
            total_after  += after_len
            method_counts[method] = method_counts.get(method, 0) + 1

            doc["content"] = cleaned
            doc["metadata"]["original_content_length"] = before_len
            doc["metadata"]["cleaned_content_length"]  = after_len
            doc["metadata"]["content_available"]       = bool(cleaned)
            doc["metadata"]["clean_method"]            = method

            stat_rows.append({
                "doc_id":          doc["id"],
                "instance_id":     qi["id"],
                "source":          netloc,
                "credibility":     doc.get("credibility", ""),
                "original_len":    before_len,
                "cleaned_len":     after_len,
                "reduction_chars": before_len - after_len,
                "reduction_pct":   round(100.0 * (before_len - after_len) / before_len, 1)
                                   if before_len > 0 else 0.0,
                "method":          method,
                "content_available": "Y" if cleaned else "N",
            })

            if sample_n > 0 and len(samples) < sample_n * 3 and before_len > 400:
                samples.append((doc["id"], netloc, raw[:600], cleaned[:600]))

    _annotate_percentiles(stat_rows)

    n_total   = len(stat_rows)
    reduction = 100.0 * (1 - total_after / max(total_before, 1))
    logger.info("Docs processed : %d", n_total)
    logger.info("Chars before   : %d   Chars after: %d   Reduction: %.1f%%",
                total_before, total_after, reduction)
    logger.info("Methods:")
    for m, c in sorted(method_counts.items()):
        logger.info("  %-15s %d  (%.1f%%)", m, c, 100 * c / max(n_total, 1))

    if sample_n > 0:
        import random
        random.shuffle(samples)
        for doc_id, src, before, after in samples[:sample_n]:
            print(f"\n{'='*60}\nDOC: {doc_id}  SOURCE: {src}")
            print(f"--- BEFORE ---\n{before}")
            print(f"--- AFTER  ---\n{after}")

    if dry_run:
        logger.info("Dry-run — no files written.")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("JSON → %s", out_json)

    _write_stats(stat_rows, stats_csv)
    _plot(stat_rows, plot_png)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Custom per-source content cleanup for CONFACT HumC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default="data/full/confact/humc.json")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="Print N random before/after samples.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    _process(args.input, args.output, dry_run=args.dry_run, sample_n=args.sample)


if __name__ == "__main__":
    main()
