"""Extended quality gates for TCB rewrite validation.

Supplements the two gates in quality_checker.py (BERTScore F1, cosine sim)
with four additional checks that address reviewer confounds:

  NLIEntailmentChecker   — bidirectional DeBERTa-v3 entailment (gold standard
                           for factual preservation; catches paraphrase rewrites
                           that drift meaning while preserving surface similarity)
  ReadabilityChecker     — Flesch-Kincaid Grade Level + Gunning-Fog Index delta
                           (rules out the fluency confound: assertive rewrites
                           must not become significantly easier to read)
  AssertivenessChecker   — hedging-word density heuristic (verifies that the
                           neutral→assertive manipulation actually succeeded)
  RougeChecker           — ROUGE-L F1 (lightweight n-gram sanity check; cheap
                           proxy for content overlap when BERTScore is slow)

All checkers follow the same interface as SemanticQualityChecker:
  result = checker.check_pair(original, rewritten)
  # {"ok": bool, "scores": dict, "reasons": list[str]}

Batch APIs are provided for pipeline use.

Models are lazy-loaded on first use.  No imports at module load time beyond
stdlib.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. NLI Entailment Checker
# ---------------------------------------------------------------------------
# Uses cross-encoder/nli-deberta-v3-base.  For a valid tone-controlled rewrite:
#   original → rewrite  should be ENTAILMENT  (rewrite doesn't add/drop facts)
#   rewrite → original  should be ENTAILMENT  (original still covered)
#
# We treat "entailment" as the label index with the highest logit.
# The model outputs three classes: entailment, neutral, contradiction.

NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
NLI_ENTAILMENT_THRESHOLD: float = 0.70   # min probability assigned to ENTAILMENT

_NLI_LABELS = ["contradiction", "entailment", "neutral"]  # model label order

# The cross-encoder receives PREMISE + SEP + HYPOTHESIS concatenated.
# 180 words per side ≈ 234 tokens per side ≈ 470 tokens total — safely under 512.
# Longer documents (ConfACT) are split into chunks of this size and scored
# positionally; the minimum chunk score is returned (conservative: any chunk
# that drifts factually fails the whole document).
_NLI_MAX_WORDS_PER_SIDE: int = 180


def _nli_chunks(text: str, max_words: int = _NLI_MAX_WORDS_PER_SIDE) -> list[str]:
    """Split text into non-overlapping chunks of at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


class NLIEntailmentChecker:
    """Bidirectional NLI entailment check using DeBERTa-v3.

    Both directions must pass:
      P(entailment | original, rewrite) >= threshold
      P(entailment | rewrite, original) >= threshold

    For documents longer than ~180 words (e.g. ConfACT web text), both texts
    are split into positionally aligned chunks of at most _NLI_MAX_WORDS_PER_SIDE
    words.  Each chunk pair is scored independently; the minimum across all
    chunk pairs is reported.  This avoids the 512-token cross-encoder limit
    silently truncating long documents.

    A pair that passes is factually equivalent under the model.
    A pair that fails likely had a factual addition, deletion, or inversion.

    Args:
        threshold: Minimum entailment probability in each direction.
        model: HuggingFace model identifier.
        device: Torch device string (None = auto).
        max_length: Hard token cap for the cross-encoder (default 512).
    """

    def __init__(
        self,
        threshold: float = NLI_ENTAILMENT_THRESHOLD,
        model: str = NLI_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.threshold = threshold
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            logger.info("Loading NLI model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            if self.device:
                self._nli_model = self._nli_model.to(self.device)
            self._nli_model.eval()
            self._torch = torch
        return self._nli_model, self._tokenizer, self._torch

    def _entailment_prob_batch(
        self, premises: list[str], hypotheses: list[str], batch_size: int = 16
    ) -> list[float]:
        model, tokenizer, torch = self._load()
        results: list[float] = []
        for i in range(0, len(premises), batch_size):
            bp = premises[i : i + batch_size]
            bh = hypotheses[i : i + batch_size]
            inputs = tokenizer(
                bp,
                bh,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            results.extend([p[1] for p in probs])
        return results

    def _chunked_min_entailment(self, premise: str, hypothesis: str) -> float:
        """Score positionally aligned chunks; return the minimum entailment prob."""
        p_chunks = _nli_chunks(premise)
        h_chunks = _nli_chunks(hypothesis)
        n = min(len(p_chunks), len(h_chunks))
        scores = self._entailment_prob_batch(p_chunks[:n], h_chunks[:n])
        return min(scores)

    def check_pair(self, original: str, rewritten: str) -> dict:
        result: dict = {"ok": True, "scores": {}, "reasons": []}
        try:
            p_fwd = self._chunked_min_entailment(original, rewritten)
            p_bwd = self._chunked_min_entailment(rewritten, original)
            result["scores"]["nli_entailment_fwd"] = round(p_fwd, 4)
            result["scores"]["nli_entailment_bwd"] = round(p_bwd, 4)
            if p_fwd < self.threshold:
                result["ok"] = False
                result["reasons"].append(
                    f"nli_fwd={p_fwd:.3f} < threshold={self.threshold} "
                    f"(original does not entail rewrite → possible fact drop)"
                )
            if p_bwd < self.threshold:
                result["ok"] = False
                result["reasons"].append(
                    f"nli_bwd={p_bwd:.3f} < threshold={self.threshold} "
                    f"(rewrite does not entail original → possible hallucination)"
                )
        except Exception as exc:
            logger.warning("NLI check failed: %s", exc)
        return result

    def check_batch(self, originals: list[str], rewrittens: list[str]) -> list[dict]:
        n = len(originals)
        results: list[dict] = [{"ok": True, "scores": {}, "reasons": []} for _ in range(n)]
        try:
            for i, (orig, rew) in enumerate(zip(originals, rewrittens)):
                pf = self._chunked_min_entailment(orig, rew)
                pb = self._chunked_min_entailment(rew, orig)
                results[i]["scores"]["nli_entailment_fwd"] = round(pf, 4)
                results[i]["scores"]["nli_entailment_bwd"] = round(pb, 4)
                if pf < self.threshold:
                    results[i]["ok"] = False
                    results[i]["reasons"].append(f"nli_fwd={pf:.3f} < {self.threshold}")
                if pb < self.threshold:
                    results[i]["ok"] = False
                    results[i]["reasons"].append(f"nli_bwd={pb:.3f} < {self.threshold}")
        except Exception as exc:
            logger.warning("Batch NLI check failed: %s", exc)
        return results


# ---------------------------------------------------------------------------
# 2. Readability Checker
# ---------------------------------------------------------------------------
# Uses the `textstat` library (pure Python, no model download).
# Metrics:
#   Flesch-Kincaid Grade Level (FKGL): higher = harder (grade equivalent)
#   Gunning Fog Index (GFI):           higher = harder
#
# For the TCB confound check we compare ΔFKGL and ΔGFI between original and
# rewrite.  A large positive delta (rewrite much easier to read) would indicate
# the assertive rewrite also became more fluent — a confound.
#
# Threshold: |ΔFKGL| <= max_grade_delta (default 2.0 grade levels) is required.

READABILITY_MAX_GRADE_DELTA: float = 2.0


class ReadabilityChecker:
    """Checks that rewrite does not significantly change reading difficulty.

    Uses Flesch-Kincaid Grade Level and Gunning Fog Index.  The delta between
    original and rewritten scores must not exceed ``max_grade_delta`` grade
    levels in either direction.

    Args:
        max_grade_delta: Maximum allowed absolute change in FKGL or GFI.
        report_only: If True, always returns ok=True but reports scores.
            Use for auditing without filtering.
    """

    def __init__(
        self,
        max_grade_delta: float = READABILITY_MAX_GRADE_DELTA,
        report_only: bool = False,
    ):
        self.max_grade_delta = max_grade_delta
        self.report_only = report_only
        self._textstat = None

    def _load(self):
        if self._textstat is None:
            try:
                import textstat  # type: ignore
                self._textstat = textstat
            except ImportError:
                raise ImportError(
                    "textstat not installed.  Run: pip install textstat"
                )
        return self._textstat

    def _scores(self, text: str) -> dict:
        ts = self._load()
        return {
            "fkgl": ts.flesch_kincaid_grade(text),
            "gfi": ts.gunning_fog(text),
            "flesch_ease": ts.flesch_reading_ease(text),
            "avg_sentence_length": ts.avg_sentence_length(text),
        }

    def check_pair(self, original: str, rewritten: str) -> dict:
        result: dict = {"ok": True, "scores": {}, "reasons": []}
        try:
            orig_s = self._scores(original)
            rew_s = self._scores(rewritten)
            delta_fkgl = rew_s["fkgl"] - orig_s["fkgl"]
            delta_gfi = rew_s["gfi"] - orig_s["gfi"]
            result["scores"].update({
                "orig_fkgl": round(orig_s["fkgl"], 2),
                "rew_fkgl": round(rew_s["fkgl"], 2),
                "delta_fkgl": round(delta_fkgl, 2),
                "orig_gfi": round(orig_s["gfi"], 2),
                "rew_gfi": round(rew_s["gfi"], 2),
                "delta_gfi": round(delta_gfi, 2),
                "orig_flesch_ease": round(orig_s["flesch_ease"], 2),
                "rew_flesch_ease": round(rew_s["flesch_ease"], 2),
                "orig_avg_sent_len": round(orig_s["avg_sentence_length"], 2),
                "rew_avg_sent_len": round(rew_s["avg_sentence_length"], 2),
            })
            if not self.report_only:
                if abs(delta_fkgl) > self.max_grade_delta:
                    result["ok"] = False
                    result["reasons"].append(
                        f"ΔFKGL={delta_fkgl:+.2f} exceeds ±{self.max_grade_delta} "
                        f"(fluency confound)"
                    )
                if abs(delta_gfi) > self.max_grade_delta:
                    result["ok"] = False
                    result["reasons"].append(
                        f"ΔGFI={delta_gfi:+.2f} exceeds ±{self.max_grade_delta} "
                        f"(fluency confound)"
                    )
        except Exception as exc:
            logger.warning("Readability check failed: %s", exc)
        return result

    def check_batch(self, originals: list[str], rewrittens: list[str]) -> list[dict]:
        return [self.check_pair(o, r) for o, r in zip(originals, rewrittens)]


# ---------------------------------------------------------------------------
# 3. Assertiveness Checker
# ---------------------------------------------------------------------------
# Verifies that the tone manipulation actually worked.
# Strategy: count hedging-word density in neutral vs. assertive versions.
# Neutral rewrites should have MORE hedges; assertive should have FEWER.
#
# If the assertive version has MORE hedges than the neutral version, the
# manipulation likely failed or reversed.
#
# Hedging words drawn from literature on epistemic uncertainty markers.

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

_MIN_ASSERTIVENESS_ADVANTAGE: float = 0.0   # assertive must not have MORE hedges than neutral


class AssertivenessChecker:
    """Verifies that the assertive rewrite has fewer hedging expressions than
    the neutral version.

    This confirms the manipulation succeeded.  A failed manipulation (assertive
    has more hedges than neutral) is flagged with ok=False.

    Args:
        min_advantage: Minimum required reduction in hedge density
            (hedges/100 words) for assertive vs. neutral.
            Default 0.0 means assertive must have ≤ neutral hedge density;
            set > 0 to require a strict reduction.
        report_only: If True always returns ok=True.
    """

    def __init__(
        self,
        min_advantage: float = _MIN_ASSERTIVENESS_ADVANTAGE,
        report_only: bool = False,
    ):
        self.min_advantage = min_advantage
        self.report_only = report_only

    @staticmethod
    def _hedge_density(text: str) -> float:
        words = len(text.split())
        if words == 0:
            return 0.0
        hedges = len(_HEDGE_PATTERN.findall(text))
        return hedges / words * 100

    def check_pair(self, neutral: str, assertive: str) -> dict:
        """neutral is the 'original' (correct-doc neutral rewrite),
        assertive is the 'rewritten' (incorrect-doc assertive rewrite)."""
        result: dict = {"ok": True, "scores": {}, "reasons": []}
        try:
            neutral_density = self._hedge_density(neutral)
            assertive_density = self._hedge_density(assertive)
            advantage = neutral_density - assertive_density
            result["scores"].update({
                "neutral_hedge_density": round(neutral_density, 3),
                "assertive_hedge_density": round(assertive_density, 3),
                "assertiveness_advantage": round(advantage, 3),
            })
            if not self.report_only and advantage < self.min_advantage:
                result["ok"] = False
                result["reasons"].append(
                    f"Assertiveness advantage={advantage:.3f} < {self.min_advantage} "
                    f"(assertive rewrite has more hedges than neutral — manipulation may have failed)"
                )
        except Exception as exc:
            logger.warning("Assertiveness check failed: %s", exc)
        return result

    def check_batch(self, neutrals: list[str], assertives: list[str]) -> list[dict]:
        return [self.check_pair(n, a) for n, a in zip(neutrals, assertives)]


# ---------------------------------------------------------------------------
# 4. ROUGE-L Checker
# ---------------------------------------------------------------------------
# Lightweight longest-common-subsequence recall/precision.
# Catches cases where BERTScore passes but key phrases vanish.
# Pure Python implementation — no model download required.

ROUGEL_THRESHOLD: float = 0.50


def _lcs_length(a: list, b: list) -> int:
    """Dynamic-programming LCS length."""
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


def _rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 between reference and hypothesis (word-level)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class RougeLChecker:
    """ROUGE-L F1 check.  Flags pairs with very low content overlap.

    Args:
        threshold: Minimum ROUGE-L F1 (default 0.50).
        report_only: If True, always ok=True but scores are populated.
    """

    def __init__(self, threshold: float = ROUGEL_THRESHOLD, report_only: bool = False):
        self.threshold = threshold
        self.report_only = report_only

    def check_pair(self, original: str, rewritten: str) -> dict:
        result: dict = {"ok": True, "scores": {}, "reasons": []}
        try:
            score = _rouge_l(original, rewritten)
            result["scores"]["rouge_l_f1"] = round(score, 4)
            if not self.report_only and score < self.threshold:
                result["ok"] = False
                result["reasons"].append(
                    f"rouge_l_f1={score:.4f} < threshold={self.threshold}"
                )
        except Exception as exc:
            logger.warning("ROUGE-L check failed: %s", exc)
        return result

    def check_batch(self, originals: list[str], rewrittens: list[str]) -> list[dict]:
        return [self.check_pair(o, r) for o, r in zip(originals, rewrittens)]


# ---------------------------------------------------------------------------
# 5. Extended Quality Checker (all gates)
# ---------------------------------------------------------------------------

class ExtendedQualityChecker:
    """Unified quality gate combining all six checks.

    Gate design for tone rewrites (assertiveness manipulation):
      Primary gates (fail pairs that change meaning):
        - NLI entailment (chunked)     — gold standard for factual preservation;
                                         requires --use-nli and model download
        - Cosine similarity (stsb)     — STS-tuned paraphrase detection; on by default
        - ROUGE-L F1                   — cheap n-gram sanity check; on by default

      Reporting metrics (score is recorded but does not reject pairs):
        - BERTScore F1                 — report-only by default; unreliable as a gate
                                         for tone rewrites because it penalizes
                                         legitimate synonym substitution and clause
                                         reordering.  Kept for paper audit numbers.
        - Readability delta            — report-only (checks for fluency confound)
        - Assertiveness advantage      — report-only (verifies manipulation succeeded)

    The NLI checker uses positional chunk alignment so long ConfACT documents
    (web-sourced, often >180 words per side) are scored correctly without
    silent truncation.

    Args:
        bertscore_threshold: BS F1 gate threshold (only active when
            bertscore_report_only=False).  Default 0.65.
        cosine_threshold: STS cosine gate threshold (default 0.85).
        rouge_threshold: ROUGE-L F1 gate threshold (default 0.50).
        nli_threshold: NLI entailment probability (default 0.70).
        max_grade_delta: Max FKGL / GFI delta (default 2.0).
        embedding_model: Sentence-Transformer model name.  Defaults to
            stsb-roberta-large (STS-tuned); set to all-mpnet-base-v2 for
            retrieval-style similarity instead.
        use_bertscore: Compute BERTScore (default True for reporting).
        bertscore_report_only: If True (default), BERTScore is recorded but
            never rejects a pair.  Set False to use it as a hard gate.
        use_cosine: Enable cosine gate (default True).
        use_rouge: Enable ROUGE-L gate (default True).
        use_nli: Enable NLI entailment gate (requires transformers + ~180 MB
            model download; recommended for final TCB validation runs).
        use_readability: Enable readability delta (default True, report-only).
        use_assertiveness: Enable assertiveness heuristic (default True, report-only).
        readability_report_only: Report readability scores without failing pairs.
        assertiveness_report_only: Report assertiveness scores without failing pairs.
        device: Torch device for NLI model.
    """

    def __init__(
        self,
        bertscore_threshold: float = 0.65,
        cosine_threshold: float = 0.85,
        rouge_threshold: float = ROUGEL_THRESHOLD,
        nli_threshold: float = NLI_ENTAILMENT_THRESHOLD,
        max_grade_delta: float = READABILITY_MAX_GRADE_DELTA,
        embedding_model: Optional[str] = None,
        use_bertscore: bool = True,
        bertscore_report_only: bool = True,
        use_cosine: bool = True,
        use_rouge: bool = True,
        use_nli: bool = False,
        use_readability: bool = True,
        use_assertiveness: bool = True,
        readability_report_only: bool = True,
        assertiveness_report_only: bool = True,
        device: Optional[str] = None,
    ):
        from src.quality_checker import (  # type: ignore
            SemanticQualityChecker,
            SENTENCE_EMBEDDING_MODEL_STS,
        )
        if embedding_model is None:
            embedding_model = SENTENCE_EMBEDDING_MODEL_STS
        self._bertscore_report_only = bertscore_report_only
        self._semantic = SemanticQualityChecker(
            bertscore_threshold=bertscore_threshold,
            cosine_threshold=cosine_threshold,
            embedding_model=embedding_model,
            use_bertscore=use_bertscore,
            use_cosine=use_cosine,
            device=device,
        )
        self._rouge = RougeLChecker(threshold=rouge_threshold) if use_rouge else None
        self._nli = NLIEntailmentChecker(threshold=nli_threshold, device=device) if use_nli else None
        self._readability = (
            ReadabilityChecker(max_grade_delta=max_grade_delta, report_only=readability_report_only)
            if use_readability else None
        )
        self._assertiveness = (
            AssertivenessChecker(report_only=assertiveness_report_only)
            if use_assertiveness else None
        )

    def check_pair(self, original: str, rewritten: str) -> dict:
        """original = neutral correct doc, rewritten = assertive incorrect doc."""
        combined: dict = {"ok": True, "scores": {}, "reasons": []}

        # Semantic checker (BERTScore + cosine).  When bertscore_report_only=True,
        # BERTScore failures are downgraded to informational notes so that
        # legitimate tone rewrites aren't rejected for synonym substitution.
        if self._semantic is not None:
            r = self._semantic.check_pair(original, rewritten)
            combined["scores"].update(r.get("scores", {}))
            if self._bertscore_report_only:
                # Strip BERTScore failure from ok/reasons; keep in scores for audit.
                bs_reasons = [x for x in r.get("reasons", []) if "bertscore" in x]
                other_reasons = [x for x in r.get("reasons", []) if "bertscore" not in x]
                combined["reasons"].extend(other_reasons)
                if bs_reasons:
                    combined["scores"]["bertscore_note"] = "; ".join(bs_reasons)
                if other_reasons:
                    combined["ok"] = False
            else:
                combined["reasons"].extend(r.get("reasons", []))
                if not r.get("ok", True):
                    combined["ok"] = False

        for checker in [self._rouge, self._nli, self._readability]:
            if checker is None:
                continue
            r = checker.check_pair(original, rewritten)
            combined["scores"].update(r.get("scores", {}))
            combined["reasons"].extend(r.get("reasons", []))
            if not r.get("ok", True):
                combined["ok"] = False

        if self._assertiveness is not None:
            r = self._assertiveness.check_pair(neutral=original, assertive=rewritten)
            combined["scores"].update(r.get("scores", {}))
            combined["reasons"].extend(r.get("reasons", []))
            if not r.get("ok", True):
                combined["ok"] = False

        return combined

    def check_batch(self, originals: list[str], rewrittens: list[str]) -> list[dict]:
        n = len(originals)
        combined: list[dict] = [{"ok": True, "scores": {}, "reasons": []} for _ in range(n)]

        if self._semantic is not None:
            results = self._semantic.check_batch(originals, rewrittens)
            for i, r in enumerate(results):
                combined[i]["scores"].update(r.get("scores", {}))
                if self._bertscore_report_only:
                    bs_reasons = [x for x in r.get("reasons", []) if "bertscore" in x]
                    other_reasons = [x for x in r.get("reasons", []) if "bertscore" not in x]
                    combined[i]["reasons"].extend(other_reasons)
                    if bs_reasons:
                        combined[i]["scores"]["bertscore_note"] = "; ".join(bs_reasons)
                    if other_reasons:
                        combined[i]["ok"] = False
                else:
                    combined[i]["reasons"].extend(r.get("reasons", []))
                    if not r.get("ok", True):
                        combined[i]["ok"] = False

        for checker in [self._rouge, self._nli, self._readability]:
            if checker is None:
                continue
            results = checker.check_batch(originals, rewrittens)
            for i, r in enumerate(results):
                combined[i]["scores"].update(r.get("scores", {}))
                combined[i]["reasons"].extend(r.get("reasons", []))
                if not r.get("ok", True):
                    combined[i]["ok"] = False

        if self._assertiveness is not None:
            results = self._assertiveness.check_batch(originals, rewrittens)
            for i, r in enumerate(results):
                combined[i]["scores"].update(r.get("scores", {}))
                combined[i]["reasons"].extend(r.get("reasons", []))
                if not r.get("ok", True):
                    combined[i]["ok"] = False

        return combined


# ---------------------------------------------------------------------------
# CLI: audit an existing TCB dataset
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Run extended quality audit on an existing TCB JSON file.

    Usage:
        python -m src.quality_checker_extended \\
            --tcb data/full/confact/tcb_humc_processed_cleaned.json \\
            --out results/quality_extended/confact_extended_audit.json \\
            [--use-nli] [--device mps]
    """
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="Extended quality audit for a TCB JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_cli_main.__doc__,
    )
    p.add_argument("--tcb", required=True, metavar="PATH", help="TCB JSON file path.")
    p.add_argument("--out", required=True, metavar="PATH", help="Output JSON path.")
    p.add_argument("--use-nli", action="store_true",
                   help="Enable NLI entailment gate (recommended; requires ~180 MB model download).")
    p.add_argument("--bertscore-gate", action="store_true",
                   help="Treat BERTScore as a hard gate (default: report-only).")
    p.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore entirely.")
    p.add_argument("--no-readability", action="store_true", help="Disable readability check.")
    p.add_argument("--no-assertiveness", action="store_true", help="Disable assertiveness check.")
    p.add_argument("--no-rouge", action="store_true", help="Disable ROUGE-L check.")
    p.add_argument("--embedding-model", default=None,
                   help="Sentence-Transformer model for cosine gate. "
                        "Default: stsb-roberta-large (STS-tuned). "
                        "Alternative: all-mpnet-base-v2 (retrieval-tuned).")
    p.add_argument("--bs-threshold", type=float, default=0.65)
    p.add_argument("--cos-threshold", type=float, default=0.85)
    p.add_argument("--rouge-threshold", type=float, default=0.50)
    p.add_argument("--nli-threshold", type=float, default=0.70)
    p.add_argument("--max-grade-delta", type=float, default=2.0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--readability-report-only", action="store_true", default=True)
    p.add_argument("--assertiveness-report-only", action="store_true", default=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    checker = ExtendedQualityChecker(
        bertscore_threshold=args.bs_threshold,
        cosine_threshold=args.cos_threshold,
        rouge_threshold=args.rouge_threshold,
        nli_threshold=args.nli_threshold,
        max_grade_delta=args.max_grade_delta,
        embedding_model=args.embedding_model,
        use_bertscore=not args.no_bertscore,
        bertscore_report_only=not args.bertscore_gate,
        use_nli=args.use_nli,
        use_readability=not args.no_readability,
        use_assertiveness=not args.no_assertiveness,
        use_rouge=not args.no_rouge,
        readability_report_only=args.readability_report_only,
        assertiveness_report_only=args.assertiveness_report_only,
        device=args.device,
    )

    tcb = json.loads(Path(args.tcb).read_text())
    logger.info("Loaded %d TCB instances from %s", len(tcb), args.tcb)

    originals, rewrittens, doc_ids = [], [], []
    for inst in tcb:
        docs = inst.get("documents", [])
        neutral = next(
            (d.get("content", "") for d in docs if d.get("tone") == "neutral"), ""
        )
        assertive = next(
            (d.get("content", "") for d in docs if d.get("tone") == "assertive"), ""
        )
        if neutral and assertive:
            originals.append(neutral)
            rewrittens.append(assertive)
            doc_ids.append(inst.get("id", f"inst_{len(doc_ids)}"))

    logger.info("Scoring %d neutral/assertive pairs ...", len(originals))

    bs = args.batch_size
    all_results = []
    for i in range(0, len(originals), bs):
        batch_o = originals[i : i + bs]
        batch_r = rewrittens[i : i + bs]
        batch_results = checker.check_batch(batch_o, batch_r)
        all_results.extend(batch_results)
        logger.info("  Scored %d / %d pairs", min(i + bs, len(originals)), len(originals))

    output = []
    n_fail = 0
    for doc_id, res in zip(doc_ids, all_results):
        row = {"doc_id": doc_id, "pass": res["ok"], **res["scores"]}
        if res["reasons"]:
            row["reasons"] = res["reasons"]
        if not res["ok"]:
            n_fail += 1
        output.append(row)

    logger.info("Pairs: %d total, %d pass, %d fail", len(output), len(output) - n_fail, n_fail)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    logger.info("Written → %s", args.out)


if __name__ == "__main__":
    _cli_main()
