"""Semantic quality checking for TCB rewrites.

Validates meaning preservation using rescaled BERTScore F1 (roberta-large)
and sentence-embedding cosine similarity.  Models are lazy-loaded on first
use so importing this module does not trigger any downloads.

Thresholds were calibrated from the distribution analysis on the full
RAMDocs-test TCB corpus.  See
experiments/analyze_tcb_bertscore.py for the methodology.

Known limitation: BERTScore uses a 512-token BERT window.  Documents longer
than ~350 words are silently truncated before scoring, producing artificially
low F1.  This module chunks long texts into 280-word windows and averages
chunk scores to avoid the truncation artifact.  For ConfACT (web-sourced,
often >400 words per document) this is the primary cause of low BERTScore.

Usage:
    from src.quality_checker import SemanticQualityChecker
    checker = SemanticQualityChecker()
    result = checker.check_pair(original_text, rewritten_text)
    # result = {"ok": True/False, "scores": {...}, "reasons": [...]}
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibrated thresholds
# ---------------------------------------------------------------------------
# Rescaled BERTScore F1 with roberta-large.
# Rescaling maps the expected score for random pairs to 0 and identical to 1.
# Good tone-controlled rewrites reliably land at or above 0.65.
BERTSCORE_THRESHOLD: float = 0.65
BERTSCORE_MODEL: str = "roberta-large"

# Sentence-Transformer cosine similarity.
# all-mpnet-base-v2 is a retrieval model; stsb-roberta-large is trained
# on semantic textual similarity tasks and gives better-calibrated scores for
# "same meaning, different wording" comparisons.
COSINE_SIM_THRESHOLD: float = 0.80
SENTENCE_EMBEDDING_MODEL: str = "all-mpnet-base-v2"
# Better alternative for STS (drop-in replacement, more accurate for paraphrases):
SENTENCE_EMBEDDING_MODEL_STS: str = "sentence-transformers/stsb-roberta-large"

# ---------------------------------------------------------------------------
# Long-document chunking for BERTScore
# ---------------------------------------------------------------------------
# BERT tokenizes ~1.3 tokens per word on average.  280 words ≈ 364 tokens,
# safely under the 512-token limit with room for special tokens.
_BERT_MAX_WORDS: int = 280


def _word_chunks(text: str, max_words: int = _BERT_MAX_WORDS) -> list[str]:
    """Split text into non-overlapping chunks of at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# Internal helpers (lazy-loaded)
# ---------------------------------------------------------------------------

class _BertScoreChecker:
    """Thin wrapper around the bert_score library with a persistent scorer."""

    def __init__(self, model: str = BERTSCORE_MODEL, device: Optional[str] = None):
        self.model = model
        self.device = device
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            from bert_score import BERTScorer  # type: ignore
            logger.info("Loading BERTScorer model=%s (this may take a moment)", self.model)
            # lang="en" is required when rescale_with_baseline=True (bert-score>=0.3.12).
            # verbose is not a valid BERTScorer parameter.
            self._scorer = BERTScorer(
                model_type=self.model,
                rescale_with_baseline=True,
                lang="en",
                device=self.device,
            )
        return self._scorer

    def score_pair(self, reference: str, candidate: str) -> float:
        scorer = self._load()
        ref_chunks = _word_chunks(reference)
        cand_chunks = _word_chunks(candidate)
        n = min(len(ref_chunks), len(cand_chunks))
        _, _, F1 = scorer.score(cand_chunks[:n], ref_chunks[:n])
        return float(F1.mean().item())

    def score_batch(self, references: list[str], candidates: list[str]) -> list[float]:
        scorer = self._load()
        # Short-circuit: if all pairs are short, score in one batch (fast path).
        if all(len(r.split()) <= _BERT_MAX_WORDS and len(c.split()) <= _BERT_MAX_WORDS
               for r, c in zip(references, candidates)):
            _, _, F1 = scorer.score(candidates, references)
            return [float(x.item()) for x in F1]
        # Mixed: score each pair individually with chunking.
        return [self.score_pair(r, c) for r, c in zip(references, candidates)]


class _CosineChecker:
    """Thin wrapper around sentence-transformers for cosine similarity."""

    def __init__(self, model: str = SENTENCE_EMBEDDING_MODEL, device: Optional[str] = None):
        self.model = model
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading SentenceTransformer model=%s", self.model)
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model, **kwargs)
        return self._model

    def score_pair(self, text1: str, text2: str) -> float:
        model = self._load()
        from sentence_transformers.util import cos_sim  # type: ignore
        # batch_size=2 avoids the default (no batching) overhead on MPS/GPU.
        embs = model.encode([text1, text2], batch_size=2, convert_to_tensor=True)
        return float(cos_sim(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item())

    def score_batch(self, texts1: list[str], texts2: list[str]) -> list[float]:
        model = self._load()
        from sentence_transformers.util import cos_sim  # type: ignore
        embs1 = model.encode(texts1, convert_to_tensor=True, batch_size=64)
        embs2 = model.encode(texts2, convert_to_tensor=True, batch_size=64)
        sims = cos_sim(embs1, embs2)  # (N, N) matrix
        return [float(sims[i, i].item()) for i in range(len(texts1))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SemanticQualityChecker:
    """Check meaning preservation using BERTScore F1 and cosine similarity.

    Both checks are optional (controlled by ``use_bertscore`` and
    ``use_cosine``).  Disable one or both to trade accuracy for speed.
    A pair is accepted only when *all* enabled checks pass.

    Args:
        bertscore_threshold: Minimum rescaled BERTScore F1 (default 0.65).
        cosine_threshold: Minimum cosine similarity (default 0.80).
        bertscore_model: HuggingFace model for BERTScore (default roberta-large).
        embedding_model: Sentence-Transformer model name (default all-mpnet-base-v2).
        use_bertscore: Enable BERTScore check (default True).
        use_cosine: Enable cosine similarity check (default True).
        device: Torch device string, e.g. "cuda", "cpu", or None for auto.
    """

    def __init__(
        self,
        bertscore_threshold: float = BERTSCORE_THRESHOLD,
        cosine_threshold: float = COSINE_SIM_THRESHOLD,
        bertscore_model: str = BERTSCORE_MODEL,
        embedding_model: str = SENTENCE_EMBEDDING_MODEL,
        use_bertscore: bool = True,
        use_cosine: bool = True,
        device: Optional[str] = None,
    ):
        self.bertscore_threshold = bertscore_threshold
        self.cosine_threshold = cosine_threshold
        self.use_bertscore = use_bertscore
        self.use_cosine = use_cosine

        self._bert = _BertScoreChecker(bertscore_model, device) if use_bertscore else None
        self._cosine = _CosineChecker(embedding_model, device) if use_cosine else None

    # ------------------------------------------------------------------
    # Single-pair API (used inline in ToneControlledBenchmarkBuilder)
    # ------------------------------------------------------------------

    def check_pair(self, original: str, rewritten: str) -> dict:
        """Return ``{"ok": bool, "scores": dict, "reasons": list[str]}``.

        Scores are only populated for the enabled checks.  ``ok`` is True
        only when every enabled check exceeds its threshold.
        """
        result: dict = {"ok": True, "scores": {}, "reasons": []}

        if self._bert is not None:
            try:
                f1 = self._bert.score_pair(original, rewritten)
                result["scores"]["bertscore_f1"] = round(f1, 4)
                if f1 < self.bertscore_threshold:
                    result["ok"] = False
                    result["reasons"].append(
                        f"bertscore_f1={f1:.4f} < threshold={self.bertscore_threshold}"
                    )
            except Exception as exc:
                logger.warning("BERTScore check skipped: %s", exc)

        if self._cosine is not None:
            try:
                cos = self._cosine.score_pair(original, rewritten)
                result["scores"]["cosine_similarity"] = round(cos, 4)
                if cos < self.cosine_threshold:
                    result["ok"] = False
                    result["reasons"].append(
                        f"cosine_similarity={cos:.4f} < threshold={self.cosine_threshold}"
                    )
            except Exception as exc:
                logger.warning("Cosine similarity check skipped: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Batch API (used in analysis scripts for efficiency)
    # ------------------------------------------------------------------

    def check_batch(self, originals: list[str], rewrittens: list[str]) -> list[dict]:
        """Check a batch of pairs.  Returns a list of result dicts."""
        n = len(originals)
        results: list[dict] = [{"ok": True, "scores": {}, "reasons": []} for _ in range(n)]

        if self._bert is not None:
            try:
                scores = self._bert.score_batch(originals, rewrittens)
                for i, f1 in enumerate(scores):
                    results[i]["scores"]["bertscore_f1"] = round(f1, 4)
                    if f1 < self.bertscore_threshold:
                        results[i]["ok"] = False
                        results[i]["reasons"].append(
                            f"bertscore_f1={f1:.4f} < threshold={self.bertscore_threshold}"
                        )
            except Exception as exc:
                logger.warning("Batch BERTScore check failed: %s", exc)

        if self._cosine is not None:
            try:
                scores = self._cosine.score_batch(originals, rewrittens)
                for i, cos in enumerate(scores):
                    results[i]["scores"]["cosine_similarity"] = round(cos, 4)
                    if cos < self.cosine_threshold:
                        results[i]["ok"] = False
                        results[i]["reasons"].append(
                            f"cosine_similarity={cos:.4f} < threshold={self.cosine_threshold}"
                        )
            except Exception as exc:
                logger.warning("Batch cosine similarity check failed: %s", exc)

        return results


__all__ = [
    "SemanticQualityChecker",
    "BERTSCORE_THRESHOLD",
    "COSINE_SIM_THRESHOLD",
    "BERTSCORE_MODEL",
    "SENTENCE_EMBEDDING_MODEL",
]
