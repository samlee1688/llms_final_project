"""Tone-Controlled Benchmark (TCB) construction utilities.

TCB is built by taking instances with conflicting retrieved documents and
rewriting the relevant documents into neutral vs assertive registers while
preserving meaning. If Gemini is available, it is used for the rewrite step;
otherwise a deterministic local fallback is used so the pipeline still works.

Optional semantic quality checks (BERTScore F1 + cosine similarity) can be
enabled on ToneControlledBenchmarkBuilder via ``semantic_checker=``.  They
are disabled by default because they load heavy ML models; enable them for
final production runs where quality is paramount.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from src.data_loader import Document, QueryInstance, instance_to_dict, save_dataset
from src.prompts import TCB_REWRITE_SYSTEM, TCB_REWRITE_USER
from src.quality_checker import SemanticQualityChecker

logger = logging.getLogger(__name__)

_MIN_TOKEN_OVERLAP = 0.5


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _heuristic_meaning_preserved_details(original: str, rewritten: str) -> Dict[str, Any]:
    orig = _normalise_text(original)
    rew = _normalise_text(rewritten)

    if not rew:
        return {"ok": False, "reason": "empty_rewrite"}

    # Keep explicit numeric facts stable.
    orig_nums = set(re.findall(r"\d+(?:\.\d+)?", orig))
    rew_nums = set(re.findall(r"\d+(?:\.\d+)?", rew))
    if orig_nums != rew_nums:
        return {
            "ok": False,
            "reason": "numeric_facts_not_preserved",
            "original_numbers": sorted(orig_nums),
            "rewritten_numbers": sorted(rew_nums),
        }

    orig_tokens = _token_set(orig)
    rew_tokens = _token_set(rew)
    if not orig_tokens:
        return {"ok": False, "reason": "token_overlap_too_low", "token_overlap": 0.0, "min_token_overlap": _MIN_TOKEN_OVERLAP}

    overlap = len(orig_tokens & rew_tokens) / len(orig_tokens)
    if overlap < _MIN_TOKEN_OVERLAP:
        return {
            "ok": False,
            "reason": "token_overlap_too_low",
            "token_overlap": overlap,
            "min_token_overlap": _MIN_TOKEN_OVERLAP,
        }

    return {
        "ok": True,
        "reason": "ok",
        "token_overlap": overlap,
    }


def _heuristic_meaning_preserved(original: str, rewritten: str) -> bool:
    """Conservative lexical sanity check for meaning preservation.

    This does not replace semantic evaluation, but helps reject malformed
    rewrites (e.g., dropped content or contradictory numeric values).
    """
    return bool(_heuristic_meaning_preserved_details(original, rewritten).get("ok", False))


def _credibility_rank(value: str) -> int:
    order = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    return order.get((value or "unknown").lower(), 0)


def _best_correct_doc(instance: QueryInstance) -> Optional[Document]:
    docs = instance.correct_documents
    if not docs:
        return None
    return sorted(docs, key=lambda d: _credibility_rank(d.credibility), reverse=True)[0]


def _best_incorrect_doc(instance: QueryInstance) -> Optional[Document]:
    docs = instance.incorrect_documents
    if not docs:
        return None
    return sorted(docs, key=lambda d: _credibility_rank(d.credibility))[0]


def _local_rewrite(content: str, tone: str) -> str:
    content = _normalise_text(content)
    if tone == "assertive":
        return (
            "This is clear and well-established: "
            + content.replace("may", "does").replace("might", "does").replace("suggests", "shows")
        )
    # neutral
    return (
        "Available evidence suggests that "
        + content.replace("well-established", "supported")
        .replace("clearly", "likely")
        .replace("definitely", "likely")
    )


class ToneRewriter:
    """Rewrite factual text into a requested rhetorical tone.

    Supported providers:
        ``"anthropic"`` — uses ``LLMClient`` with the Anthropic API (default).
        ``"gemini"``    — uses the Google GenAI SDK directly.
        ``"local"``     — deterministic heuristic fallback, no API required.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 6,
        backoff_base_seconds: float = 1.5,
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._client = None

        if provider == "anthropic":
            self.model = model or config.ANTHROPIC_MODEL
            self.api_key = api_key or config.ANTHROPIC_API_KEY
            try:
                from src.models import LLMClient  # type: ignore
                self._client = LLMClient(
                    provider="anthropic",
                    model=self.model,
                    enable_cache=True,
                )
            except Exception as exc:
                logger.warning("Anthropic unavailable (%s); falling back to local rewrites.", exc)
        elif provider == "gemini":
            self.model = model or config.GEMINI_MODEL
            self.api_key = api_key or config.GEMINI_API_KEY
            if self.api_key:
                try:
                    from google import genai  # type: ignore
                    self._client = genai.Client(api_key=self.api_key)
                except Exception as exc:
                    logger.warning("Gemini unavailable (%s); falling back to local rewrites.", exc)
        elif provider == "local":
            self.model = model or "local"
            self.api_key = ""
        else:
            raise ValueError(
                f"Unknown rewriter provider {provider!r}. Choose from: anthropic, gemini, local"
            )

    def rewrite(self, content: str, tone: str, temperature: Optional[float] = None) -> str:
        result = self.rewrite_with_metadata(content, tone, temperature=temperature)
        return result["rewritten_content"]

    def rewrite_with_metadata(
        self, content: str, tone: str, temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Rewrite *content* into *tone*.

        Args:
            content: Source text to rewrite.
            tone: Target tone string (e.g. ``"neutral"`` or ``"assertive"``).
            temperature: Gemini sampling temperature.  ``None`` uses 0.0 (fully
                deterministic).  Pass a positive value (e.g. 0.3, 0.6) when
                retrying a failed rewrite to obtain a different output.
        """
        _temperature = 0.0 if temperature is None else float(temperature)

        if self._client is None:
            rewritten = _local_rewrite(content, tone)
            check = _heuristic_meaning_preserved_details(content, rewritten)
            return {
                "rewritten_content": rewritten,
                "meaning_preserved": bool(check.get("ok", False)),
                "meaning_failure_reason": "" if check.get("ok", False) else str(check.get("reason", "unknown")),
                "verification_source": "local_heuristic",
                "used_fallback": True,
                "meaning_check": check,
            }

        prompt = TCB_REWRITE_USER.format(content=content, tone=tone)

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "anthropic":
                    llm_response = self._client.generate(
                        system_prompt=TCB_REWRITE_SYSTEM,
                        user_prompt=prompt,
                        temperature=_temperature,
                        max_tokens=4096,
                    )
                    raw_text = llm_response.text
                else:  # gemini
                    response = self._client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "system_instruction": TCB_REWRITE_SYSTEM,
                            "temperature": _temperature,
                            "max_output_tokens": 4096,
                        },
                    )
                    raw_text = getattr(response, "text", "") or ""

                data = _extract_json(raw_text)
                rewritten = str(data.get("rewritten_content", "")).strip()
                model_claim = bool(data.get("meaning_preserved", False))

                if not rewritten:
                    fallback = _local_rewrite(content, tone)
                    check = _heuristic_meaning_preserved_details(content, fallback)
                    return {
                        "rewritten_content": fallback,
                        "meaning_preserved": bool(check.get("ok", False)),
                        "meaning_failure_reason": "" if check.get("ok", False) else str(check.get("reason", "unknown")),
                        "verification_source": "local_heuristic",
                        "used_fallback": True,
                        "meaning_check": check,
                    }

                check = _heuristic_meaning_preserved_details(content, rewritten)
                heuristic_ok = bool(check.get("ok", False))
                if model_claim and heuristic_ok:
                    reason = ""
                elif not model_claim:
                    reason = "model_claim_false"
                else:
                    reason = str(check.get("reason", "unknown"))
                return {
                    "rewritten_content": rewritten,
                    "meaning_preserved": bool(model_claim and heuristic_ok),
                    "meaning_failure_reason": reason,
                    "verification_source": "model_and_heuristic",
                    "used_fallback": False,
                    "meaning_check": check,
                }
            except Exception as exc:
                if attempt >= self.max_retries:
                    logger.debug(
                        "Rewrite failed after %d attempts for tone=%s (%s). Falling back to local rewrite.",
                        attempt, tone, exc,
                    )
                    fallback = _local_rewrite(content, tone)
                    check = _heuristic_meaning_preserved_details(content, fallback)
                    return {
                        "rewritten_content": fallback,
                        "meaning_preserved": bool(check.get("ok", False)),
                        "meaning_failure_reason": "" if check.get("ok", False) else str(check.get("reason", "unknown")),
                        "verification_source": "local_heuristic",
                        "used_fallback": True,
                        "meaning_check": check,
                    }
                delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                logger.debug(
                    "Rewrite attempt %d/%d failed for tone=%s (%s). Retrying in %.2fs.",
                    attempt, self.max_retries, tone, exc, delay,
                )
                time.sleep(delay)

        # Defensive fallback; the loop above always returns.
        fallback = _local_rewrite(content, tone)
        check = _heuristic_meaning_preserved_details(content, fallback)
        return {
            "rewritten_content": fallback,
            "meaning_preserved": bool(check.get("ok", False)),
            "meaning_failure_reason": "" if check.get("ok", False) else str(check.get("reason", "unknown")),
            "verification_source": "local_heuristic",
            "used_fallback": True,
            "meaning_check": check,
        }


@dataclass
class TCBInstanceMetadata:
    source_benchmark: str
    original_instance_id: str
    correct_doc_id: str
    incorrect_doc_id: str
    tone_target: str = "neutral_vs_assertive"
    meaning_preserved: bool = True
    rewrite_model: str = config.ANTHROPIC_MODEL
    rewrite_provider: str = "anthropic"


class ToneControlledBenchmarkBuilder:
    """Construct a TCB dataset from an existing benchmark.

    Args:
        rewriter: ToneRewriter instance (or None for default Gemini rewriter).
        semantic_checker: Optional SemanticQualityChecker.  When provided,
            each rewrite pair is additionally checked with BERTScore F1 and/or
            cosine similarity.  Pairs that fail are treated the same as
            heuristic failures — the instance is rejected.  Disabled by
            default to avoid loading heavy ML models on every run.
    """

    def __init__(
        self,
        rewriter: Optional[ToneRewriter] = None,
        semantic_checker: Optional[SemanticQualityChecker] = None,
    ):
        self.rewriter = rewriter or ToneRewriter()
        self.semantic_checker = semantic_checker
        self.last_failure_reason = ""

    def build_instance(
        self,
        instance: QueryInstance,
        source_benchmark: str = "unknown",
        temperature: Optional[float] = None,
    ) -> Optional[QueryInstance]:
        """Build a single TCB instance.

        Args:
            instance: Source QueryInstance with at least one correct and one
                incorrect document.
            source_benchmark: Label stored in output metadata.
            temperature: Gemini sampling temperature forwarded to the rewriter.
                Use ``None`` (deterministic) on the first pass; escalate
                (0.2, 0.4, …) when retrying a previously failing instance to
                obtain a different rewrite.
        """
        self.last_failure_reason = ""
        correct_doc = _best_correct_doc(instance)
        incorrect_doc = _best_incorrect_doc(instance)
        if not correct_doc or not incorrect_doc:
            self.last_failure_reason = "missing_correct_or_incorrect_doc"
            return None

        neutral = self.rewriter.rewrite_with_metadata(correct_doc.content, "neutral", temperature=temperature)
        assertive = self.rewriter.rewrite_with_metadata(incorrect_doc.content, "assertive", temperature=temperature)

        if not bool(neutral.get("meaning_preserved", False)):
            reason = str(neutral.get("meaning_failure_reason", "unknown"))
            self.last_failure_reason = f"neutral_{reason}"
            return None
        if not bool(assertive.get("meaning_preserved", False)):
            reason = str(assertive.get("meaning_failure_reason", "unknown"))
            self.last_failure_reason = f"assertive_{reason}"
            return None

        neutral_content = str(neutral.get("rewritten_content", "")).strip()
        assertive_content = str(assertive.get("rewritten_content", "")).strip()
        if not neutral_content or not assertive_content:
            self.last_failure_reason = "empty_rewritten_content"
            return None

        # ------------------------------------------------------------------
        # Optional semantic quality gate (BERTScore F1 + cosine similarity)
        # ------------------------------------------------------------------
        neutral_sem_scores: dict = {}
        assertive_sem_scores: dict = {}
        if self.semantic_checker is not None:
            neutral_check = self.semantic_checker.check_pair(
                correct_doc.content, neutral_content
            )
            assertive_check = self.semantic_checker.check_pair(
                incorrect_doc.content, assertive_content
            )
            neutral_sem_scores = neutral_check.get("scores", {})
            assertive_sem_scores = assertive_check.get("scores", {})

            if not neutral_check.get("ok", True):
                reasons = "; ".join(neutral_check.get("reasons", []))
                self.last_failure_reason = f"neutral_semantic_quality: {reasons}"
                logger.debug(
                    "Instance %s neutral rewrite failed semantic check: %s",
                    instance.id,
                    reasons,
                )
                return None
            if not assertive_check.get("ok", True):
                reasons = "; ".join(assertive_check.get("reasons", []))
                self.last_failure_reason = f"assertive_semantic_quality: {reasons}"
                logger.debug(
                    "Instance %s assertive rewrite failed semantic check: %s",
                    instance.id,
                    reasons,
                )
                return None

        docs = [
            Document(
                id=f"{instance.id}_correct_neutral",
                content=neutral_content,
                source=correct_doc.source,
                credibility=correct_doc.credibility or "high",
                supports_answer=correct_doc.supports_answer,
                tone="neutral",
                metadata={
                    **correct_doc.metadata,
                    "tone_controlled": True,
                    "original_doc_id": correct_doc.id,
                    "rewrite_meaning_preserved": True,
                    "rewrite_verification_source": str(neutral.get("verification_source", "")),
                    "rewrite_used_fallback": bool(neutral.get("used_fallback", False)),
                    **{f"semantic_{k}": v for k, v in neutral_sem_scores.items()},
                },
            ),
            Document(
                id=f"{instance.id}_incorrect_assertive",
                content=assertive_content,
                source=incorrect_doc.source,
                credibility=incorrect_doc.credibility or "low",
                supports_answer=incorrect_doc.supports_answer,
                tone="assertive",
                metadata={
                    **incorrect_doc.metadata,
                    "tone_controlled": True,
                    "original_doc_id": incorrect_doc.id,
                    "rewrite_meaning_preserved": True,
                    "rewrite_verification_source": str(assertive.get("verification_source", "")),
                    "rewrite_used_fallback": bool(assertive.get("used_fallback", False)),
                    **{f"semantic_{k}": v for k, v in assertive_sem_scores.items()},
                },
            ),
        ]

        metadata = {
            **instance.metadata,
            "constructed_from": instance.id,
            "source_benchmark": source_benchmark,
            "tone_controlled": True,
            "correct_doc_id": correct_doc.id,
            "incorrect_doc_id": incorrect_doc.id,
            "rewrite_provider": self.rewriter.provider,
            "rewrite_model": self.rewriter.model,
            "meaning_preserved": True,
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
        }
        return QueryInstance(
            id=f"tcb_{instance.id}",
            question=instance.question,
            documents=docs,
            correct_answer=instance.correct_answer,
            metadata=metadata,
        )

    def build_dataset(
        self,
        instances: List[QueryInstance],
        source_benchmark: str = "unknown",
        max_workers: int = 1,
    ) -> List[QueryInstance]:
        if max_workers <= 1:
            built: List[QueryInstance] = []
            for instance in instances:
                created = self.build_instance(instance, source_benchmark=source_benchmark)
                if created is not None:
                    built.append(created)
            return built

        # Parallel path: each worker thread gets its own builder instance to avoid
        # shared mutable state (last_failure_reason).
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _local = threading.local()

        def _build_one(inst: QueryInstance):
            if not hasattr(_local, "builder"):
                _local.builder = ToneControlledBenchmarkBuilder(
                    rewriter=ToneRewriter(
                        provider=self.rewriter.provider,
                        model=self.rewriter.model,
                        api_key=self.rewriter.api_key,
                    ),
                    semantic_checker=self.semantic_checker,
                )
            try:
                return inst.id, _local.builder.build_instance(inst, source_benchmark=source_benchmark)
            except Exception as exc:
                logger.warning("Instance %s failed: %s", inst.id, exc)
                return inst.id, None

        built_map: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_build_one, inst): inst.id for inst in instances}
            for fut in as_completed(futures):
                inst_id, result = fut.result()
                if result is not None:
                    built_map[inst_id] = result

        return [built_map[inst.id] for inst in instances if inst.id in built_map]

    def build_and_save(
        self,
        instances: List[QueryInstance],
        output_path: str,
        source_benchmark: str = "unknown",
    ) -> List[QueryInstance]:
        built = self.build_dataset(instances, source_benchmark=source_benchmark)
        save_dataset(built, output_path)
        logger.info("Saved %d TCB instances to %s", len(built), output_path)
        return built


__all__ = ["ToneRewriter", "ToneControlledBenchmarkBuilder", "TCBInstanceMetadata", "SemanticQualityChecker"]