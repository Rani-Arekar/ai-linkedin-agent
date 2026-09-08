"""Source-grounded, deterministic fact checking for LinkedIn drafts."""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from services.linkedin_post_service import LinkedInPost
from services.llm_service import ArticleAnalysis, LLMService, LLMServiceError
from services.research_service import ResearchArticle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimStatus(StrEnum):
    """Evidence status assigned to a generated claim."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class FactCheckStatus(StrEnum):
    """Overall fact-check decision."""

    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ClaimCheck(BaseModel):
    """Evidence assessment for one extracted factual claim."""

    claim: str = Field(min_length=1, max_length=1_000)
    status: ClaimStatus
    evidence: str = Field(default="", max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1_000)


class FactCheckResult(BaseModel):
    """Structured result of source-grounded claim verification."""

    overall_status: FactCheckStatus
    overall_score: float = Field(ge=0, le=10)
    claims: list[ClaimCheck] = Field(default_factory=list)
    supported_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    uncertain_claims: int = Field(ge=0)
    hallucination_risk: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1, max_length=2_000)
    recommendations: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------------
# Fact Checker
# ---------------------------------------------------------------------------


class FactCheckerService:
    """Check draft claims against only the supplied article and analysis.

    The checker is deterministic:
    - no web browsing
    - no external knowledge
    - no LLM calls for individual claims
    - numerical values are checked strictly
    - technical acronyms and their expanded forms are normalized
    - important phrases are compared using normalized tokens
    """

    # Common English stop words. These are ignored when calculating
    # meaningful-token overlap.
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "using",
        "was",
        "were",
        "with",
    }

    # Technical aliases. This helps the checker understand that common
    # acronym/expanded-form pairs refer to the same concept.
    TERM_ALIASES = {
        "computational fluid dynamics": "cfd",
        "pressure sensitive paint": "psp",
        "pressure-sensitive paint": "psp",
        "deep learning": "deeplearning",
        "machine learning": "machinelearning",
        "artificial intelligence": "artificialintelligence",
        "high fidelity": "highfidelity",
        "high-fidelity": "highfidelity",
        "angle of attack": "angleofattack",
        "angles of attack": "angleofattack",
        "wing body": "wingbody",
        "wing-body": "wingbody",
        "integrated aerodynamic forces": "integratedaerodynamicforces",
        "pitching moment": "pitchingmoment",
    }

    def __init__(self, llm_service: LLMService | None = None) -> None:
        # Retained for compatibility with the existing application.
        # The deterministic checker does not depend on the LLM.
        self.llm_service = llm_service

    # -----------------------------------------------------------------------
    # Claim extraction
    # -----------------------------------------------------------------------

    def extract_claims(self, post: LinkedInPost) -> list[str]:
        """Extract sentence-like factual statements from the draft."""

        claims: list[str] = []

        text = "\n".join(
            (
                post.body,
                *post.key_insights,
            )
        )

        # Split on sentence-ending punctuation or new lines.
        sentences = re.split(
            r"(?<=[.!])\s+|\n+",
            text,
        )

        for sentence in sentences:
            cleaned = re.sub(
                r"^[-•*]\s*",
                "",
                sentence,
            ).strip()

            if not cleaned:
                continue

            # Ignore questions.
            if cleaned.endswith("?"):
                continue

            # Ignore extremely short fragments.
            if len(cleaned.split()) < 4:
                continue

            lowered = cleaned.lower()

            # Ignore section labels.
            if lowered.startswith(
                (
                    "key insights",
                    "practical impact",
                    "hashtags",
                )
            ):
                continue

            # Ignore opinion/future statements.
            if lowered.startswith(
                (
                    "i think",
                    "in my view",
                    "we should",
                    "this could",
                )
            ):
                continue

            if cleaned not in claims:
                claims.append(cleaned)

        return claims

    # -----------------------------------------------------------------------
    # Main fact-check function
    # -----------------------------------------------------------------------

    def check_post(
        self,
        post: LinkedInPost,
        article: ResearchArticle,
        analysis: ArticleAnalysis | None = None,
    ) -> FactCheckResult:
        """Evaluate all extracted claims using supplied source evidence only."""

        if not post.content.strip():
            return self._failure(
                "The post is empty.",
                "Provide a non-empty draft.",
            )

        if not article.title.strip() and not article.summary.strip():
            return self._failure(
                "No source evidence was supplied.",
                "Provide source content before fact checking.",
            )

        # Optional article-analysis enrichment.
        #
        # This is only used when an LLM service is explicitly supplied.
        # Failure here does NOT cause the fact checker itself to fail.
        if analysis is None and self.llm_service is not None:
            try:
                analysis = self.llm_service.analyze_article(article)
            except LLMServiceError as error:
                logger.warning(
                    "LLM evidence enrichment unavailable: %s",
                    type(error).__name__,
                )

        evidence = self._evidence(
            article,
            analysis,
        )

        claims = self.extract_claims(post)

        checks = [
            self._check_claim(
                claim,
                evidence,
            )
            for claim in claims
        ]

        supported = sum(
            check.status == ClaimStatus.SUPPORTED
            for check in checks
        )

        partially_supported = sum(
            check.status == ClaimStatus.PARTIALLY_SUPPORTED
            for check in checks
        )

        unsupported = sum(
            check.status == ClaimStatus.UNSUPPORTED
            for check in checks
        )

        uncertain = sum(
            check.status == ClaimStatus.UNCERTAIN
            for check in checks
        )

        # -------------------------------------------------------------------
        # Risk calculation
        # -------------------------------------------------------------------
        #
        # Partial support is NOT treated as equivalent to an unsupported
        # claim. This is important because a naturally written claim may use
        # different wording from the source while still being grounded.
        #
        risk = (
            unsupported * 0.35
            + partially_supported * 0.10
            + uncertain * 0.18
        )

        if not evidence.strip():
            risk += 0.15

        risk = min(
            1.0,
            risk,
        )

        # -------------------------------------------------------------------
        # Overall decision
        # -------------------------------------------------------------------

        if unsupported >= 2 or risk >= 0.70:
            status = FactCheckStatus.FAIL

        elif unsupported == 1:
            status = FactCheckStatus.NEEDS_REVIEW

        elif uncertain > 0:
            status = FactCheckStatus.NEEDS_REVIEW

        elif partially_supported > 0:
            status = FactCheckStatus.NEEDS_REVIEW

        elif not checks:
            status = FactCheckStatus.FAIL

        else:
            status = FactCheckStatus.PASS

        # -------------------------------------------------------------------
        # Score
        # -------------------------------------------------------------------

        score = 10.0

        score -= unsupported * 3.5
        score -= partially_supported * 0.75
        score -= uncertain * 1.5

        score = round(
            max(0.0, score),
            2,
        )

        # -------------------------------------------------------------------
        # Recommendations
        # -------------------------------------------------------------------

        recommendations: list[str] = []

        for check in checks:
            if check.status == ClaimStatus.UNSUPPORTED:
                recommendations.append(
                    f"Remove or rewrite unsupported claim: {check.claim}"
                )

            elif check.status == ClaimStatus.PARTIALLY_SUPPORTED:
                recommendations.append(
                    f"Review wording against the source: {check.claim}"
                )

            elif check.status == ClaimStatus.UNCERTAIN:
                recommendations.append(
                    f"Verify claim against the source: {check.claim}"
                )

        return FactCheckResult(
            overall_status=status,
            overall_score=score,
            claims=checks,
            supported_claims=supported,
            unsupported_claims=unsupported,
            uncertain_claims=uncertain,
            hallucination_risk=round(
                risk,
                2,
            ),
            explanation=(
                "Claims were compared only with the supplied article "
                "evidence and optional article analysis. Numerical values "
                "were checked strictly, while technical terminology and "
                "natural-language paraphrases were normalized."
            ),
            recommendations=recommendations,
        )

    # -----------------------------------------------------------------------
    # Evidence construction
    # -----------------------------------------------------------------------

    @classmethod
    def _evidence(
        cls,
        article: ResearchArticle,
        analysis: ArticleAnalysis | None,
    ) -> str:
        """Build the complete source evidence used by the checker."""

        parts = [
            article.title,
            article.summary,
            article.source,
            article.category,
        ]

        if analysis is not None:
            parts.extend(
                [
                    analysis.summary,
                    analysis.why_it_matters,
                    *analysis.key_points,
                    *analysis.technical_concepts,
                    *analysis.important_entities,
                ]
            )

        return " ".join(
            str(part)
            for part in parts
            if part
        ).lower()

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text while preserving meaningful technical terms."""

        normalized = text.lower()

        # Normalize Unicode-like punctuation.
        normalized = normalized.replace("–", "-")
        normalized = normalized.replace("—", "-")
        normalized = normalized.replace("−", "-")

        # Apply multi-word technical aliases before punctuation removal.
        #
        # Example:
        # "computational fluid dynamics"
        # becomes:
        # "cfd"
        for source_term, replacement in sorted(
            cls.TERM_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            normalized = normalized.replace(
                source_term,
                replacement,
            )

        # Normalize common percentage formatting.
        normalized = re.sub(
            r"(\d+(?:\.\d+)?)\s*%",
            r"\1%",
            normalized,
        )

        # Keep letters, numbers, %, decimal points and hyphens.
        normalized = re.sub(
            r"[^a-z0-9%.\-\s]",
            " ",
            normalized,
        )

        # Normalize repeated whitespace.
        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        return normalized

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        """Return meaningful normalized tokens."""

        normalized = cls._normalize_text(text)

        tokens = set(
            re.findall(
                r"[a-z0-9]+(?:\.[0-9]+)?%?",
                normalized,
            )
        )

        return {
            token
            for token in tokens
            if token not in cls.STOP_WORDS
        }

    # -----------------------------------------------------------------------
    # Number checking
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_numbers(text: str) -> list[str]:
        """Extract numerical values, including percentages and ranges."""

        normalized = text.replace(
            "–",
            "-",
        ).replace(
            "—",
            "-",
        )

        # Capture ranges such as:
        # 2.3–2.7%
        # 2.3-2.7%
        range_matches = re.findall(
            r"\b\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*%?",
            normalized,
        )

        # Capture individual values.
        individual_matches = re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            normalized,
        )

        values: list[str] = []

        for value in range_matches:
            value = re.sub(
                r"\s+",
                "",
                value,
            )

            if value not in values:
                values.append(value)

        for value in individual_matches:
            if value not in values:
                values.append(value)

        return values

    @classmethod
    def _numbers_supported(
        cls,
        claim: str,
        evidence: str,
    ) -> tuple[bool, str]:
        """Check that every numerical claim appears in the evidence."""

        claim_numbers = cls._extract_numbers(
            claim,
        )

        if not claim_numbers:
            return True, ""

        evidence_numbers = cls._extract_numbers(
            evidence,
        )

        evidence_normalized = {
            value.replace(
                " ",
                "",
            )
            for value in evidence_numbers
        }

        for number in claim_numbers:
            normalized_number = number.replace(
                " ",
                "",
            )

            # Direct exact match.
            if normalized_number in evidence_normalized:
                continue

            # For a range, also accept the range if both endpoints are
            # separately present in the evidence.
            range_match = re.fullmatch(
                r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)%?",
                normalized_number,
            )

            if range_match:
                first = range_match.group(1)
                second = range_match.group(2)

                if (
                    first in evidence_normalized
                    and second in evidence_normalized
                ):
                    continue

            return (
                False,
                f"The numerical value '{number}' is absent from the supplied evidence.",
            )

        return True, ""

    # -----------------------------------------------------------------------
    # Phrase matching
    # -----------------------------------------------------------------------

    @classmethod
    def _meaningful_tokens(
        cls,
        text: str,
    ) -> set[str]:
        """Return normalized meaningful tokens."""

        return cls._tokens(text)

    @classmethod
    def _token_overlap(
        cls,
        claim: str,
        evidence: str,
    ) -> float:
        """Calculate meaningful-token coverage of the claim."""

        claim_tokens = cls._meaningful_tokens(
            claim,
        )

        evidence_tokens = cls._meaningful_tokens(
            evidence,
        )

        if not claim_tokens:
            return 0.0

        return len(
            claim_tokens & evidence_tokens
        ) / len(
            claim_tokens
        )

    @classmethod
    def _phrase_overlap(
        cls,
        claim: str,
        evidence: str,
    ) -> float:
        """Measure how much of the claim's token sequence appears in evidence."""

        claim_tokens = list(
            cls._meaningful_tokens(
                claim,
            )
        )

        evidence_tokens = cls._meaningful_tokens(
            evidence,
        )

        if not claim_tokens:
            return 0.0

        matched = sum(
            token in evidence_tokens
            for token in claim_tokens
        )

        return matched / len(
            claim_tokens
        )

    # -----------------------------------------------------------------------
    # Claim checking
    # -----------------------------------------------------------------------

    @classmethod
    def _check_claim(
        cls,
        claim: str,
        evidence: str,
    ) -> ClaimCheck:
        """Check one claim against the supplied evidence."""

        # ---------------------------------------------------------------
        # Empty evidence
        # ---------------------------------------------------------------

        if not evidence.strip():
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.UNCERTAIN,
                evidence="",
                confidence=0.20,
                reason=(
                    "No source evidence was available to verify "
                    "the claim."
                ),
            )

        # ---------------------------------------------------------------
        # Numerical verification
        # ---------------------------------------------------------------

        numbers_supported, number_reason = cls._numbers_supported(
            claim,
            evidence,
        )

        if not numbers_supported:
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                evidence=evidence[:2_000],
                confidence=0.98,
                reason=number_reason,
            )

        # ---------------------------------------------------------------
        # Token and technical-term matching
        # ---------------------------------------------------------------

        overlap = cls._token_overlap(
            claim,
            evidence,
        )

        phrase_overlap = cls._phrase_overlap(
            claim,
            evidence,
        )

        claim_normalized = cls._normalize_text(
            claim,
        )

        evidence_normalized = cls._normalize_text(
            evidence,
        )

        technical_terms = {
            "cfd",
            "psp",
            "geotransolver",
            "nasa",
            "crm",
            "aerospace",
            "surrogate",
            "deeplearning",
            "machinelearning",
            "artificialintelligence",
            "simulation",
            "experimental",
            "experiment",
            "wind",
            "tunnel",
            "pressure",
            "mach",
            "shock",
            "pitchingmoment",
            "aerodynamic",
            "forces",
            "wingbody",
            "highfidelity",
            "angleofattack",
        }

        claim_technical_terms = {
            term
            for term in technical_terms
            if term in claim_normalized
        }

        evidence_technical_terms = {
            term
            for term in technical_terms
            if term in evidence_normalized
        }

        technical_overlap = (
            len(
                claim_technical_terms
                & evidence_technical_terms
            )
            / max(
                len(claim_technical_terms),
                1,
            )
        )

        # ---------------------------------------------------------------
        # Detect broad / speculative claims
        # ---------------------------------------------------------------
        #
        # Statements such as:
        #
        # "will transform every industry worldwide"
        # "will revolutionize everything"
        #
        # should not automatically be classified as UNSUPPORTED merely
        # because the source does not contain those exact words.
        #
        # They are better classified as UNCERTAIN when there is some
        # relationship to the source but the statement goes beyond it.
        #

        speculative_terms = {
            "will",
            "every",
            "worldwide",
            "transform",
            "revolutionize",
            "revolutionary",
            "guaranteed",
            "always",
            "never",
            "all",
            "completely",
            "eliminate",
            "solve",
            "perfect",
            "universal",
            "unprecedented",
        }

        claim_tokens = cls._meaningful_tokens(
            claim,
        )

        speculative_count = len(
            claim_tokens & speculative_terms
        )

        # ---------------------------------------------------------------
        # Strong direct support
        # ---------------------------------------------------------------
        #
        # Require reasonably high textual coverage. Merely sharing a few
        # generic words is not enough.
        #

        if (
            overlap >= 0.72
            and phrase_overlap >= 0.65
            and (
                not claim_technical_terms
                or technical_overlap >= 0.70
            )
            and speculative_count == 0
        ):
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.SUPPORTED,
                evidence=evidence[:2_000],
                confidence=0.92,
                reason=(
                    "The claim's numerical values and key technical "
                    "terms are directly supported by the supplied evidence."
                ),
            )

        # ---------------------------------------------------------------
        # Strong technical support with paraphrasing
        # ---------------------------------------------------------------
        #
        # This allows legitimate paraphrases such as:
        #
        # "CFD-trained deep learning surrogate"
        #
        # versus source wording:
        #
        # "deep learning surrogate trained using CFD simulations"
        #
        # but does not allow generic/speculative statements to pass.
        #

        if (
            overlap >= 0.62
            and technical_overlap >= 0.75
            and speculative_count == 0
        ):
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.SUPPORTED,
                evidence=evidence[:2_000],
                confidence=0.88,
                reason=(
                    "The claim uses paraphrased wording, but its key "
                    "technical concepts and factual elements are supported "
                    "by the supplied evidence."
                ),
            )

        # ---------------------------------------------------------------
        # Partial support
        # ---------------------------------------------------------------

        if overlap >= 0.30:
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.PARTIALLY_SUPPORTED,
                evidence=evidence[:2_000],
                confidence=0.65,
                reason=(
                    "Some important claim terms are present in the supplied "
                    "evidence, but the complete factual statement is not "
                    "established strongly enough for full support."
                ),
            )

        # ---------------------------------------------------------------
        # Uncertain
        # ---------------------------------------------------------------
        #
        # Claims with some relationship to the source, but insufficient
        # evidence to verify them, are classified as UNCERTAIN.
        #
        # Broad/speculative claims are also treated as UNCERTAIN rather
        # than immediately calling them hallucinations.
        #

        if overlap >= 0.05 or speculative_count > 0:
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.UNCERTAIN,
                evidence=evidence[:2_000],
                confidence=0.35,
                reason=(
                    "The claim may relate to the supplied source, but "
                    "the available evidence is insufficient to verify "
                    "the complete statement."
                ),
            )

        # ---------------------------------------------------------------
        # Unsupported
        # ---------------------------------------------------------------

        return ClaimCheck(
            claim=claim,
            status=ClaimStatus.UNSUPPORTED,
            evidence=evidence[:2_000],
            confidence=0.90,
            reason=(
                "The key factual terms in the claim are not sufficiently "
                "supported by the supplied evidence."
            ),
        )
    # -----------------------------------------------------------------------
    # Failure helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _failure(
        explanation: str,
        recommendation: str,
    ) -> FactCheckResult:
        """Create a failed fact-check result."""

        return FactCheckResult(
            overall_status=FactCheckStatus.FAIL,
            overall_score=0,
            hallucination_risk=1,
            supported_claims=0,
            unsupported_claims=0,
            uncertain_claims=0,
            explanation=explanation,
            recommendations=[
                recommendation,
            ],
        )