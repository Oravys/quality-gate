"""Tests for quality_gate.scorer."""

import math

import numpy as np
import pytest

from quality_gate import QualityResult, QualityScorer, assess_quality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sr() -> int:
    return 16000


@pytest.fixture
def clean_tone(sr: int) -> np.ndarray:
    """A 1-second 440 Hz tone at a comfortable level."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return 0.25 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def scorer() -> QualityScorer:
    return QualityScorer()


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

class TestQualityResult:
    def test_fields(self) -> None:
        r = QualityResult(mos=3.5, verdict="PASS")
        assert r.mos == 3.5
        assert r.verdict == "PASS"
        assert r.dimensions == {}
        assert r.clip_penalty == 0.0
        assert r.user_advice == ""


class TestQualityScorer:
    def test_returns_quality_result(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr)
        assert isinstance(result, QualityResult)

    def test_mos_in_range(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr)
        assert 1.0 <= result.mos <= 5.0

    def test_verdict_is_valid(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr)
        assert result.verdict in {"PASS", "WARN", "REJECT", "FAIL"}

    def test_dimensions_present(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr)
        expected_keys = {"noisiness", "coloration", "discontinuity", "loudness"}
        assert set(result.dimensions.keys()) == expected_keys

    def test_all_dimension_scores_in_range(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr)
        for dim, score in result.dimensions.items():
            assert 1.0 <= score <= 5.0, f"{dim} score {score} out of range"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_audio(self, scorer: QualityScorer, sr: int) -> None:
        result = scorer.assess(np.array([]), sr)
        assert result.mos == 1.0
        assert result.verdict == "FAIL"

    def test_silence(self, scorer: QualityScorer, sr: int) -> None:
        silence = np.zeros(sr)
        result = scorer.assess(silence, sr)
        assert result.mos <= 2.0

    def test_pure_noise(self, scorer: QualityScorer, sr: int) -> None:
        np.random.seed(42)
        noise = np.random.randn(sr) * 0.3
        result = scorer.assess(noise, sr)
        # Pure noise should score lower than a clean tone
        assert result.mos < 4.5

    def test_clipped_signal(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        clipped = np.clip(clean_tone * 10.0, -1.0, 1.0)
        result = scorer.assess(clipped, sr)
        assert result.clip_penalty > 0.0

    def test_very_short_audio(self, scorer: QualityScorer, sr: int) -> None:
        short = np.array([0.1, -0.1, 0.05])
        result = scorer.assess(short, sr)
        assert isinstance(result.mos, float)


# ---------------------------------------------------------------------------
# SNR and clipping overrides
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_explicit_snr(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        high = scorer.assess(clean_tone, sr, snr_db=40.0)
        low = scorer.assess(clean_tone, sr, snr_db=5.0)
        assert high.dimensions["noisiness"] > low.dimensions["noisiness"]

    def test_explicit_clipping_ratio(
        self, scorer: QualityScorer, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = scorer.assess(clean_tone, sr, clipping_ratio=0.5)
        assert result.clip_penalty > 0.0


# ---------------------------------------------------------------------------
# Custom weights and thresholds
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_weights(self, clean_tone: np.ndarray, sr: int) -> None:
        scorer = QualityScorer(
            weights={
                "noisiness": 0.50,
                "coloration": 0.20,
                "discontinuity": 0.15,
                "loudness": 0.15,
            }
        )
        result = scorer.assess(clean_tone, sr)
        assert 1.0 <= result.mos <= 5.0

    def test_custom_thresholds(self, clean_tone: np.ndarray, sr: int) -> None:
        strict = QualityScorer(pass_threshold=4.5)
        result = strict.assess(clean_tone, sr)
        # With a very strict threshold, a simple tone may not pass
        assert result.verdict in {"PASS", "WARN", "REJECT", "FAIL"}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_assess_quality(self, clean_tone: np.ndarray, sr: int) -> None:
        result = assess_quality(clean_tone, sr)
        assert isinstance(result, QualityResult)
        assert 1.0 <= result.mos <= 5.0

    def test_assess_quality_with_kwargs(
        self, clean_tone: np.ndarray, sr: int
    ) -> None:
        result = assess_quality(clean_tone, sr, snr_db=25.0, clipping_ratio=0.0)
        assert result.clip_penalty == 0.0
