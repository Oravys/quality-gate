"""
quality-gate -- NISQA-proxy audio quality scorer.

Lightweight, dependency-light MOS estimator for speech audio.
Scores four perceptual dimensions (noisiness, coloration, discontinuity,
loudness) and returns a single Mean Opinion Score on the 1-5 scale
defined by ITU-T P.808.

Quick start
-----------
>>> from quality_gate import assess_quality
>>> result = assess_quality(audio_array, sample_rate)
>>> print(result.mos, result.verdict)
"""

from quality_gate.scorer import (
    QualityResult,
    QualityScorer,
    assess_quality,
)

__all__ = [
    "QualityResult",
    "QualityScorer",
    "assess_quality",
]

__version__ = "0.1.0"
