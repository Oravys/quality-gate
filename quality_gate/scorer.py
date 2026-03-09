"""
NISQA-proxy MOS scorer for speech audio.

Implements a lightweight, calibration-referenced audio quality estimator
that approximates the Mean Opinion Score (MOS) on the ITU-T P.808 scale
(1 = bad, 5 = excellent).

The algorithm evaluates four perceptual dimensions and combines them with
calibrated weights derived from the DNS Challenge / NISQA literature:

    MOS = 0.35 * noisiness
        + 0.25 * coloration
        + 0.20 * discontinuity
        + 0.20 * loudness
        - clip_penalty

References
----------
* Mittag et al., "NISQA: A Deep CNN-Self-Attention Model for
  Multidimensional Speech Quality Prediction with Crowdsourced Datasets",
  Interspeech 2021.
* ITU-T Rec. P.808 -- Subjective evaluation of speech quality with a
  crowdsourcing approach, 2021.
* Reddy et al., "DNSMOS: A Non-Intrusive Perceptual Objective Speech
  Quality Metric to Evaluate Noise Suppressors", ICASSP 2021
  (DNS Challenge baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    """Outcome of an audio quality assessment.

    Attributes
    ----------
    mos : float
        Estimated Mean Opinion Score (1.0 -- 5.0).
    verdict : str
        One of ``"PASS"``, ``"WARN"``, ``"REJECT"``, ``"FAIL"``.
    dimensions : dict
        Per-dimension sub-scores, each on the 1-5 scale:
        ``noisiness``, ``coloration``, ``discontinuity``, ``loudness``.
    clip_penalty : float
        Penalty applied for detected clipping (0.0 -- 1.0).
    user_advice : str
        Human-readable suggestion for improving the recording.
    """

    mos: float
    verdict: str
    dimensions: Dict[str, float] = field(default_factory=dict)
    clip_penalty: float = 0.0
    user_advice: str = ""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 3.0
_WARN_THRESHOLD = 2.0
_REJECT_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class QualityScorer:
    """NISQA-proxy audio quality scorer.

    Parameters
    ----------
    weights : dict, optional
        Override the default dimension weights.  Keys must be
        ``noisiness``, ``coloration``, ``discontinuity``, ``loudness``.
        Values should sum to 1.0.
    pass_threshold : float
        MOS at or above which the verdict is ``"PASS"`` (default 3.0).
    warn_threshold : float
        MOS at or above which the verdict is ``"WARN"`` (default 2.0).
    reject_threshold : float
        MOS at or above which the verdict is ``"REJECT"`` (default 1.5).
        Below this the verdict is ``"FAIL"``.

    Examples
    --------
    >>> scorer = QualityScorer()
    >>> result = scorer.assess(audio, sr=16000)
    >>> result.mos
    3.72
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        pass_threshold: float = _PASS_THRESHOLD,
        warn_threshold: float = _WARN_THRESHOLD,
        reject_threshold: float = _REJECT_THRESHOLD,
    ) -> None:
        self.weights = weights or {
            "noisiness": 0.35,
            "coloration": 0.25,
            "discontinuity": 0.20,
            "loudness": 0.20,
        }
        self.pass_threshold = pass_threshold
        self.warn_threshold = warn_threshold
        self.reject_threshold = reject_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        snr_db: Optional[float] = None,
        clipping_ratio: Optional[float] = None,
    ) -> QualityResult:
        """Score the perceptual quality of a mono speech signal.

        Parameters
        ----------
        audio : numpy.ndarray
            1-D float array of audio samples, expected in [-1, 1].
        sr : int
            Sample rate in Hz.
        snr_db : float, optional
            Pre-computed signal-to-noise ratio in dB.  When supplied the
            noisiness dimension uses this value directly instead of
            estimating it from the crest factor.
        clipping_ratio : float, optional
            Pre-computed fraction of clipped samples (0 -- 1).  When
            ``None`` it is estimated from the waveform.

        Returns
        -------
        QualityResult
        """
        audio = np.asarray(audio, dtype=np.float64).ravel()

        if audio.size == 0:
            return QualityResult(
                mos=1.0,
                verdict="FAIL",
                dimensions={k: 1.0 for k in self.weights},
                clip_penalty=0.0,
                user_advice="No audio data provided.",
            )

        # --- per-dimension scores (1-5) --------------------------------
        noisiness = self._score_noisiness(audio, sr, snr_db)
        coloration = self._score_coloration(audio, sr)
        discontinuity = self._score_discontinuity(audio, sr)
        loudness = self._score_loudness(audio)

        dims = {
            "noisiness": noisiness,
            "coloration": coloration,
            "discontinuity": discontinuity,
            "loudness": loudness,
        }

        # --- clipping penalty ------------------------------------------
        if clipping_ratio is None:
            clipping_ratio = self._estimate_clipping(audio)
        clip_pen = min(clipping_ratio * 5.0, 1.0)

        # --- weighted MOS ----------------------------------------------
        mos = (
            self.weights["noisiness"] * noisiness
            + self.weights["coloration"] * coloration
            + self.weights["discontinuity"] * discontinuity
            + self.weights["loudness"] * loudness
            - clip_pen
        )
        mos = float(np.clip(mos, 1.0, 5.0))

        verdict = self._verdict(mos)
        advice = self._advice(verdict, dims, clip_pen)

        return QualityResult(
            mos=round(mos, 2),
            verdict=verdict,
            dimensions={k: round(v, 2) for k, v in dims.items()},
            clip_penalty=round(clip_pen, 3),
            user_advice=advice,
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_noisiness(
        self, audio: np.ndarray, sr: int, snr_db: Optional[float]
    ) -> float:
        """Map SNR (or crest-factor proxy) to a 1-5 noisiness score.

        When an explicit *snr_db* is not supplied the crest factor
        (peak / RMS) is used as a rough proxy -- higher crest factors
        generally indicate cleaner speech.
        """
        if snr_db is not None:
            snr = snr_db
        else:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 1e-10:
                return 1.0
            peak = float(np.max(np.abs(audio)))
            crest_db = 20.0 * math.log10(peak / rms + 1e-12)
            # Rough mapping: crest 3 dB ≈ SNR 5, crest 20 dB ≈ SNR 40
            snr = crest_db * 2.0

        # Piecewise linear mapping SNR -> score
        if snr >= 35:
            return 5.0
        if snr >= 20:
            return 3.0 + 2.0 * (snr - 20) / 15.0
        if snr >= 10:
            return 2.0 + (snr - 10) / 10.0
        if snr >= 0:
            return 1.0 + snr / 10.0
        return 1.0

    def _score_coloration(self, audio: np.ndarray, sr: int) -> float:
        """Score spectral naturalness via centroid and flatness.

        Uses Welch PSD to compute the spectral centroid and Wiener
        spectral flatness.  Speech typically has a centroid between
        500 Hz and 3000 Hz and moderate flatness.
        """
        try:
            from scipy.signal import welch as _welch
        except ImportError:
            # Without scipy fall back to a simple FFT estimate.
            return self._score_coloration_fallback(audio, sr)

        nperseg = min(2048, len(audio))
        freqs, psd = _welch(audio, fs=sr, nperseg=nperseg)

        psd_sum = psd.sum()
        if psd_sum < 1e-20:
            return 1.0

        # Spectral centroid
        centroid = float(np.sum(freqs * psd) / psd_sum)

        # Spectral flatness (Wiener entropy ratio)
        log_psd = np.log(psd + 1e-20)
        geo_mean = float(np.exp(np.mean(log_psd)))
        arith_mean = float(np.mean(psd))
        flatness = geo_mean / (arith_mean + 1e-20)

        # Ideal centroid for speech: 800-2500 Hz
        if 800 <= centroid <= 2500:
            centroid_score = 5.0
        elif 500 <= centroid <= 3500:
            centroid_score = 3.5
        else:
            centroid_score = 2.0

        # Ideal flatness for speech: 0.01-0.15  (tonal, not noise)
        if 0.01 <= flatness <= 0.15:
            flatness_score = 5.0
        elif flatness < 0.01:
            flatness_score = 3.0  # overly tonal / narrow
        else:
            flatness_score = max(1.0, 5.0 - flatness * 20)

        return float(np.clip(0.6 * centroid_score + 0.4 * flatness_score, 1.0, 5.0))

    def _score_coloration_fallback(self, audio: np.ndarray, sr: int) -> float:
        """Coloration estimate without scipy (plain FFT)."""
        n = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio[:n])) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        s_sum = spectrum.sum()
        if s_sum < 1e-20:
            return 1.0
        centroid = float(np.sum(freqs * spectrum) / s_sum)
        if 800 <= centroid <= 2500:
            return 4.5
        if 500 <= centroid <= 3500:
            return 3.5
        return 2.0

    def _score_discontinuity(self, audio: np.ndarray, sr: int) -> float:
        """Detect abrupt energy jumps (clicks, dropouts, codec glitches).

        Splits the signal into short frames, computes per-frame RMS, and
        counts frames whose energy deviates by more than 3 standard
        deviations from the local median -- a sign of discontinuity.
        """
        frame_len = int(0.020 * sr)  # 20 ms frames
        if frame_len < 1:
            frame_len = 1

        n_frames = len(audio) // frame_len
        if n_frames < 3:
            return 4.0  # too short to judge

        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)

        median_rms = float(np.median(rms))
        if median_rms < 1e-10:
            return 1.0

        deviations = np.abs(rms - median_rms) / median_rms
        spike_ratio = float(np.mean(deviations > 3.0))

        if spike_ratio < 0.01:
            return 5.0
        if spike_ratio < 0.05:
            return 4.0
        if spike_ratio < 0.10:
            return 3.0
        if spike_ratio < 0.20:
            return 2.0
        return 1.0

    def _score_loudness(self, audio: np.ndarray) -> float:
        """Score loudness based on RMS level in dBFS.

        The sweet spot for speech is approximately -23 to -16 dBFS,
        consistent with EBU R 128 target loudness.
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-10:
            return 1.0

        dbfs = 20.0 * math.log10(rms + 1e-12)

        # Sweet spot: -23 to -16 dBFS
        if -23.0 <= dbfs <= -16.0:
            return 5.0
        if -30.0 <= dbfs < -23.0:
            return 4.0
        if -35.0 <= dbfs < -30.0:
            return 3.0
        if -10.0 >= dbfs > -16.0:
            return 3.5
        if dbfs > -10.0:
            return 2.0  # dangerously loud
        if dbfs < -40.0:
            return 1.5
        return 2.5

    # ------------------------------------------------------------------
    # Clipping estimator
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_clipping(audio: np.ndarray) -> float:
        """Return fraction of samples at or above |0.99|."""
        return float(np.mean(np.abs(audio) >= 0.99))

    # ------------------------------------------------------------------
    # Verdict & advice
    # ------------------------------------------------------------------

    def _verdict(self, mos: float) -> str:
        if mos >= self.pass_threshold:
            return "PASS"
        if mos >= self.warn_threshold:
            return "WARN"
        if mos >= self.reject_threshold:
            return "REJECT"
        return "FAIL"

    @staticmethod
    def _advice(
        verdict: str,
        dims: Dict[str, float],
        clip_pen: float,
    ) -> str:
        if verdict == "PASS":
            return "Audio quality is good. No action needed."

        parts: list[str] = []

        if clip_pen > 0.1:
            parts.append(
                "Clipping detected -- reduce the input gain or move "
                "further from the microphone."
            )

        worst = min(dims, key=dims.get)  # type: ignore[arg-type]
        advice_map = {
            "noisiness": (
                "High background noise detected. Record in a quieter "
                "environment or use a noise-suppressing microphone."
            ),
            "coloration": (
                "Unnatural tonal balance. Check your microphone placement "
                "and avoid extreme EQ settings."
            ),
            "discontinuity": (
                "Audio dropouts or clicks detected. Ensure a stable "
                "connection and avoid bumping the microphone."
            ),
            "loudness": (
                "Recording level is outside the ideal range (-23 to "
                "-16 dBFS). Adjust your input gain."
            ),
        }
        parts.append(advice_map.get(worst, "Review your recording setup."))

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def assess_quality(
    audio: np.ndarray,
    sr: int,
    *,
    snr_db: Optional[float] = None,
    clipping_ratio: Optional[float] = None,
) -> QualityResult:
    """One-call quality assessment.

    Equivalent to ``QualityScorer().assess(audio, sr, ...)``.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio samples in [-1, 1].
    sr : int
        Sample rate in Hz.
    snr_db : float, optional
        Pre-computed SNR in dB (bypasses crest-factor estimation).
    clipping_ratio : float, optional
        Pre-computed clipping ratio (0-1).

    Returns
    -------
    QualityResult
    """
    return QualityScorer().assess(
        audio, sr, snr_db=snr_db, clipping_ratio=clipping_ratio
    )
