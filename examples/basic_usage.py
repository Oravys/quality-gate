"""
Basic usage of the quality-gate audio quality scorer.

This example demonstrates how to score synthetic audio signals.
Replace the synthetic signals with real speech loaded via your
preferred audio library (soundfile, librosa, scipy.io.wavfile, etc.).
"""

import numpy as np

from quality_gate import QualityScorer, assess_quality


def main() -> None:
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # -- 1. Clean tone at a comfortable level ----------------------------
    clean = 0.25 * np.sin(2 * np.pi * 440 * t)
    result = assess_quality(clean, sr)
    print("=== Clean tone ===")
    print(f"  MOS:     {result.mos}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Dims:    {result.dimensions}")
    print(f"  Advice:  {result.user_advice}")
    print()

    # -- 2. Noisy signal -------------------------------------------------
    noisy = clean + 0.5 * np.random.randn(len(clean))
    result = assess_quality(noisy, sr)
    print("=== Noisy signal ===")
    print(f"  MOS:     {result.mos}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Dims:    {result.dimensions}")
    print(f"  Advice:  {result.user_advice}")
    print()

    # -- 3. Clipped signal -----------------------------------------------
    clipped = np.clip(clean * 5.0, -1.0, 1.0)
    result = assess_quality(clipped, sr)
    print("=== Clipped signal ===")
    print(f"  MOS:     {result.mos}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Clip penalty: {result.clip_penalty}")
    print(f"  Advice:  {result.user_advice}")
    print()

    # -- 4. Using the class API with a known SNR -------------------------
    scorer = QualityScorer(pass_threshold=3.5)
    result = scorer.assess(clean, sr, snr_db=30.0)
    print("=== Clean tone, known SNR=30 dB, stricter threshold ===")
    print(f"  MOS:     {result.mos}")
    print(f"  Verdict: {result.verdict}")
    print()

    # -- 5. Very quiet signal --------------------------------------------
    quiet = clean * 0.001
    result = assess_quality(quiet, sr)
    print("=== Very quiet signal ===")
    print(f"  MOS:     {result.mos}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Advice:  {result.user_advice}")


if __name__ == "__main__":
    main()
