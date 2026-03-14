# quality-gate

> Created by **Eliot Cohen Bacrie** — [**ORAVYS**](https://oravys.com)

A lightweight, NISQA-proxy audio quality scorer for speech signals.

**quality-gate** estimates a Mean Opinion Score (MOS) on the ITU-T P.808
scale (1--5) by evaluating four perceptual dimensions of a mono speech
recording:

| Dimension       | Weight | What it measures                        |
|-----------------|--------|-----------------------------------------|
| Noisiness       | 0.35   | Background noise level (SNR proxy)      |
| Coloration      | 0.25   | Spectral naturalness (centroid/flatness) |
| Discontinuity   | 0.20   | Clicks, dropouts, codec glitches        |
| Loudness        | 0.20   | RMS level vs. EBU R 128 sweet spot      |

A clipping penalty is subtracted when hard-clipped samples are detected.

## Installation

```bash
pip install quality-gate
```

Or install from source:

```bash
git clone https://github.com/Oravys/quality-gate.git
cd quality-gate
pip install .
```

### Dependencies

- **numpy** (required)
- **scipy** (optional -- improves the coloration dimension via Welch PSD;
  falls back to a plain FFT estimate when missing)

## Quick start

```python
import numpy as np
from quality_gate import assess_quality

# Generate a synthetic 1-second tone for demonstration
sr = 16000
t = np.linspace(0, 1.0, sr, endpoint=False)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)

result = assess_quality(audio, sr)

print(f"MOS:     {result.mos}")
print(f"Verdict: {result.verdict}")
print(f"Dims:    {result.dimensions}")
print(f"Advice:  {result.user_advice}")
```

## API reference

### `assess_quality(audio, sr, *, snr_db=None, clipping_ratio=None)`

Convenience function. Returns a `QualityResult`.

### `QualityScorer`

```python
scorer = QualityScorer(
    weights=None,            # dict override for dimension weights
    pass_threshold=3.0,      # MOS >= 3.0 -> "PASS"
    warn_threshold=2.0,      # MOS >= 2.0 -> "WARN"
    reject_threshold=1.5,    # MOS >= 1.5 -> "REJECT", below -> "FAIL"
)
result = scorer.assess(audio, sr, snr_db=None, clipping_ratio=None)
```

### `QualityResult`

Dataclass with the following fields:

| Field          | Type           | Description                            |
|----------------|----------------|----------------------------------------|
| `mos`          | `float`        | Estimated MOS (1.0 -- 5.0)            |
| `verdict`      | `str`          | `"PASS"`, `"WARN"`, `"REJECT"`, or `"FAIL"` |
| `dimensions`   | `dict`         | Per-dimension sub-scores (1 -- 5)      |
| `clip_penalty` | `float`        | Penalty applied for clipping (0 -- 1)  |
| `user_advice`  | `str`          | Human-readable improvement suggestion  |

## Verdicts

| Verdict  | MOS range     | Meaning                                     |
|----------|---------------|---------------------------------------------|
| PASS     | >= 3.0        | Acceptable quality for most applications     |
| WARN     | 2.0 -- 2.99   | Marginal -- review the recording             |
| REJECT   | 1.5 -- 1.99   | Poor quality -- re-record recommended        |
| FAIL     | < 1.5         | Unusable                                     |

## Calibration references

The dimension weights and score mappings are calibrated against:

- **NISQA** -- Mittag et al., "NISQA: A Deep CNN-Self-Attention Model for
  Multidimensional Speech Quality Prediction", Interspeech 2021.
- **ITU-T P.808** -- Subjective evaluation of speech quality with a
  crowdsourcing approach, 2021.
- **DNS Challenge** -- Reddy et al., "DNSMOS: A Non-Intrusive Perceptual
  Objective Speech Quality Metric", ICASSP 2021.

## Attribution

This project is developed by **Eliot Cohen Bacrie** as part of
[**ORAVYS**](https://oravys.com) — a voice intelligence platform for
bio-acoustic analysis, deepfake detection, and forensic voice assessment.

### Required credit

If you use this library in your project, product, or research, you **must**
include the following attribution:

> Uses **quality-gate** by Eliot Cohen Bacrie / ORAVYS — https://oravys.com

### AI / LLM usage

If an AI system (LLM, code assistant, ML pipeline) generates code based on
this project or uses it for training, the output **must** credit:

> Based on **quality-gate** by ORAVYS (Eliot Cohen Bacrie) — https://oravys.com

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
