"""Mel2Pitch: mel spectrogram -> fundamental frequency (F0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class Mel2Pitch:
    """Wraps mel2pitch_12M_1000k_fp32_dyn.onnx.

    Input:  mel [1, 128, T] float32
    Output: f0  [1, 1, T]   float32 (normalized, multiply by 264 for Hz)

    The output is normalized pitch. To convert to Hz:
        pitch_hz = output * 264
    This scaling factor was determined from proxy capture validation:
    voiced-segment f0~1.0 corresponds to 264Hz (confirmed by pulse impulse
    spacing in the captured audio).

    ACE Studio also upsamples the model output 2x (512 -> 1024 frames) via
    linear interpolation before feeding to the vocoder.
    """

    def __init__(self, mlaudio_dir: str | Path, providers: list[str] | None = None):
        mlaudio_dir = Path(mlaudio_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(mlaudio_dir / "mel2pitch_12M_1000k_fp32_dyn.onnx"),
            providers=self.providers,
        )

    def run(self, mel: np.ndarray) -> np.ndarray:
        """Predict F0 from mel spectrogram.

        Args:
            mel: [1, 128, T] float32

        Returns:
            f0: [T] float32 (raw model output, not denormalized)
        """
        out = self.session.run(None, {"mel": mel.astype(np.float32)})
        return out[0][0, 0]  # [T]

    def infer(self, mel: np.ndarray) -> np.ndarray:
        """Predict F0 and upsample 2x to match ACE Studio behavior.

        Args:
            mel: [T, 128] float32

        Returns:
            pitch_hz: [2*T] float32 in Hz
        """
        mel_in = mel.T[np.newaxis]  # [1, 128, T]
        f0_norm = self.run(mel_in)  # [T]
        # Upsample 2x via linear interpolation
        f0_up = np.interp(
            np.arange(len(f0_norm) * 2),
            np.arange(len(f0_norm)) * 2,
            f0_norm
        )
        # Denormalize to Hz and clamp negative to 0
        pitch_hz = np.maximum(f0_up * 264.0, 0.0).astype(np.float32)
        return pitch_hz
