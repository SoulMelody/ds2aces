"""DCAE decoder: codes -> mel + f0 (first stage of inference pipeline).

DCAE denormalization (from ACE-Step source code):
    mel = (dcae_output * 0.5 + 0.5) * (max_mel - min_mel) + min_mel
    Default: min_mel=-11.0, max_mel=3.0 -> mel = dcae_out * 7.0 - 4.0

Time dimension: latent_dim = mel_dim / 8 (time_dimention_multiple=8)
Upsampling: 64 code frames -> 1024 mel frames (16x)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

# ACE-Step defaults (may differ for ACE Studio singing model)
DEFAULT_MIN_MEL = -11.0
DEFAULT_MAX_MEL = 3.0


class DCAEDecoder:
    """Wraps dcae_decoder_v4lite_1024_fp16.onnx model.

    Input:  codes  [1, 6, 64, 8] float16
    Output: pred_mel [1, 1, 1024, 128] float16
            pred_f0  [1, 1, 1024, 1]   float16
    """

    def __init__(
        self,
        mlaudio_dir: str | Path,
        providers: list[str] | None = None,
        min_mel: float = DEFAULT_MIN_MEL,
        max_mel: float = DEFAULT_MAX_MEL,
    ):
        mlaudio_dir = Path(mlaudio_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self.min_mel = min_mel
        self.max_mel = max_mel
        self.session = ort.InferenceSession(
            str(mlaudio_dir / "dcae_decoder_v4lite_1024_fp16.onnx"),
            providers=self.providers,
        )

    def denormalize_mel(self, mel: np.ndarray) -> np.ndarray:
        """Apply DCAE mel denormalization (from ACE-Step MusicDCAE.decode).

        Args:
            mel: raw DCAE output, typically in [-1, 1] range

        Returns:
            denormalized mel in log-power range
        """
        mel = mel * 0.5 + 0.5  # undo Normalize(0.5, 0.5)
        mel = mel * (self.max_mel - self.min_mel) + self.min_mel
        return mel

    def run_block(self, codes: np.ndarray, denormalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Run DCAE on a single block of codes.

        Args:
            codes: [1, 6, 64, 8] float32 or float16
            denormalize: if True, apply mel denormalization

        Returns:
            mel: [1024, 128] float32
            f0: [1024] float32
        """
        out = self.session.run(
            None, {"codes": codes.astype(np.float16)}
        )
        mel = out[0][0, 0].astype(np.float32)  # [1024, 128]
        f0 = out[1][0, 0, :, 0].astype(np.float32)  # [1024]
        if denormalize:
            mel = self.denormalize_mel(mel)
        return mel, f0
