"""RefineGAN vocoder: mel + pulse + noise -> audio waveform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class Vocoder:
    """Wraps refinegan2_sota_grouped_fp16.onnx.

    Input:  mel          [1, 128, 960]   float32
            pulse_folded [1, 16, 15360]  float32
            noise        [1, 16, 15360]  float32
    Output: audio_folded [1, 16, 15360]  float32

    960 mel frames -> 15360 x 16 = 245760 audio samples @ 44100 Hz.
    """

    BLOCK_MEL_FRAMES = 960
    FOLD_CHANNELS = 16
    FOLD_SAMPLES = 15360
    SAMPLES_PER_BLOCK = FOLD_CHANNELS * FOLD_SAMPLES  # 245760
    SAMPLES_PER_FRAME = SAMPLES_PER_BLOCK // BLOCK_MEL_FRAMES  # 256

    def __init__(self, mlaudio_dir: str | Path, providers: list[str] | None = None):
        mlaudio_dir = Path(mlaudio_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(mlaudio_dir / "refinegan2_sota_grouped_fp16.onnx"),
            providers=self.providers,
        )

    def run_block(
        self, mel: np.ndarray, pulse_folded: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        """Run vocoder on a single block.

        Args:
            mel:          [1, 128, 960]   float32
            pulse_folded: [1, 16, 15360]  float32
            noise:        [1, 16, 15360]  float32

        Returns:
            audio: [245760] float32 (unfolded mono samples)
        """
        out = self.session.run(
            None,
            {
                "mel": mel.astype(np.float32),
                "pulse_folded": pulse_folded.astype(np.float32),
                "noise": noise.astype(np.float32),
            },
        )
        # Unfold: audio_folded is stored as a flat time-domain vector
        # reshaped as [1, 16, 15360]. Plain flatten recovers mono audio.
        folded = out[0][0]  # [16, 15360] nominal shape
        audio = folded.flatten()[: self.SAMPLES_PER_BLOCK]  # [245760]
        return audio

    @staticmethod
    def _synthesize_excitation(
        pitch_hz: np.ndarray, sample_rate: int = 44100, spf: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize raw pulse + noise excitation from a pitch contour.

        Args:
            pitch_hz: [T] per-frame f0 in Hz (0 = unvoiced).
            sample_rate: audio sample rate.
            spf: samples per mel frame (256 -> 44100 Hz, 960 frames per block).

        Returns:
            (pulse, noise) each [T * spf] float32.
        """
        n_frames = len(pitch_hz)
        n_samples = n_frames * spf
        f0_hz = np.asarray(pitch_hz, dtype=np.float64)

        pulse = np.zeros(n_samples, dtype=np.float32)
        rng = np.random.default_rng()

        # Walk the whole block sample by sample, deciding pulse vs noise from
        # the f0 of the frame that each sample falls into.
        i = 0
        while i < n_samples:
            frame = i // spf
            f0 = f0_hz[frame]
            if f0 <= 100.0:
                # Unvoiced: fill to the end of this frame with uniform [0, 0.5).
                frame_end = min((frame + 1) * spf, n_samples)
                span = frame_end - i
                pulse[i:frame_end] = rng.random(span, dtype=np.float32) * 0.5
                i = frame_end
            else:
                # Voiced: place one 0.5 impulse, advance by one pitch period.
                pulse[i] = 0.5
                period = int(round(sample_rate / f0))
                if period < 1:
                    period = 1
                i += period

        # Noise is full-duration independent Gaussian N(0,1), unrelated to pitch.
        noise = rng.standard_normal(n_samples, dtype=np.float32)

        return pulse, noise

    @staticmethod
    def fold(signal: np.ndarray) -> np.ndarray:
        """Fold a [N] time-domain signal into [1, 16, N/16] for the vocoder.

        The vocoder output is unfolded with a plain flatten, so folding is the
        C-order reshape.
        """
        return signal.reshape(1, Vocoder.FOLD_CHANNELS, -1).astype(np.float32)

    @staticmethod
    def generate_pulse(
        pitch: np.ndarray, sample_rate: int = 44100
    ) -> np.ndarray:
        """Generate folded pulse excitation from a pitch contour."""
        pulse, _ = Vocoder._synthesize_excitation(pitch, sample_rate)
        return Vocoder.fold(pulse)

    @staticmethod
    def generate_noise(
        pitch: np.ndarray, sample_rate: int = 44100
    ) -> np.ndarray:
        """Generate the folded noise excitation (independent Gaussian)."""
        _, noise = Vocoder._synthesize_excitation(pitch, sample_rate)
        return Vocoder.fold(noise)

    @staticmethod
    def unfold_audio(folded: np.ndarray) -> np.ndarray:
        """Unfold [1, 16, N] folded audio to [1, 1, N*16] mono."""
        n_ch, n_samples = folded.shape[-2], folded.shape[-1]
        total = n_ch * n_samples
        audio = np.zeros(total, dtype=np.float32)
        for ch in range(n_ch):
            audio[ch::n_ch] = folded[ch]
        return audio[np.newaxis, np.newaxis, :]
