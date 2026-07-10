"""End-to-end inference pipeline: tokens -> audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ds2aces.local.decoder import DCAEDecoder
from ds2aces.local.mel2pitch import Mel2Pitch
from ds2aces.local.token2codes import Token2CodesTool
from ds2aces.local.vocoder import Vocoder


class Pipeline:
    """Runs the full ACE Studio inference pipeline locally.

    Pipeline stages:
        1. Token2Codes: uint16 tokens -> float32 codes via RVQ lookup
        2. DCAE Decoder: codes -> mel spectrogram + f0 (CPU to avoid NaN)
        3. Mel2Pitch: mel -> refined pitch contour (GPU-capable)
        4. Vocoder: mel + pitch -> audio waveform (GPU-capable, RefineGAN)
    """

    def __init__(
        self,
        mlaudio_dir: str | Path,
        providers: list[str] | None = None,
    ):
        """Initialize inference pipeline.

        Args:
            mlaudio_dir: Path to mlaudio directory containing ONNX models and codebooks
            providers: ONNX Runtime provider list (default: CPU)
        """
        self.mlaudio_dir = Path(mlaudio_dir)
        self.providers = providers or ["CPUExecutionProvider"]

        # Initialize components (DCAE always on CPU to avoid NaN)
        self.token2codes = Token2CodesTool(mlaudio_dir)
        self.dcae = DCAEDecoder(mlaudio_dir, ["CPUExecutionProvider"])
        self.mel2pitch = Mel2Pitch(mlaudio_dir, self.providers)
        self.vocoder = Vocoder(mlaudio_dir, self.providers)

    def run_from_tokens(
        self,
        tokens: np.ndarray,
        *,
        include_audio: bool = True,
    ) -> dict[str, np.ndarray]:
        """Run full pipeline from server tokens to audio or pred_f0.

        Args:
            tokens: [N, 32] uint16 token array from server (N = number of rows)
            include_audio: when False, stop after DCAE and return pred_f0 only.

        Returns:
            dict with intermediate arrays and, when requested, audio.
        """
        # 1. Token2Codes: [N, 32] -> [N, 6, 8] codes
        codes = self.token2codes.process(tokens)

        # Convert codes to DCAE-ready blocks: list of [1, 6, 64, 8]
        blocks = self.token2codes.process_to_blocks(tokens, block_size=64)

        # 2. DCAE: codes -> mel + f0
        lead_mel = 20 * 16  # 320 mel frames padding before valid data
        valid_per_block = 17  # token frames per block
        mel_per_valid = 16   # mel frames per code frame

        all_mel, all_f0 = [], []
        n_tokens = tokens.shape[0]
        for i, block in enumerate(blocks):
            mel_block, f0_block = self.dcae.run_block(block, denormalize=True)
            n_valid = min(valid_per_block, n_tokens - i * valid_per_block)
            start = lead_mel
            end = start + n_valid * mel_per_valid
            all_mel.append(mel_block[start:end])
            all_f0.append(f0_block[start:end])

        mel = np.concatenate(all_mel, axis=0)
        pred_f0 = np.concatenate(all_f0, axis=0)

        result = {
            "codes": codes,
            "mel": mel,
            "pred_f0": pred_f0,
        }
        if not include_audio:
            return result

        # 3. Mel2Pitch: mel -> refined pitch (Hz, upsampled 2x)
        pitch = self.mel2pitch.infer(mel)

        # 4. Vocoder: mel + pitch -> audio
        audio = self._run_vocoder(mel, pitch)

        result.update({"pitch": pitch, "audio": audio})
        return result

    def _run_vocoder(self, mel: np.ndarray, pitch: np.ndarray) -> np.ndarray:
        """Run vocoder in blocks of 960 mel frames.

        Args:
            mel: [T, 128] mel spectrogram
            pitch: [T] pitch contour in Hz

        Returns:
            audio: [M] mono audio samples
        """
        n_frames = mel.shape[0]
        n_blocks = (n_frames + self.vocoder.BLOCK_MEL_FRAMES - 1) // self.vocoder.BLOCK_MEL_FRAMES
        audio_blocks = []

        for i in range(n_blocks):
            start = i * self.vocoder.BLOCK_MEL_FRAMES
            end = min(start + self.vocoder.BLOCK_MEL_FRAMES, n_frames)
            mel_block = mel[start:end]
            pitch_block = pitch[start:end]

            # Pad to 960 frames if needed
            if mel_block.shape[0] < self.vocoder.BLOCK_MEL_FRAMES:
                pad_len = self.vocoder.BLOCK_MEL_FRAMES - mel_block.shape[0]
                mel_block = np.pad(mel_block, ((0, pad_len), (0, 0)), mode="constant")
                pitch_block = np.pad(pitch_block, (0, pad_len), mode="constant")

            # Prepare inputs
            mel_input = mel_block.T[np.newaxis]  # [1, 128, 960]
            pulse_folded = self.vocoder.generate_pulse(pitch_block)
            noise = self.vocoder.generate_noise(pitch_block)

            # Run vocoder
            audio_block = self.vocoder.run_block(mel_input, pulse_folded, noise)

            # Trim to actual length
            actual_samples = (end - start) * self.vocoder.SAMPLES_PER_FRAME
            audio_blocks.append(audio_block[:actual_samples])

        return np.concatenate(audio_blocks)
