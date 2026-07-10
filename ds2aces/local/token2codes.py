from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class Token2CodesTool:
    def __init__(self, mlaudio_dir: str | Path):
        mlaudio_dir = Path(mlaudio_dir)
        self.codebooks = np.load(mlaudio_dir / "codebooks.npy")  # [4, 64000, 6]
        self.scales = np.load(mlaudio_dir / "scales.npy")  # [4, 6]
        self.zero_codes = np.load(mlaudio_dir / "zero_codes.npy")  # [1, 6, 64, 8]

    def process(self, tokens: NDArray[np.uint16]) -> NDArray[np.float32]:
        if tokens.ndim != 2 or tokens.shape[1] != 32:
            raise ValueError(
                f"Expected 2D input tensor with second dimension 32, got shape {tokens.shape}"
            )

        n_frames = tokens.shape[0]
        result = np.zeros((n_frames, 6, 8), dtype=np.float32)

        for g in range(4):
            indices = tokens[:, g * 8 : (g + 1) * 8]  # [N, 8]
            looked_up = self.codebooks[g][indices]  # [N, 8, 6]
            scaled = looked_up * self.scales[g]  # broadcast [N, 8, 6] * [6]
            result += scaled.transpose(0, 2, 1)  # [N, 8, 6] -> [N, 6, 8]

        return result  # [N, 6, 8]

    def process_to_blocks(
        self, tokens: NDArray[np.uint16], block_size: int = 64
    ) -> list[NDArray[np.float32]]:
        """Convert tokens to DCAE-ready [1, 6, block_size, 8] blocks.

        ACE Studio pads each block with ``zero_codes`` (the learned padding
        baseline), NOT with np.zeros.  Proxy captures confirm that DCAE
        blocks have the following layout:

        * Frames 0-19: zero_codes (leading padding)
        * Frames 20-36: 17 frames of actual token codes
        * Frames 37-63: zero_codes (trailing padding)

        The token data is placed at a fixed offset of 20 within each 64-frame
        DCAE block.

        Args:
            tokens: [N, 32] uint16 token array.
            block_size: code-frames per DCAE block (default 64).

        Returns:
            List of [1, 6, block_size, 8] float32 arrays.
        """
        codes = self.process(tokens)  # [N, 6, 8]
        n_frames = codes.shape[0]
        blocks: list[NDArray[np.float32]] = []

        # Leading padding = 20 frames (from proxy capture analysis)
        lead_pad = 20

        for start in range(0, n_frames, 17):  # each block holds at most 17 valid frames
            block_codes = codes[start: start + 17]  # at most 17 frames [N, 6, 8]
            actual_len = block_codes.shape[0]

            # Build 64-frame block
            block = self.zero_codes[0].copy()  # [6, 64, 8]

            # Place token codes starting at frame 20
            block_codes_t = block_codes.transpose(1, 0, 2)  # [6, actual_len, 8]
            end = lead_pad + actual_len
            if end > 64:
                # Truncate if exceeding 64 frames
                actual_len = 64 - lead_pad
                end = 64
                block_codes_t = block_codes_t[:, :actual_len, :]

            block[:, lead_pad:lead_pad + actual_len, :] = block_codes_t

            blocks.append(block.reshape(1, 6, block_size, 8))

        return blocks
