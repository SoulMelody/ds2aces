"""Utility functions for the local synthesis pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED_MLAUDIO_ASSETS = (
    "codebooks.npy",
    "scales.npy",
    "zero_codes.npy",
    "dcae_decoder_v4lite_1024_fp16.onnx",
    "mel2pitch_12M_1000k_fp32_dyn.onnx",
    "refinegan2_sota_grouped_fp16.onnx",
)


def candidate_mlaudio_dirs() -> list[Path]:
    """Return likely ACE Studio ``mlaudio`` directories."""
    candidates: list[Path] = []
    if ace_studio_path := os.environ.get("ACE_STUDIO_PATH"):
        candidates.append(Path(ace_studio_path) / "mlaudio")
    if sys.platform == "win32":
        candidates.append(Path("C:/Program Files/ACE Studio/mlaudio"))
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/ACE Studio.app/Contents/Resources/mlaudio"))
    return list(dict.fromkeys(candidates))


def missing_mlaudio_assets(mlaudio_dir: str | Path) -> list[str]:
    """Return required local synthesis assets missing from ``mlaudio_dir``."""
    root = Path(mlaudio_dir)
    return [name for name in REQUIRED_MLAUDIO_ASSETS if not (root / name).is_file()]


def resolve_mlaudio_dir(mlaudio_dir: str | Path | None = None) -> Path:
    """Resolve and validate the ACE Studio ``mlaudio`` directory.

    Returns:
        Path to a directory containing all required local synthesis assets.

    Raises:
        ValueError: if no valid directory is found.
    """
    candidates = [Path(mlaudio_dir)] if mlaudio_dir is not None else candidate_mlaudio_dirs()
    for candidate in candidates:
        if candidate.is_dir() and not missing_mlaudio_assets(candidate):
            return candidate

    candidate_text = ", ".join(str(path) for path in candidates) or "<none>"
    if mlaudio_dir is not None:
        missing = missing_mlaudio_assets(mlaudio_dir)
        raise ValueError(
            f"Invalid mlaudio directory {Path(mlaudio_dir)}; "
            f"missing: {', '.join(missing) if missing else 'directory not found'}"
        )
    raise ValueError(
        "Could not find ACE Studio mlaudio assets. Set ACE_STUDIO_PATH or pass --mlaudio-dir. "
        f"Checked: {candidate_text}"
    )


def provider_names(provider: str) -> list[str]:
    """Map a CLI provider name to ONNX Runtime provider priority."""
    provider_map = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    }
    return provider_map.get(provider, ["CPUExecutionProvider"])
