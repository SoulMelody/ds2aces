import functools
import hashlib
import json
import math
import os
import pathlib
from typing import Optional

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import typer
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding


MODEL_PATH = os.path.join(
    os.environ["ACE_STUDIO_PATH"], "ml_beat_analyze", "new_beat_this_uint8.onnx.enc"
)
KEY_SEED = "6866820113205650261"
IV_SEED = "11748920987862562174"


SAMPLE_RATE = 22050
HOP_LENGTH = 441             # 22050 / 441 = 50.0 fps
N_FFT = 1024
N_MELS = 128
FMIN = 30.0
FMAX = 11000.0
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 50.0

PAD_SIZE = 512               # reflect-pad samples before STFT
MAG_SCALE = 0.03125          # 1/32: |STFT| * MAG_SCALE -> mel filterbank
LOG_SCALE = 1000.0           # log1p(x * LOG_SCALE)

# Sliding-window ONNX inference. Matches the decompiled chunk loop:
#   chunk_start in [-CHUNK_CONTEXT .. n-CHUNK_CONTEXT), stride = CHUNK_FRAME_STRIDE
#   mel window fed to the model = ONNX_CHUNK_LEN frames starting at
#   chunk_start + CHUNK_CONTEXT.
CHUNK_FRAME_STRIDE = 1488
CHUNK_CONTEXT = 6
ONNX_CHUNK_LEN = 1500

# DFT-based tempo detection (used for the global BPM estimate).
DFT_TABLE_SIZE = 4096
BPM_MIN = 60
BPM_MAX = 240                # inclusive upper bound on the integer search

# Peak detection (running-max + local-max equality).
PEAK_RUNNING_MAX_RADIUS = 3

# Adjacent-peak clustering (frames within 1 of each other).
CLUSTER_GAP_FRAMES = 1

# Postprocessing: when snapping a downbeat to a beat, accept nearest beat if
# it is within ``mean_interval * TOLERANCE_FACTOR``; otherwise insert the
# downbeat itself as a synthetic beat.
DOWNBEAT_TOLERANCE_FACTOR = 0.25


def derive_key() -> bytes:
    return hashlib.sha256(KEY_SEED.encode()).digest()


def derive_iv() -> bytes:
    return hashlib.md5(IV_SEED.encode()).digest()


def decrypt_model(encrypted_path: str = MODEL_PATH) -> bytes:
    key = derive_key()
    iv = derive_iv()

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(padded_data) + unpadder.finalize()
    return decrypted_data


@functools.lru_cache(maxsize=4)
def load_model_from_memory(encrypted_path: str = MODEL_PATH) -> ort.InferenceSession:
    model_bytes = decrypt_model(encrypted_path)
    session = ort.InferenceSession(model_bytes)
    return session


# --- Mel-spectrogram stage --------------------------------------------------


def _compute_mel(audio: np.ndarray) -> np.ndarray:
    """Match the decompiled mel pipeline:

        reflect-pad by PAD_SIZE -> |STFT(center=False)| * MAG_SCALE ->
        mel @ mag -> log1p(x * LOG_SCALE) -> drop PAD_SIZE//HOP_LENGTH frames.
    """
    padded = np.pad(audio, PAD_SIZE, mode="reflect")

    spec = np.abs(
        librosa.stft(
            padded, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False
        )
    )
    spec = spec * MAG_SCALE

    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel = mel_fb @ spec

    compressed = np.log1p(mel * LOG_SCALE).astype(np.float32)

    start_frame = PAD_SIZE // HOP_LENGTH
    return compressed.T[start_frame:]


# --- Peak detection (matches the decompiled running-max path) --------------


def _running_max(activation: np.ndarray, radius: int = PEAK_RUNNING_MAX_RADIUS) -> np.ndarray:
    n = len(activation)
    if n == 0:
        return activation.copy()
    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n - 1, i + radius)
        result[i] = float(np.max(activation[lo:hi + 1]))
    return result


def _detect_peaks(activation: np.ndarray) -> np.ndarray:
    """Local-max peaks: strictly greater than zero and equal to the running max
    over a radius-``PEAK_RUNNING_MAX_RADIUS`` window."""
    act = np.asarray(activation, dtype=np.float32)
    if len(act) < 2:
        return np.array([], dtype=np.int64)

    local_max = _running_max(act, radius=PEAK_RUNNING_MAX_RADIUS)
    mask = (act > 0.0) & (act == local_max)
    return np.nonzero(mask)[0].astype(np.int64)


def _cluster_adjacent_peaks(frame_indices: np.ndarray) -> np.ndarray:
    """Average frames whose neighbours differ by at most ``CLUSTER_GAP_FRAMES``."""
    if len(frame_indices) < 2:
        return frame_indices.copy().astype(np.float64)

    groups: list[list[int]] = []
    current: list[int] = [int(frame_indices[0])]
    for i in range(1, len(frame_indices)):
        cur = int(frame_indices[i])
        if cur - int(frame_indices[i - 1]) <= CLUSTER_GAP_FRAMES:
            current.append(cur)
        else:
            groups.append(current)
            current = [cur]
    groups.append(current)

    return np.array([float(np.mean(g)) for g in groups], dtype=np.float64)


# --- DFT tempo estimator (matches dfttempo_awd / meas_key_tempo) -----------


@functools.lru_cache(maxsize=1)
def _dft_tables() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * math.pi, DFT_TABLE_SIZE, endpoint=False)
    return np.cos(angles), np.sin(angles)


def _dft_tempo(
    onset_envelope: np.ndarray,
    envelope_sr: float,
    bpm_min: int = BPM_MIN,
    bpm_max: int = BPM_MAX,
) -> float:
    """Match the decompiled DFT tempo detector: integer BPMs in
    ``[bpm_min, bpm_max)``, sum of |DFT(bpm)| and 0.5 * |DFT(2*bpm)| for
    fundamental + first harmonic."""
    n = len(onset_envelope)
    if n == 0:
        return 0.0

    cos_table, sin_table = _dft_tables()
    indices = np.arange(n, dtype=np.float64)

    best_bpm = 0.0
    best_energy = 0.0

    for bpm in range(bpm_min, bpm_max):
        step = DFT_TABLE_SIZE * 2.0 * bpm / (60.0 * envelope_sr)
        phases = indices * step

        idx_fund = (phases.astype(np.int64) >> 1) & (DFT_TABLE_SIZE - 1)
        idx_harm = phases.astype(np.int64) & (DFT_TABLE_SIZE - 1)

        sc_fund = float(np.dot(onset_envelope, cos_table[idx_fund]))
        ss_fund = float(np.dot(onset_envelope, sin_table[idx_fund]))
        sc_harm = float(np.dot(onset_envelope, cos_table[idx_harm]))
        ss_harm = float(np.dot(onset_envelope, sin_table[idx_harm]))

        fund = math.hypot(sc_fund, ss_fund)
        harm = math.hypot(sc_harm, ss_harm)
        total = fund + 0.5 * harm

        if total > best_energy:
            best_energy = total
            best_bpm = float(bpm)

    return best_bpm


# --- Postprocessing --------------------------------------------------------


def _postprocess(
    beat_peaks: np.ndarray,
    downbeat_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster adjacent peaks, convert to seconds, snap downbeats onto the
    nearest beat, and insert synthetic beats when a downbeat lies further than
    ``DOWNBEAT_TOLERANCE_FACTOR * mean_interval`` from any existing beat."""
    beat_frames = _cluster_adjacent_peaks(beat_peaks)
    downbeat_frames = _cluster_adjacent_peaks(downbeat_peaks)

    beat_times = beat_frames / FRAME_RATE
    downbeat_times = downbeat_frames / FRAME_RATE

    if len(beat_times) == 0 or len(downbeat_times) == 0:
        return beat_times, downbeat_times

    snapped_db = np.array(
        [float(beat_times[np.argmin(np.abs(beat_times - db))]) for db in downbeat_times],
        dtype=np.float64,
    )
    snapped_db = np.sort(snapped_db)
    snapped_db = np.array(
        [
            v for i, v in enumerate(snapped_db)
            if i == 0 or snapped_db[i] != snapped_db[i - 1]
        ],
        dtype=np.float64,
    )

    if len(beat_times) >= 2:
        mean_interval = float(np.mean(np.diff(beat_times)))
        tolerance = mean_interval * DOWNBEAT_TOLERANCE_FACTOR
    else:
        tolerance = 0.1

    for db in snapped_db:
        distances = np.abs(beat_times - db)
        nearest_dist = float(distances[np.argmin(distances)])

        if nearest_dist > tolerance:
            insert_pos = int(np.searchsorted(beat_times, db))
            beat_times = np.insert(beat_times, insert_pos, db)

    return beat_times, snapped_db


# --- Top-level entry point -------------------------------------------------


def analyze_bpm(
    audio_path: str,
    session: Optional[ort.InferenceSession] = None,
) -> dict:
    if session is None:
        session = load_model_from_memory()

    info = sf.info(audio_path)
    file_sr = info.samplerate

    audio, _ = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SAMPLE_RATE)

    audio = audio.astype(np.float32)

    # 1. Mel spectrogram at the reference 50 fps frame rate.
    mel_data = _compute_mel(audio)

    # 2. Onset envelope = max(diff(mel), 0) summed over the mel axis; this is
    #    what the decompiled tempo estimate consumes.
    onset_env = np.sum(np.maximum(np.diff(mel_data, axis=0), 0.0), axis=1)
    bpm = _dft_tempo(onset_env, FRAME_RATE)

    # 3. Sliding ONNX inference over the mel spectrogram, overlap-merged.
    total_mel_frames = mel_data.shape[0]

    chunk_starts: list[int] = []
    n = -CHUNK_CONTEXT
    while n < total_mel_frames - CHUNK_CONTEXT:
        chunk_starts.append(n)
        n += CHUNK_FRAME_STRIDE

    beat_activation = np.zeros(total_mel_frames, dtype=np.float32)
    downbeat_activation = np.zeros(total_mel_frames, dtype=np.float32)
    overlap_count = np.zeros(total_mel_frames, dtype=np.float32)

    for chunk_start in chunk_starts:
        mel_start = chunk_start + CHUNK_CONTEXT
        mel_end = mel_start + ONNX_CHUNK_LEN

        actual_start = max(mel_start, 0)
        actual_end = min(mel_end, total_mel_frames)
        available = actual_end - actual_start

        if available <= 0:
            continue

        chunk_mel = np.zeros((ONNX_CHUNK_LEN, N_MELS), dtype=np.float32)
        offset_in_chunk = actual_start - mel_start
        chunk_mel[offset_in_chunk:offset_in_chunk + available] = mel_data[actual_start:actual_end]

        mel_input = chunk_mel[np.newaxis, :, :]

        outputs = session.run(None, {"mel": mel_input})
        beat_act = np.asarray(outputs[0][0], dtype=np.float32)
        downbeat_act = np.asarray(outputs[1][0], dtype=np.float32)

        write_start = mel_start
        if write_start < 0:
            skip = -write_start
            beat_act = beat_act[skip:]
            downbeat_act = downbeat_act[skip:]
            write_start = 0

        write_end = min(mel_end, total_mel_frames)
        write_len = write_end - write_start

        if write_len <= 0 or write_len > len(beat_act):
            continue

        beat_activation[write_start:write_start + write_len] += beat_act[:write_len]
        downbeat_activation[write_start:write_start + write_len] += downbeat_act[:write_len]
        overlap_count[write_start:write_start + write_len] += 1.0

    has_overlap = overlap_count > 0
    beat_activation[has_overlap] /= overlap_count[has_overlap]
    downbeat_activation[has_overlap] /= overlap_count[has_overlap]

    # 4. Peak detection, clustering, and downbeat snapping.
    beat_peaks = _detect_peaks(beat_activation)
    downbeat_peaks = _detect_peaks(downbeat_activation)

    beat_times, downbeat_times = _postprocess(beat_peaks, downbeat_peaks)

    return {
        "bpm": bpm,
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times.tolist(),
    }


app = typer.Typer()


@app.command()
def analyze(
    audio_path: pathlib.Path = typer.Argument(..., help="音频文件路径", exists=True, dir_okay=False),
    model_path: Optional[pathlib.Path] = typer.Option(None, "--model-path", "-m", help="加密模型文件路径"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出结果"),
) -> None:
    session = load_model_from_memory(str(model_path) if model_path else MODEL_PATH)

    result = analyze_bpm(
        str(audio_path),
        session=session,
    )

    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"BPM: {result['bpm']}")


if __name__ == "__main__":
    app()
