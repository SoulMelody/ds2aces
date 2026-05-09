import functools
import hashlib
import json
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


MODEL_PATH = os.path.join(os.environ["ACE_STUDIO_PATH"], "ml_beat_analyze", "new_beat_this_uint8.onnx.enc")
KEY_SEED = "6866820113205650261"
IV_SEED = "11748920987862562174"

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
FMIN = 27.5
FMAX = 8000.0

CHUNK_DURATION = 30.0
OVERLAP_DURATION = 5.0
BEATS_PER_BAR = 4


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


def _compute_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)


def _detect_peaks(
    activation: np.ndarray,
    hop_length: int,
    sr: int,
    min_bpm: float = 40.0,
    max_bpm: float = 300.0,
) -> np.ndarray:
    min_distance_sec = 60.0 / max_bpm
    min_distance = max(1, int(min_distance_sec * sr / hop_length))

    threshold = np.mean(activation) + 0.3 * np.std(activation)

    peaks = []
    for i in range(1, len(activation) - 1):
        if activation[i] <= threshold:
            continue
        if activation[i] <= activation[i - 1] or activation[i] <= activation[i + 1]:
            continue
        if peaks and i - peaks[-1] < min_distance:
            if activation[i] > activation[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)

    peak_positions = []
    for peak in peaks:
        left = activation[peak - 1]
        center = activation[peak]
        right = activation[peak + 1]
        denominator = left - 2.0 * center + right
        if denominator < 0:
            offset = 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -0.5, 0.5))
        else:
            offset = 0.0
        peak_positions.append(peak + offset)

    return np.array(peak_positions, dtype=np.float64) * hop_length / sr


def _estimate_bpm_from_activation(
    activation: np.ndarray,
    hop_length: int,
    sr: int,
    min_bpm: float,
    max_bpm: float,
    candidate_bpms: Optional[np.ndarray] = None,
    preferred_min_bpm: float = 80.0,
    preferred_max_bpm: float = 180.0,
) -> float:
    x = np.asarray(activation, dtype=np.float64)
    if len(x) < 4:
        return 0.0

    x = np.nan_to_num(x, copy=False)
    x = x - np.percentile(x, 70)
    x[x < 0] = 0.0
    peak = np.max(x)
    if peak <= 0:
        return 0.0
    x /= peak

    frame_positions = np.arange(len(x), dtype=np.float64)

    def score_bpm(bpm: float, phase_step: float) -> float:
        if bpm <= 0:
            return 0.0

        period = sr * 60.0 / (hop_length * bpm)
        if period <= 0:
            return 0.0

        best_score = 0.0
        for phase in np.arange(0.0, period, phase_step):
            points = np.arange(phase, len(x), period)
            if len(points) < 8:
                continue
            values = np.interp(points, frame_positions, x)
            values = np.sort(values)[int(len(values) * 0.3) :]
            score = float(np.mean(values))
            if score > best_score:
                best_score = score
        return best_score

    search_bpms: list[float] = []
    if candidate_bpms is not None and len(candidate_bpms) > 0:
        for candidate in candidate_bpms:
            aliases = [candidate]
            if candidate < preferred_min_bpm:
                aliases.append(candidate * 2.0)
            if candidate > preferred_max_bpm:
                aliases.append(candidate / 2.0)
            for alias in aliases:
                if min_bpm <= alias <= max_bpm:
                    search_bpms.extend(np.arange(max(min_bpm, alias - 6.0), min(max_bpm, alias + 6.0) + 0.001, 0.5))
    else:
        search_bpms.extend(np.arange(min_bpm, max_bpm + 0.001, 1.0))

    coarse: list[tuple[float, float]] = []
    for bpm in sorted(set(round(float(bpm), 4) for bpm in search_bpms)):
        coarse.append((score_bpm(float(bpm), phase_step=1.0), float(bpm)))
    if len(coarse) == 0:
        return 0.0

    coarse.sort(reverse=True)
    refined: list[tuple[float, float]] = []
    for _, coarse_bpm in coarse[:8]:
        fine_min = max(min_bpm, coarse_bpm - 2.0)
        fine_max = min(max_bpm, coarse_bpm + 2.0)
        for bpm in np.arange(fine_min, fine_max + 0.001, 0.05):
            refined.append((score_bpm(float(bpm), phase_step=0.5), float(bpm)))
    if len(refined) == 0:
        return 0.0

    score, bpm = max(refined)
    while bpm < preferred_min_bpm and bpm * 2.0 <= max_bpm:
        doubled_score = score_bpm(bpm * 2.0, phase_step=0.5)
        if doubled_score < score * 0.75:
            break
        bpm *= 2.0
        score = doubled_score

    while bpm > preferred_max_bpm and bpm / 2.0 >= min_bpm:
        halved_score = score_bpm(bpm / 2.0, phase_step=0.5)
        if halved_score < score * 0.75:
            break
        bpm /= 2.0
        score = halved_score

    return float(bpm)


def _prepare_activation_context(activation: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    x = np.asarray(activation, dtype=np.float64)
    if len(x) < 4:
        return None

    x = np.nan_to_num(x, copy=False)
    x = x - np.percentile(x, 70)
    x[x < 0] = 0.0
    peak = np.max(x)
    if peak <= 0:
        return None
    x /= peak

    frame_positions = np.arange(len(x), dtype=np.float64)
    return x, frame_positions


def _score_activation_context(
    context: Optional[tuple[np.ndarray, np.ndarray]],
    bpm: float,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
    phase_step: float = 0.5,
    score_cache: Optional[dict[tuple[float, float], float]] = None,
) -> float:
    if context is None or bpm <= 0:
        return 0.0

    cache_key = (round(float(bpm), 6), phase_step)
    if score_cache is not None and cache_key in score_cache:
        return score_cache[cache_key]

    x, frame_positions = context
    period = sr * 60.0 / (hop_length * bpm)
    if period <= 0:
        return 0.0

    best_score = 0.0
    trim = 0
    for phase in np.arange(0.0, period, phase_step):
        points = np.arange(phase, len(x), period)
        if len(points) < 8:
            continue
        values = np.interp(points, frame_positions, x)
        trim = int(len(values) * 0.3)
        if trim > 0:
            values = np.partition(values, trim)[trim:]
        score = float(np.mean(values))
        if score > best_score:
            best_score = score

    if score_cache is not None:
        score_cache[cache_key] = best_score
    return best_score


def _score_bpm_on_activation(
    activation: np.ndarray,
    bpm: float,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
    phase_step: float = 0.5,
    context: Optional[tuple[np.ndarray, np.ndarray]] = None,
    score_cache: Optional[dict[tuple[float, float], float]] = None,
) -> float:
    if context is None:
        context = _prepare_activation_context(activation)
    return _score_activation_context(
        context,
        bpm,
        hop_length=hop_length,
        sr=sr,
        phase_step=phase_step,
        score_cache=score_cache,
    )


def _score_bpm_on_grid(
    beat_times: np.ndarray,
    bpm: float,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> float:
    if len(beat_times) < 2 or bpm <= 0:
        return 0.0

    interval = 60.0 / bpm
    if interval <= 0:
        return 0.0

    tolerance = max(interval * 0.18, hop_length / sr * 1.5)
    phase_candidates = np.linspace(0.0, interval, 64, endpoint=False)
    best_score = 0.0
    total_slots = max(1, int(np.rint((beat_times[-1] - beat_times[0]) / interval)) + 1)

    for phase in phase_candidates:
        indices = np.rint((beat_times - phase) / interval)
        grid = phase + indices * interval
        errors = np.abs(beat_times - grid)
        hit_rate = float(np.mean(errors <= tolerance))
        occupancy = len(np.unique(indices)) / total_slots
        mean_error = float(np.mean(np.minimum(errors, tolerance)))
        score = hit_rate * 0.7 + occupancy * 0.2 - (mean_error / interval) * 0.4
        if score > best_score:
            best_score = score

    return best_score


def _select_supported_bpm(
    candidates: list[tuple[float, float]],
    beat_times: np.ndarray,
    beat_activation: np.ndarray,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
    preferred_min_bpm: float = 60.0,
    preferred_max_bpm: float = 190.0,
    activation_context: Optional[tuple[np.ndarray, np.ndarray]] = None,
    activation_score_cache: Optional[dict[tuple[float, float], float]] = None,
) -> float:
    if len(candidates) == 0:
        return 0.0

    normalized: list[tuple[float, float]] = []
    for bpm, weight in candidates:
        mapped = float(bpm)
        while mapped < preferred_min_bpm and mapped * 2.0 <= preferred_max_bpm:
            mapped *= 2.0
        while mapped > preferred_max_bpm and mapped / 2.0 >= preferred_min_bpm:
            mapped /= 2.0
        normalized.append((mapped, float(weight)))

    if len(normalized) == 0:
        normalized = [(float(bpm), float(weight)) for bpm, weight in candidates]

    global_grid_score_cache: dict[float, float] = {}

    def base_score(bpm: float) -> float:
        bpm_key = round(float(bpm), 6)
        activation_score = _score_bpm_on_activation(
            beat_activation,
            float(bpm),
            hop_length=hop_length,
            sr=sr,
            context=activation_context,
            score_cache=activation_score_cache,
        )
        grid_score = global_grid_score_cache.get(bpm_key)
        if grid_score is None:
            grid_score = _score_bpm_on_grid(beat_times, float(bpm), hop_length=hop_length, sr=sr)
            global_grid_score_cache[bpm_key] = grid_score
        return grid_score * 0.6 + activation_score * 0.4

    def combined_score(bpm: float, support: float) -> float:
        return base_score(bpm) + support * 0.01

    normalized.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = []
    for bpm, weight in normalized:
        if len(clusters) == 0 or abs(bpm - clusters[-1][-1][0]) > 4.0:
            clusters.append([(bpm, weight)])
        else:
            clusters[-1].append((bpm, weight))

    cluster_centers: list[tuple[float, float]] = []
    for cluster in clusters:
        bpms = np.array([bpm for bpm, _ in cluster], dtype=np.float64)
        weights = np.array([weight for _, weight in cluster], dtype=np.float64)
        center = float(np.average(bpms, weights=weights))
        support = float(np.sum(weights))
        cluster_centers.append((center, support))

    ranked: list[tuple[float, float, list[tuple[float, float]]]] = []
    for cluster, (center, support) in zip(clusters, cluster_centers):
        cluster_best_bpm = 0.0
        cluster_best_score = -np.inf
        coarse_min = max(40.0, center - 4.0)
        coarse_max = min(300.0, center + 4.0)
        coarse_scores: list[tuple[float, float]] = []
        for bpm in np.arange(coarse_min, coarse_max + 0.001, 0.25):
            score = combined_score(float(bpm), support)
            coarse_scores.append((score, float(bpm)))
            if score > cluster_best_score:
                cluster_best_score = score
                cluster_best_bpm = float(bpm)

        coarse_scores.sort(reverse=True)
        refined_windows: list[tuple[float, float]] = []
        for _, coarse_bpm in coarse_scores[:3]:
            refined_windows.append((
                max(40.0, coarse_bpm - 1.0),
                min(300.0, coarse_bpm + 1.0),
            ))
        for fine_min, fine_max in refined_windows:
            for bpm in np.arange(fine_min, fine_max + 0.001, 0.05):
                score = combined_score(float(bpm), support)
                if score > cluster_best_score:
                    cluster_best_score = score
                    cluster_best_bpm = float(bpm)

        if cluster_best_bpm > 0:
            ranked.append((cluster_best_score, cluster_best_bpm, cluster))

    if len(ranked) == 0:
        return 0.0

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_bpm, best_cluster = ranked[0]
    for score, bpm, cluster in ranked[1:]:
        in_preferred = preferred_min_bpm <= bpm <= preferred_max_bpm
        best_in_preferred = preferred_min_bpm <= best_bpm <= preferred_max_bpm
        if best_in_preferred and not in_preferred:
            continue
        if in_preferred and not best_in_preferred:
            best_score = score
            best_bpm = bpm
            best_cluster = cluster
            continue
        if best_score - score <= 0.01 and bpm < best_bpm:
            best_score = score
            best_bpm = bpm
            best_cluster = cluster

    # Snap ambiguous continuous results back onto a discrete supported tempo in the same family.
    snap_candidates: list[tuple[float, float]] = []
    for bpm, weight in best_cluster:
        if 40.0 <= bpm <= 300.0:
            snap_candidates.append((float(bpm), float(weight)))

    if len(snap_candidates) > 0:
        scored_snaps: list[tuple[float, float]] = []
        cluster_support = float(sum(weight for _, weight in best_cluster))
        for bpm, weight in snap_candidates:
            snap_score = combined_score(bpm, cluster_support) + weight * 0.01
            scored_snaps.append((snap_score, bpm))

        scored_snaps.sort(reverse=True)
        snap_score, snap_bpm = scored_snaps[0]
        for candidate_score, candidate_bpm in scored_snaps[1:]:
            in_preferred = preferred_min_bpm <= candidate_bpm <= preferred_max_bpm
            snap_in_preferred = preferred_min_bpm <= snap_bpm <= preferred_max_bpm
            if snap_in_preferred and not in_preferred:
                continue
            if in_preferred and not snap_in_preferred:
                snap_score = candidate_score
                snap_bpm = candidate_bpm
                continue
            if snap_score - candidate_score <= 0.005 and candidate_bpm < snap_bpm:
                snap_score = candidate_score
                snap_bpm = candidate_bpm

        if best_score - snap_score <= 0.01:
            best_bpm = snap_bpm

    supported_base_score = base_score(best_bpm)

    # Conservative global fallback for cases where interval families miss the true tempo.
    broad_scores: list[tuple[float, float]] = []
    for bpm in np.arange(preferred_min_bpm, preferred_max_bpm + 0.001, 1.0):
        broad_scores.append((base_score(float(bpm)), float(bpm)))
    broad_scores.sort(reverse=True)

    broad_best_score = -np.inf
    broad_best_bpm = 0.0
    for _, coarse_bpm in broad_scores[:3]:
        for bpm in np.arange(max(preferred_min_bpm, coarse_bpm - 1.0), min(preferred_max_bpm, coarse_bpm + 1.0) + 0.001, 0.1):
            score = base_score(float(bpm))
            if score > broad_best_score:
                broad_best_score = score
                broad_best_bpm = float(bpm)

    if (
        broad_best_bpm > 0
        and broad_best_score >= supported_base_score + 0.015
        and abs(broad_best_bpm - best_bpm) >= 6.0
    ):
        return broad_best_bpm

    return best_bpm


def _regularize_beat_times(
    beat_times: np.ndarray,
    beat_activation: np.ndarray,
    bpm: float,
    total_duration: float,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    if len(beat_times) == 0 or bpm <= 0 or total_duration <= 0:
        return beat_times

    interval = 60.0 / bpm
    if interval <= 0:
        return beat_times

    activation = np.asarray(beat_activation, dtype=np.float64)
    activation = np.nan_to_num(activation, copy=False)
    if len(activation) == 0:
        return beat_times

    frame_times = np.arange(len(activation), dtype=np.float64) * hop_length / sr
    if len(frame_times) == 0:
        return beat_times

    activation = activation - np.percentile(activation, 60)
    activation[activation < 0] = 0.0
    if np.max(activation) > 0:
        activation /= np.max(activation)

    def activation_at(times: np.ndarray) -> np.ndarray:
        clipped = np.clip(times, 0.0, frame_times[-1])
        return np.interp(clipped, frame_times, activation)

    phase_candidates = np.linspace(0.0, interval, max(24, int(np.ceil(interval * 80.0))), endpoint=False)
    best_phase = 0.0
    best_score = -1.0
    for phase in phase_candidates:
        grid = np.arange(phase, total_duration + interval, interval)
        if len(grid) == 0:
            continue
        score = float(np.mean(np.sort(activation_at(grid))[int(len(grid) * 0.2) :]))
        if score > best_score:
            best_score = score
            best_phase = float(phase)

    beat_indices = np.rint((beat_times - best_phase) / interval).astype(np.int64)
    if len(beat_indices) == 0:
        return beat_times

    first_index = int(np.min(beat_indices)) - 1
    last_index = int(np.max(beat_indices)) + 1
    candidate_indices = np.arange(first_index, last_index + 1, dtype=np.int64)

    tolerance = max(interval * 0.22, hop_length / sr * 1.5)
    regularized = []

    for idx in candidate_indices:
        target_time = best_phase + idx * interval
        if target_time < 0.0 or target_time > total_duration:
            continue

        distances = np.abs(beat_times - target_time)
        nearby = np.flatnonzero(distances <= tolerance)
        if len(nearby) > 0:
            nearby_times = beat_times[nearby]
            weights = activation_at(nearby_times)
            if np.sum(weights) > 0:
                snapped = float(np.average(nearby_times, weights=weights))
            else:
                snapped = float(np.median(nearby_times))
            regularized.append(snapped)
        else:
            regularized.append(float(target_time))

    if len(regularized) == 0:
        return beat_times

    regularized_times = np.array(regularized, dtype=np.float64)
    regularized_times = np.unique(np.round(regularized_times, 6))
    return regularized_times


def _extract_interval_bpm_candidates(
    times: np.ndarray,
    multiplier: float,
    min_bpm: float,
    max_bpm: float,
    base_weight: float,
    min_interval_step: float,
) -> list[tuple[float, float]]:
    if len(times) < 2:
        return []

    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) == 0:
        return []

    median_interval = float(np.median(intervals))
    interval_step = max(min_interval_step, median_interval * 0.1)

    lo = max(np.min(intervals), 60.0 * multiplier / max_bpm)
    hi = min(np.max(intervals), 60.0 * multiplier / min_bpm)
    valid = intervals[(intervals >= lo) & (intervals <= hi)]
    if len(valid) == 0 or hi <= lo:
        return []

    bins = np.arange(lo, hi + interval_step, interval_step)
    if len(bins) < 3:
        bins = np.array([lo, lo + interval_step, hi + interval_step], dtype=np.float64)
    hist, edges = np.histogram(valid, bins=bins)
    if not np.any(hist):
        return []

    peak_indices = []
    for idx in range(len(hist)):
        left = hist[idx - 1] if idx > 0 else -1
        right = hist[idx + 1] if idx + 1 < len(hist) else -1
        if hist[idx] > 0 and hist[idx] >= left and hist[idx] >= right:
            peak_indices.append(idx)
    if len(peak_indices) == 0:
        peak_indices = [int(np.argmax(hist))]

    peak_indices = sorted(peak_indices, key=lambda idx: (hist[idx], -idx), reverse=True)[:6]
    candidates: list[tuple[float, float]] = []
    for idx in peak_indices:
        dominant_interval = 0.5 * (edges[idx] + edges[idx + 1])
        tolerance = max(interval_step, dominant_interval * 0.12)
        cluster = valid[np.abs(valid - dominant_interval) <= tolerance]
        if len(cluster) == 0:
            continue

        dominant_interval = float(np.median(cluster))
        bpm = 60.0 * multiplier / dominant_interval
        if not np.isfinite(bpm) or bpm < min_bpm or bpm > max_bpm:
            continue

        coverage = len(cluster) / len(valid)
        prominence = hist[idx] / np.max(hist)
        candidates.append((float(bpm), base_weight * (1.0 + coverage + prominence)))

    return candidates


def _calculate_bpm(
    beat_times: np.ndarray,
    downbeat_times: Optional[np.ndarray] = None,
    beat_activation: Optional[np.ndarray] = None,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
    min_bpm: float = 40.0,
    max_bpm: float = 300.0,
    beats_per_bar: int = 4,
    activation_context: Optional[tuple[np.ndarray, np.ndarray]] = None,
    activation_score_cache: Optional[dict[tuple[float, float], float]] = None,
) -> float:
    candidates: list[tuple[float, float]] = []

    min_interval_step = hop_length / sr
    candidates.extend(
        _extract_interval_bpm_candidates(
            beat_times,
            multiplier=1.0,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            base_weight=1.0,
            min_interval_step=min_interval_step,
        )
    )
    if downbeat_times is not None and beats_per_bar > 0:
        candidates.extend(
            _extract_interval_bpm_candidates(
                downbeat_times,
                multiplier=float(beats_per_bar),
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                base_weight=1.5,
                min_interval_step=min_interval_step,
            )
    )
    if beat_activation is not None:
        if activation_context is None:
            activation_context = _prepare_activation_context(beat_activation)
        supported_bpm = _select_supported_bpm(
            candidates,
            beat_times=beat_times,
            beat_activation=beat_activation,
            hop_length=hop_length,
            sr=sr,
            activation_context=activation_context,
            activation_score_cache=activation_score_cache,
        )
        if supported_bpm > 0:
            return supported_bpm

    if len(candidates) == 0:
        return 0.0

    bpms = np.array([bpm for bpm, _ in candidates], dtype=np.float64)
    weights = np.array([weight for _, weight in candidates], dtype=np.float64)

    # Collapse octave-equivalent estimates, then keep the strongest supported tempo.
    for i, bpm in enumerate(bpms):
        aliases = [bpm]
        if bpm * 2.0 <= max_bpm:
            aliases.append(bpm * 2.0)
        if bpm / 2.0 >= min_bpm:
            aliases.append(bpm / 2.0)

        best_alias = min(aliases, key=lambda alias: np.average(np.abs(bpms - alias), weights=weights))
        bpms[i] = best_alias

    bins = np.arange(min_bpm, max_bpm + 1.0, 1.0)
    hist, edges = np.histogram(bpms, bins=bins, weights=weights)
    peak_idx = int(np.argmax(hist))
    dominant_bpm = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    cluster = np.abs(bpms - dominant_bpm) <= max(1.5, dominant_bpm * 0.03)
    if np.any(cluster):
        dominant_bpm = float(np.average(bpms[cluster], weights=weights[cluster]))

    return float(dominant_bpm)


def analyze_bpm(
    audio_path: str,
    session: Optional[ort.InferenceSession] = None,
    chunk_duration: float = CHUNK_DURATION,
    overlap_duration: float = OVERLAP_DURATION,
) -> dict:
    if session is None:
        session = load_model_from_memory()

    info = sf.info(audio_path)
    file_sr = info.samplerate
    total_frames = info.frames

    chunk_file_samples = int(chunk_duration * file_sr)
    overlap_file_samples = int(overlap_duration * file_sr)
    step_file_samples = chunk_file_samples - overlap_file_samples
    if step_file_samples <= 0:
        step_file_samples = chunk_file_samples

    all_beat = []
    all_downbeat = []

    with sf.SoundFile(audio_path) as f:
        offset = 0
        while offset < total_frames:
            f.seek(offset)
            chunk = f.read(chunk_file_samples)

            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)

            if file_sr != SAMPLE_RATE:
                chunk = librosa.resample(chunk, orig_sr=file_sr, target_sr=SAMPLE_RATE)

            mel_input = _compute_mel(chunk, SAMPLE_RATE)
            mel_input = mel_input[np.newaxis, ...]

            outputs = session.run(None, {"mel": mel_input})
            beat_act = np.asarray(outputs[0][0], dtype=np.float64)
            downbeat_act = np.asarray(outputs[1][0], dtype=np.float64)

            if offset > 0:
                overlap_frames = int(overlap_duration * SAMPLE_RATE / HOP_LENGTH)
                if overlap_frames < len(beat_act):
                    beat_act = beat_act[overlap_frames:]
                    downbeat_act = downbeat_act[overlap_frames:]

            all_beat.append(beat_act)
            all_downbeat.append(downbeat_act)

            offset += step_file_samples

    beat_activation = np.concatenate(all_beat)
    downbeat_activation = np.concatenate(all_downbeat)
    total_duration = len(beat_activation) * HOP_LENGTH / SAMPLE_RATE
    activation_context = _prepare_activation_context(beat_activation)
    activation_score_cache: dict[tuple[float, float], float] = {}

    beat_times = _detect_peaks(beat_activation, HOP_LENGTH, SAMPLE_RATE)
    downbeat_times = _detect_peaks(
        downbeat_activation,
        HOP_LENGTH,
        SAMPLE_RATE,
        max_bpm=300.0 / BEATS_PER_BAR,
    )

    bpm = _calculate_bpm(
        beat_times,
        beat_activation=beat_activation,
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
        beats_per_bar=BEATS_PER_BAR,
        activation_context=activation_context,
        activation_score_cache=activation_score_cache,
    )
    regularized_beat_times = _regularize_beat_times(
        beat_times,
        beat_activation=beat_activation,
        bpm=bpm,
        total_duration=total_duration,
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
    )
    regularized_bpm = _calculate_bpm(
        regularized_beat_times,
        beat_activation=beat_activation,
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
        beats_per_bar=BEATS_PER_BAR,
        activation_context=activation_context,
        activation_score_cache=activation_score_cache,
    )

    raw_score = _score_bpm_on_activation(
        beat_activation,
        bpm,
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
        context=activation_context,
        score_cache=activation_score_cache,
    )
    regularized_score = _score_bpm_on_activation(
        beat_activation,
        regularized_bpm,
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
        context=activation_context,
        score_cache=activation_score_cache,
    )

    beat_count_ratio = (
        len(regularized_beat_times) / len(beat_times)
        if len(beat_times) > 0
        else 1.0
    )
    score_gain = regularized_score - raw_score
    bpm_drift = abs(regularized_bpm - bpm)
    should_use_regularized = (
        len(regularized_beat_times) > 0
        and 0.8 <= beat_count_ratio <= 1.35
        and (
            score_gain >= 0.01
            or (score_gain >= 0.003 and bpm_drift <= max(2.0, bpm * 0.03))
        )
    )

    if should_use_regularized:
        beat_times = regularized_beat_times
        bpm = regularized_bpm

    return {
        "bpm": round(bpm, 2),
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times.tolist(),
    }


app = typer.Typer()


@app.command()
def analyze(
    audio_path: pathlib.Path = typer.Argument(..., help="音频文件路径", exists=True, dir_okay=False),
    chunk_duration: float = typer.Option(CHUNK_DURATION, "--chunk-duration", "-c", help="分块时长（秒）"),
    overlap_duration: float = typer.Option(OVERLAP_DURATION, "--overlap-duration", "-o", help="分块重叠时长（秒）"),
    model_path: Optional[pathlib.Path] = typer.Option(None, "--model-path", "-m", help="加密模型文件路径"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出结果"),
) -> None:
    session = load_model_from_memory(str(model_path) if model_path else MODEL_PATH)

    result = analyze_bpm(
        str(audio_path),
        session=session,
        chunk_duration=chunk_duration,
        overlap_duration=overlap_duration,
    )

    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"BPM: {result['bpm']}")


if __name__ == "__main__":
    app()
