from __future__ import annotations

import base64
import binascii
import asyncio
import concurrent.futures
import contextlib
import hashlib
import hmac
import io
import json
import math
import os
import pathlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default
from urllib.parse import urljoin
from typing import Any, BinaryIO, Literal, TypeAlias

import asyncio_oss
import anyio
import httpx2
import more_itertools
import numpy as np
import oss2
import pypinyin
import soundfile as sf
import tenacity
import typer
import wanakana
from httpx2._content import encode_multipart_data
from loguru import logger
from Crypto.Cipher import AES
from Crypto.Util import Padding
from pydub import AudioSegment
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ds2aces.ace.api_client import default_headers
from ds2aces.ace.config import (
    ACE_CONFIG_ROOT,
    ACE_API_BASE_URL,
    read_ace_login_config,
    read_ace_user_info_config,
    write_ace_login_config,
    write_ace_user_info_config,
)
from ds2aces.ace.constants import DEFAULT_SEED, MIN_NOTE_DURATION, MIN_SILENCE_DURATION, MAX_PIECE_DURATION, PIECE_SPLIT_GAP_PASS1, PIECE_SPLIT_GAP_PASS2
from ds2aces.ace.model import (
    AceParam,
    AceEngineBody,
    AceEngineBodyV2,
    AcesPieceParams,
    AcesSimpleNote,
    AcesProject,
    compress_ace_segment,
)
from ds2aces.ace.latin_g2p import MLG2PService
from ds2aces.ds.ds_file import DsProject
from ds2aces.ds.phoneme_dict import get_ds_phone_dict, get_vowels_set
from ds2aces.local.pipeline import Pipeline
from ds2aces.local.utils import provider_names, resolve_mlaudio_dir
from ds2aces.utils.music_math import hz2midi, note2midi

with contextlib.suppress(ImportError, OSError):
    from timsdk.manager import TIMManager

if "TIMManager" not in globals():
    TIMManager = None

app = typer.Typer()

CENTS_RE = re.compile(r"[+-]\d+$")
AcesItem: TypeAlias = dict[str, Any]
AcesNoteItem: TypeAlias = dict[str, Any]
LATIN_G2P_LANGUAGES = {"en", "spa", "fr", "it", "pt"}
LATIN_G2P_LANGUAGE_MAP = {"en": "en", "spa": "es", "fr": "fr", "it": "it", "pt": "pt"}
ACE_PLAN_LANGUAGE_MAP = {
    "ch": "zh",
    "jp": "jp",
    "ko": "ko",
    "en": "eng",
    "spa": "spa",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
}
_LATIN_G2P_SERVICE: MLG2PService | None = None


def get_latin_g2p_service() -> MLG2PService:
    global _LATIN_G2P_SERVICE
    if _LATIN_G2P_SERVICE is None:
        model_dir = pathlib.Path(__file__).resolve().parents[1] / "assets"
        _LATIN_G2P_SERVICE = MLG2PService(model_dir)
    return _LATIN_G2P_SERVICE


def get_ace_phone_plan(ace_phone_dict: dict[str, Any], language: str) -> dict[str, Any]:
    ace_language = ACE_PLAN_LANGUAGE_MAP[language]
    for plan in ace_phone_dict["plans"]:
        if plan.get("language") == ace_language:
            return plan
    raise ValueError(f"Cannot find ACE phoneme plan for language {language!r}")


def get_ace_vowels_set(ace_phone_plan: dict[str, Any]) -> set[str]:
    phon_class = ace_phone_plan.get("phon_class", {})
    return set(phon_class.get("tail", []))


@dataclass
class TranscriptionTimListener:
    request_id: str
    song_hash: str
    fallback_url: str
    download_dir: pathlib.Path
    client: httpx2.AsyncClient
    loop: asyncio.AbstractEventLoop
    result_event: threading.Event = field(default_factory=threading.Event)
    result_url: str | None = None
    downloaded_path: pathlib.Path | None = None
    matched_message: dict[str, Any] | None = None
    download_future: concurrent.futures.Future[pathlib.Path] | None = None

    def handle_log(self, level: int, message: str, metadata: dict[str, Any]) -> None:
        return None

    def handle_login(
        self, code: int, desc: str, json_param: str, metadata: dict[str, Any]
    ) -> None:
        return None

    def handle_messages(self, msgs: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        for msg in msgs:
            if self.result_event.is_set():
                return
            logger.debug(f"Received TIM message: {msg}")
            if result_url := _find_matching_transcription_url(
                msg, self.request_id, self.song_hash
            ):
                self.result_url = result_url
                self.matched_message = msg
                self.result_event.set()
                logger.info("Received transcription result through TIM")
                logger.info(result_url)
                self.download_future = asyncio.run_coroutine_threadsafe(
                    self.download_result(result_url), self.loop
                )
                return

    async def download_result(self, result_url: str) -> pathlib.Path:
        target_path = self.download_dir / f"{self.song_hash}.ace"
        response = await self.client.get(result_url)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        self.downloaded_path = target_path
        logger.info(f"Downloaded transcription to {target_path}")
        return target_path


def _decode_nested_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            with contextlib.suppress(json.JSONDecodeError):
                return _decode_nested_json(json.loads(stripped))
    if isinstance(value, list):
        return [_decode_nested_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_nested_json(item) for key, item in value.items()}
    return value


def _iter_nested_values(value: Any):
    decoded = _decode_nested_json(value)
    yield decoded
    if isinstance(decoded, dict):
        for item in decoded.values():
            yield from _iter_nested_values(item)
    elif isinstance(decoded, list):
        for item in decoded:
            yield from _iter_nested_values(item)


def _extract_urls(value: Any) -> list[str]:
    urls: list[str] = []
    for item in _iter_nested_values(value):
        if isinstance(item, dict):
            for key in ("file_url", "fileUrl", "url", "download_url", "downloadUrl"):
                url = item.get(key)
                if isinstance(url, str) and url.startswith("http"):
                    urls.append(url)
        elif isinstance(item, str) and item.startswith("http"):
            urls.append(item)
    return urls


def _message_matches_request(value: Any, request_id: str, song_hash: str) -> bool:
    for item in _iter_nested_values(value):
        if isinstance(item, dict):
            for key in ("request_id", "requestId", "task_id", "taskId"):
                if item.get(key) == request_id:
                    return True
        elif isinstance(item, str) and (request_id in item or song_hash in item):
            return True
    return False


def _find_matching_transcription_url(
    message: dict[str, Any], request_id: str, song_hash: str
) -> str | None:
    if not _message_matches_request(message, request_id, song_hash):
        return None
    for url in _extract_urls(message):
        return url
    return None

def cut_aces(aces: AcesItem) -> list[AcesItem]:
    version = aces["version"]
    original_note_list = aces["notes"]
    piece_params = aces.get("piece_params", {})

    filtered_list = []
    for note in original_note_list:
        if note["end_time"] - note["start_time"] > MAX_PIECE_DURATION:
            logger.warning("note too long, ignore!")
        else:
            filtered_list.append(note)

    def corase_cut(list_to_cut: list[AcesNoteItem]) -> list[list[AcesNoteItem]]:
        corase_result = []
        temp_list = []
        for i in range(len(list_to_cut)):
            note = list_to_cut[i]
            if i == 0:
                temp_list.append(note)
            else:
                last_note = list_to_cut[i-1]
                if note["start_time"] - last_note["end_time"] > MIN_SILENCE_DURATION:
                    corase_result.append(temp_list)
                    temp_list = []
                if i == len(list_to_cut) - 1:
                    temp_list.append(note)
                    corase_result.append(temp_list)
                else:
                    temp_list.append(note)

        return corase_result

    def gap_size(prev: AcesNoteItem, nxt: AcesNoteItem) -> float:
        return float(nxt["start_time"]) - float(prev["end_time"])

    def candidate_cut_indices(list_to_cut: list[AcesNoteItem], min_gap: float) -> list[int]:
        # cut_index points at the first note of the right half; only at real gaps
        # and never inside a slur group (which would split a syllable).
        indices = []
        for i in range(1, len(list_to_cut)):
            if list_to_cut[i]["type"] == "slur":
                continue
            if gap_size(list_to_cut[i - 1], list_to_cut[i]) >= min_gap:
                indices.append(i)
        return indices

    def best_cut(list_to_cut: list[AcesNoteItem], indices: list[int], target: float) -> int:
        # score = (target - left_len) * gap_size, maximised. Equivalent to
        # preferring large gaps that sit near the middle of the segment.
        list_start = float(list_to_cut[0]["start_time"])
        return max(
            indices,
            key=lambda i: (
                target - (float(list_to_cut[i - 1]["end_time"]) - list_start)
            ) * gap_size(list_to_cut[i - 1], list_to_cut[i]),
        )

    def simple_cut(list_to_cut: list[AcesNoteItem]) -> list[list[AcesNoteItem]]:
        target = (float(list_to_cut[-1]["end_time"]) - float(list_to_cut[0]["start_time"])) / 2

        # Multi-pass fallback:
        for min_gap in (PIECE_SPLIT_GAP_PASS1, PIECE_SPLIT_GAP_PASS2, 0.0):
            indices = candidate_cut_indices(list_to_cut, min_gap)
            if indices:
                cut_index = best_cut(list_to_cut, indices, target)
                return [list_to_cut[:cut_index], list_to_cut[cut_index:]]

        # Last-resort hard cut: find a non-slur boundary closest to the time midpoint.
        middle_time = float(list_to_cut[0]["start_time"]) + target
        cut_index = min(
            (i for i in range(1, len(list_to_cut)) if list_to_cut[i]["type"] != "slur"),
            key=lambda x: abs(float(list_to_cut[x]["start_time"]) - middle_time),
            default=len(list_to_cut) // 2,
        )
        return [list_to_cut[:cut_index], list_to_cut[cut_index:]]

    def fine_cut(list_to_cut: list[AcesNoteItem], max_length: float) -> list[list[AcesNoteItem]]:
        list_length = float(list_to_cut[-1]["end_time"]) - float(list_to_cut[0]["start_time"])
        if list_length <= max_length or len(list_to_cut) < 2:
            return [list_to_cut]
        halves = simple_cut(list_to_cut)
        result = []
        for half in halves:
            if half:
                result.extend(fine_cut(half, max_length))
        return result

    corase_list = corase_cut(filtered_list)
    fine_list = []
    for corase in corase_list:
        fine_list.extend(fine_cut(corase, MAX_PIECE_DURATION))

    aces_list = []
    for piece in fine_list:
        aces_piece = aces.copy()
        aces_piece.update({"notes": piece})
        aces_piece.update({"version": version})

        if piece_params:
            piece_start_time = float(piece[0]["start_time"])
            piece_end_time = float(piece[-1]["end_time"])
            piece_duration = piece_end_time - piece_start_time

            new_piece_params = {}

            for param_name, param_value in piece_params.items():
                if isinstance(param_value, dict):
                    new_param_value = {}
                    for sub_param_name, sub_param in param_value.items():
                        new_param_value[sub_param_name] = []
                        for sub_param_value in sub_param:
                            new_sub_param_value = sub_param_value.copy()
                            if "values" in new_sub_param_value and isinstance(new_sub_param_value["values"], list):
                                original_start = sub_param_value.get("start_time", 0)
                                original_values = sub_param_value.get("values", [])
                                hop_time = sub_param_value["hop_time"]
                                if piece_end_time <= original_start or piece_start_time >= original_start + len(original_values) * hop_time:
                                    continue
                                if piece_start_time < original_start:
                                    start_index = 0
                                    new_sub_param_value["start_time"] = original_start
                                    new_length = math.ceil((piece_end_time - original_start) / hop_time)
                                else:
                                    start_index = math.floor((piece_start_time - original_start) / hop_time)
                                    new_sub_param_value["start_time"] = piece_start_time
                                    new_length = math.ceil(piece_duration / hop_time)

                                if original_values:
                                    new_sub_param_value["values"] = original_values[start_index:start_index+new_length]
                                    if len(new_sub_param_value["values"]):
                                        new_param_value[sub_param_name].append(new_sub_param_value)
                    new_piece_params[param_name] = new_param_value
                else:
                    new_piece_params[param_name] = param_value
            aces_piece.update({"piece_params": new_piece_params})
        aces_list.append(aces_piece)
    if aces_list:
        aces_list[0].setdefault("pad", {})
        if aces_list[0]["pad"] is None:
            aces_list[0]["pad"] = {}
        aces_list[0]["pad"]["begin"] = {
            "type": "sp",
            "start_time": max(0, aces_list[0]["notes"][0]["start_time"] - 0.6),
            "end_time": aces_list[0]["notes"][0]["start_time"],
        }
        for prev_piece, next_piece in more_itertools.pairwise(aces_list):
            prev_piece["pad"]["end"] = {
                "type": "sp",
                "start_time": prev_piece["notes"][-1]["end_time"],
                "end_time": min(prev_piece["notes"][-1]["end_time"] + 0.6, next_piece["notes"][0]["start_time"]),
            }
            next_piece.setdefault("pad", {})
            if next_piece["pad"] is None:
                next_piece["pad"] = {}
            next_piece["pad"]["begin"] = {
                "type": "sp",
                "start_time": max(prev_piece["notes"][-1]["end_time"], next_piece["notes"][0]["start_time"] - 0.6),
                "end_time": next_piece["notes"][0]["start_time"],
            }
        aces_list[-1]["pad"]["end"] = {
            "type": "sp",
            "start_time": aces_list[-1]["notes"][-1]["end_time"],
            "end_time": aces_list[-1]["notes"][-1]["end_time"] + 0.6,
        }
    return aces_list


def filter_short_note(aces: AcesItem) -> AcesItem:
    filtered_notes = [
        note
        for note in aces["notes"]
        if note["end_time"] - note["start_time"] > MIN_NOTE_DURATION
    ]
    aces["notes"] = filtered_notes
    return aces


async def render_aces(client: httpx2.AsyncClient, aces_file: pathlib.Path, router_id: int) -> None:
    as_token = await fetch_as_token(client)  
    if not as_token:
        raise ValueError("fetch as token failed")

    long_aces = json.loads(
        aces_file.read_text(encoding="utf-8")
    )
    original_length = len(long_aces["notes"])
    long_aces = filter_short_note(long_aces)
    if original_length != len(long_aces["notes"]):
        logger.warning(f"!!!! notes shorter than {MIN_NOTE_DURATION}s is filtered, notes number changed from {original_length} to {len(long_aces['notes'])}")

    cutted_aces = cut_aces(long_aces)

    await rendering_ace_list(client, cutted_aces, aces_file.with_suffix(".wav"), as_token, router_id)


async def fetch_router_config(client: httpx2.AsyncClient, echo: bool = False) -> dict[int, str]:
    router_response = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/voice/seed/v3",
        ),
        params={
            "version": "2"
        }
    )
    if not router_response.is_error:
        resp_data = router_response.json()
        if resp_data["code"] == 200:
            code2router_url = {
                router["id"]: (router["router"], router["version"])
                for router in resp_data["data"]["router_list"]
                if router["is_show"]
            }
            if echo:
                console = Console(color_system="256")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("ID", justify="left", style="cyan")
                table.add_column("Router Name", justify="left", style="cyan")
                table.add_column("URL", justify="left", style="cyan")
                table.add_column("Version", justify="left", style="cyan")
                table.add_column("Supported Languages", justify="left", style="cyan")
                for router in resp_data["data"]["router_list"]:
                    table.add_row(
                        str(router["id"]),
                        router["router_name"],
                        router["router"],
                        str(router["version"]),
                        ", ".join(router["support_lan_list"]),
                    )
                console.print(table)
            return code2router_url
        else:
            logger.error(f"fetch router config failed: {resp_data['error']}")
            return {}
    else:
        logger.error(f"fetch router config failed: {router_response.text}")
        return {}

@tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, max=10),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)
async def send_request(client: httpx2.AsyncClient, compose_body: AceEngineBody | AceEngineBodyV2, files: dict[str, tuple[str, BinaryIO, str]], router_url: str) -> httpx2.Response:
    form_boundary = f"----WebKitFormBoundary{time.time()}"
    request = client.build_request("POST", router_url)
    headers, stream = encode_multipart_data(
        data=compose_body.model_dump(mode="json"),
        files=files,
        boundary=form_boundary.encode()
    )
    request.stream = stream
    request.headers.update(headers)
    return await client.send(request)


async def download_and_open_audio(client: httpx2.AsyncClient, url: str) -> tuple[np.ndarray, int]:
    response = await client.get(url)
    response.raise_for_status()

    with io.BytesIO(response.content) as file:
        data, samplerate = sf.read(file)
        return data, samplerate

async def one_piece_compose(client: httpx2.AsyncClient, ace_token: str, aces: dict, router_url: str, router_version: int) -> tuple[float, np.ndarray, int]:
    user_config = read_ace_user_info_config()
    user_id = json.loads(
        json.loads(user_config.get("user_info_group", "user_info_key"))
    )["uid"]
    timestamp = int(time.time() * 1000)
    upload_path = f"{user_id}_{timestamp}.aces"

    if router_version == 1:
        compose_body = AceEngineBody(
            compress_type="zstd",
            flag=".aces",
            ace_token=ace_token,
            pipeline_business=2,
        )
    else:
        compose_body = AceEngineBodyV2(
            ace_token=ace_token,
            session_id=str(timestamp),
            context_id="0",
            response_type="2",
            mix_info=aces["mix_info"] if "mix_info" in aces else "",
        )
    # logger.debug(aces)
    files = {
        "file": (
            upload_path,
            io.BytesIO(compress_ace_segment(json.dumps(aces), raw=router_version>=2)),
            "application/octet-stream",
        )
    }
    compose_resp = await send_request(
        client, compose_body, files, router_url
    )
    # logger.debug(compose_resp.json())
    result = compose_resp.json()["data"][0]
    audio_url = result.get('audio')
    audio_data, samplerate = await download_and_open_audio(client, audio_url)
    pst = result.get('pst')

    return pst, audio_data, samplerate


_ACE_ENGINE_BODY_V2_FIELD_ORDER = (
    "ace_token",
    "session_id",
    "context_id",
    "mix_info",
    "inpainting_time_list",
    "delete_time_list",
    "response_type",
    "version",
)


def _ordered_engine_body_v2_data(compose_body: AceEngineBodyV2) -> dict:
    data = compose_body.model_dump(mode="json")
    return {field: data[field] for field in _ACE_ENGINE_BODY_V2_FIELD_ORDER if field in data}


def _encode_file_first_multipart_data(
    *,
    data: dict,
    files: dict,
    boundary: bytes,
) -> tuple[dict[str, str], object]:
    from httpx2 import ByteStream
    from httpx2._multipart import DataField, FileField

    mp_fields = [FileField(name=name, value=value) for name, value in files.items()]
    for name, value in data.items():
        if isinstance(value, (tuple, list)):
            mp_fields.extend(DataField(name=name, value=item) for item in value)
        else:
            mp_fields.append(DataField(name=name, value=value))

    chunks: list[bytes] = []
    for mp_field in mp_fields:
        chunks.append(b"--%s\r\n" % boundary)
        chunks.extend(mp_field.render())
        chunks.append(b"\r\n")
    chunks.append(b"--%s--\r\n" % boundary)
    body = b"".join(chunks)
    headers = {
        "Content-Length": str(len(body)),
        "Content-Type": f"multipart/form-data; boundary={boundary.decode('ascii')}",
    }
    return headers, ByteStream(body)


def _inpainting_time_list_for_notes_local(aces: dict) -> str:
    """Variant for response_type=1: uses actual note start/end span."""
    notes = [
        note for note in aces.get("notes", [])
        if note.get("type") not in {"br", "sp"} and "start_time" in note and "end_time" in note
    ]
    if not notes:
        return "[]"
    start_time = min(float(note["start_time"]) for note in notes)
    end_time = max(float(note["end_time"]) for note in notes)
    return json.dumps([{"end_time": end_time, "start_time": start_time}], separators=(",", ":"))


async def one_piece_compose_tokens(
    client: httpx2.AsyncClient,
    ace_token: str,
    aces: dict,
    router_url: str,
    router_version: int,
) -> tuple[float, np.ndarray]:
    """合成单个片段，使用 response_type=1 获取中间表示（tokens）用于本地合成。

    Returns:
        (pst, tokens) where tokens is [N, 32] uint16.
    """
    user_config = read_ace_user_info_config()
    user_id = json.loads(
        json.loads(user_config.get("user_info_group", "user_info_key"))
    )["uid"]
    timestamp = int(time.time() * 1000)
    upload_path = f"{user_id}_{timestamp}.aces"

    if router_version == 1:
        compose_body = AceEngineBody(
            compress_type="zstd",
            flag=".aces",
            ace_token=ace_token,
            pipeline_business=2,
        )
    else:
        compose_body = AceEngineBodyV2(
            ace_token=ace_token,
            session_id=f"{user_id}_{timestamp}",
            context_id="1",
            response_type="1",
            mix_info=aces.get("mix_info", ""),
            inpainting_time_list=_inpainting_time_list_for_notes_local(aces),
        )

    # Strip None fields to match real ACE Studio .aces format
    aces = {k: v for k, v in aces.items() if v is not None}
    for note in aces.get("notes", []):
        for k in [k for k, v in note.items() if v is None]:
            del note[k]

    files = {
        "file": (
            upload_path,
            io.BytesIO(compress_ace_segment(json.dumps(aces), raw=router_version >= 2)),
            "application/octet-stream",
        )
    }

    form_boundary = f"----WebKitFormBoundary{time.time()}"
    request_data = (
        _ordered_engine_body_v2_data(compose_body)
        if isinstance(compose_body, AceEngineBodyV2)
        else compose_body.model_dump(mode="json")
    )

    request = client.build_request("POST", router_url)
    encode_multipart = (
        _encode_file_first_multipart_data
        if isinstance(compose_body, AceEngineBodyV2)
        else encode_multipart_data
    )
    headers, stream = encode_multipart(data=request_data, files=files, boundary=form_boundary.encode())
    request.stream = stream
    request.headers.update(headers)
    compose_resp = await client.send(request)

    # Parse multipart response — tokens are in a named part
    response_bytes = getattr(compose_resp, "content", b"") or b""
    response_headers = getattr(compose_resp, "headers", {}) or {}
    response_ct = response_headers.get("content-type", "") if hasattr(response_headers, "get") else ""

    if "multipart/form-data" not in response_ct.lower() or not response_bytes:
        msg = f"ACE Studio compose(response_type=1) returned non-multipart response for {router_url}"
        raise RuntimeError(msg)

    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {response_ct}\r\nMIME-Version: 1.0\r\n\r\n".encode() + response_bytes
    )

    pst: float = 0.0
    tokens_file_bytes: bytes | None = None
    error_msg: str | None = None

    for part in message.iter_parts():
        raw_payload = part.get_payload(decode=True)
        payload = raw_payload if isinstance(raw_payload, bytes) else b""
        name = part.get_param("name", header="content-disposition")

        if name == "data" and payload:
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                data_obj = json.loads(payload.decode("utf-8"))
                if isinstance(data_obj, dict):
                    pst = float(data_obj.get("pst") or data_obj.get("offset") or 0.0)
        elif name == "tokens" and payload:
            tokens_file_bytes = payload
        elif name == "error" and payload:
            error_msg = payload.decode("utf-8").strip()

    if error_msg:
        raise RuntimeError(f"ACE Studio compose(response_type=1) error: {error_msg}")
    if tokens_file_bytes is None:
        raise RuntimeError(
            f"No tokens.npz in compose(response_type=1) response from {router_url}"
        )

    # Load .npz archive
    npz = np.load(io.BytesIO(tokens_file_bytes))
    tokens: np.ndarray | None = None
    for key in npz.files:
        arr = npz[key]
        if arr.ndim == 2 and arr.shape[1] == 32:
            tokens = arr
            break
    if tokens is None:
        tokens = npz[npz.files[0]]

    return pst, tokens

async def rendering_ace_list(client: httpx2.AsyncClient, ace_list: list[AcesItem], save_to_path: pathlib.Path, as_token: str, router_id: int) -> None:
    vocal_offset = None
    concat_audio = []

    code2router_url = await fetch_router_config(client)
    if not code2router_url:
        return
    router_url, router_version = code2router_url[router_id]

    if not ace_list:
        return
    for aces in track(ace_list, description="Rendering ACE Sequence files..."):
        pst, audio_data, samplerate = await one_piece_compose(client, as_token, aces, router_url, router_version)

        if vocal_offset is None:
            vocal_offset = pst
        else:
            start_index = int((pst - vocal_offset) * samplerate)

            if(len(concat_audio) < start_index):
                concat_audio.extend([0] * (start_index - len(concat_audio)))
            else:
                concat_audio = concat_audio[:start_index]
        concat_audio.extend(audio_data)

    sf.write(save_to_path, concat_audio, samplerate)
    logger.info(f"In relation to the project, the vocal offset is: {vocal_offset}")

    explanation = (
        "for example:  \n"
        "   If the accompaniment starts at 1.2 seconds relative to the project \n"
        "   The vocal starts at 2 seconds relative to the project \n"
        "   The vocal actually starts 0.8 seconds after the accompaniment."
    )
    logger.info(explanation)


async def download_phoneme_data(client: httpx2.AsyncClient) -> dict[str, Any]:
    config_resp = await client.get(
        urljoin(ACE_API_BASE_URL, "/api/as/conf/client"),
    )
    config_resp.raise_for_status()
    config_data = config_resp.json()
    multi_lan_plan_data = config_data["data"]["multi_lan_plan"]
    multi_lan_plan_url_hash = hashlib.md5((multi_lan_plan_data["url"] + f'{multi_lan_plan_data["version"]:g}').encode()).hexdigest()
    multi_lan_plan_path = ACE_CONFIG_ROOT / "ClientSetUp" / f'{multi_lan_plan_url_hash}{pathlib.Path(multi_lan_plan_data["url"]).suffix}'
    if not multi_lan_plan_path.exists():
        with multi_lan_plan_path.open("wb") as f:
            async with client.stream("GET", multi_lan_plan_data["url"]) as stream:
                async for chunk in stream.aiter_bytes():
                    f.write(chunk)
    return json.loads(
        multi_lan_plan_path.read_text(encoding="utf-8")
    )

@app.command()
def start_trial() -> None:
    config_parser = read_ace_login_config()
    token = json.loads(
        json.loads(config_parser["login_token_group"]["login_token_key"])
    )["token"]
    user_info_config = read_ace_user_info_config()
    uid = str(json.loads(json.loads(user_info_config["user_info_group"]["user_info_key"]))["uid"])
    app_headers = default_headers()
    client = httpx2.Client(
        follow_redirects=True,
        timeout=10,
        headers=app_headers,
        verify=False,
    )
    app_headers["token"] = token
    machine_code = str(uuid.uuid4())
    platform = "win"
    timestamp = str(int(time.time() * 1000))
    hmac_key = os.environ.get("ACE_TRIAL_HMAC_KEY", "").encode()
    hmac_msg = f"mc={machine_code}&platform={platform}&ts={timestamp}&uid={uid}&v=1".encode()
    hmac_sign = binascii.b2a_hex(hmac.digest(
        hmac_key, hmac_msg, "sha256"
    )).decode()[:16]
    aes_key = os.environ.get("ACE_TRIAL_AES_KEY", "").encode()
    data_content = json.dumps(
        {
            "mc": machine_code,
            "platform": platform,
            "sign": hmac_sign,
            "ts": timestamp,
            "v": 1
        }, separators=(",", ":")
    ).encode()
    raw_content = Padding.pad(data_content, 16)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    enc_content = cipher.encrypt(raw_content)
    enc_data = base64.b64encode(enc_content).decode()       
    resp = client.post(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/user/vip/trial/v2"
        ),
        headers=app_headers,
        json={
            "data": enc_data
        }
    )
    resp_data = resp.json()
    logger.info(f"Post {resp.url}")
    logger.info(resp_data)



async def pre_login() -> httpx2.AsyncClient | None:
    config_parser = read_ace_login_config()
    token = json.loads(
        json.loads(config_parser["login_token_group"]["login_token_key"])
    )["token"]
    client = httpx2.AsyncClient(
        follow_redirects=True,
        timeout=10,
        headers=default_headers(token),
        verify=False,
    )
    refresh_token_resp = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/auth/token",
        ),
    )
    if not refresh_token_resp.is_error:
        resp_data = refresh_token_resp.json()
        if resp_data["code"] == 200 and resp_data["error"] is None:
            refresh_token = resp_data["data"]["auth"]["token"]
            client.headers["token"] = refresh_token
            config_parser["login_token_group"]["login_token_key"] = json.dumps(
                json.dumps({"token": refresh_token})
            )
            write_ace_login_config(config_parser)
        else:
            logger.exception("Refresh token failed")
            return None
    else:
        logger.exception("Refresh token failed")
        return None
    return client


async def fetch_as_token(client: httpx2.AsyncClient) -> str:
    as_token_response = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/ai/token/as",
        ),
    )
    if not as_token_response.is_error:
        resp_data = as_token_response.json()
        if resp_data["code"] == 200:
            return resp_data["data"]["as"]["token"]
    return ""


@app.command()
def login(phone_number: str) -> None:
    is_email = "@" in phone_number
    client = httpx2.Client(follow_redirects=True, timeout=10, headers=default_headers())
    typer.echo(f"Login using {phone_number}")
    if is_email:
        response = client.post(
            urljoin(
                ACE_API_BASE_URL,
                "/api/as/web/auth/email/check",
            ),
            json={"email": phone_number},
        )
        resp_data = response.json()
        password = typer.prompt("Please input the password", hide_input=True)
        if not resp_data["data"].get("email_exist"):
            client.get(
                urljoin(
                    ACE_API_BASE_URL,
                    "/api/as/web/auth/email/code/send",
                ),
                params={"email": phone_number},
            )
            code = typer.prompt("Please input the code sent to your email")
            response = client.post(
                urljoin(
                    ACE_API_BASE_URL,
                    "/api/as/web/auth/email/register",
                ),
                json={
                    "email": phone_number,
                    "password": password,
                    "code": code,
                },
            )
        response = client.post(
            urljoin(
                ACE_API_BASE_URL,
                "/api/as/auth/email/login",
            ),
            json={"email": phone_number, "password": password},
        )
    else:
        response = client.get(
            urljoin(
                ACE_API_BASE_URL,
                "/api/as/auth/code",
            ),
            params={"phone": phone_number},
        )
        if not response.is_error:
            resp_data = response.json()
            if resp_data["code"] == 200 and resp_data["error"] is None:
                typer.echo(f"Send login code to {phone_number}")
                verify_code = typer.prompt("Please input the verify code")[:4]
                response = client.get(
                    urljoin(
                        ACE_API_BASE_URL,
                        "/api/as/auth/phone",
                    ),
                    params={"phone": phone_number, "code": verify_code},
                )
            else:
                logger.error(f"Send login code to {phone_number} failed: {resp_data['error']}")
                return
    if not response.is_error:
        resp_data = response.json()
        if resp_data["code"] == 200 and resp_data["error"] is None:
            typer.echo("Login success")
            config_parser = read_ace_login_config()
            config_parser.set(
                "login_token_group",
                "login_token_key",
                json.dumps(json.dumps(resp_data["data"]["auth"])),
            )
            write_ace_login_config(config_parser)
            user_info_config = read_ace_user_info_config()
            user_info_config.set(
                "user_info_group",
                "user_info_key",
                json.dumps(json.dumps(resp_data["data"]["user"])),
            )
            write_ace_user_info_config(user_info_config)


async def fetch_singers(client: httpx2.AsyncClient | None = None) -> None:
    if client is None:
        client = await pre_login()
    if client is None:
        return
    response = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/singer/list/v2",
        ),
        params={
            "version": "2.1"
        }
    )
    if not response.is_error:
        singers_data = response.json()
        if singers_data["code"] == 200:
            console = Console(color_system="256")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Code", justify="left", style="cyan")
            table.add_column("Seed", justify="left", style="cyan")
            table.add_column("Singer Name", justify="left", style="cyan")
            for singer_data in sorted(singers_data["data"]["official"]["singers"], key=lambda x: x["code"]):
                table.add_row(
                    str(singer_data["code"]),
                    str(singer_data["seed_id"]),
                    singer_data["name"],
                )
            console.print(table)
        else:
            logger.info(f"GET: {response.url}")
            logger.error(f"fetch singers failed: {response.text}")

@app.command()
def singers() -> None:
    anyio.run(fetch_singers)


def g2p(lyric: str, language: Literal["ch", "en", "jp", "spa", "ko", "pt", "fr", "it"] = "ch") -> str:
    if language == "ch":
        return " ".join(pypinyin.lazy_pinyin(lyric, style=pypinyin.Style.NORMAL))
    elif language == "jp":
        return wanakana.to_romaji(lyric, custom_romaji_mapping={"っ": "cl"})
    elif language in LATIN_G2P_LANGUAGES:
        latin_language = LATIN_G2P_LANGUAGE_MAP[language]
        return " ".join(get_latin_g2p_service().predict(latin_language, lyric))
    else:
        raise NotImplementedError(f"Language {language} is not supported yet")


async def ds_to_aces(
    in_path: pathlib.Path,
    output_dir: pathlib.Path,
    param: bool = False,
    render: bool = False,
    language: Literal["ch", "en", "jp", "spa", "ko", "pt", "fr", "it"] = "ch",
    local: bool = False,
    mlaudio_dir: pathlib.Path | None = None,
    provider: Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
) -> None:
    client = await pre_login()
    if not client:
        logger.error("login failed")
        return
    ACE_PHONE_DICT = await download_phoneme_data(client)
    if language == "ch":
        ds_phone_dict = get_ds_phone_dict("opencpop-extension", g2p=False)
        vowels_set = get_vowels_set("opencpop-extension")
        ace_phone_dict = get_ace_phone_plan(ACE_PHONE_DICT, language)
    elif language == "jp":
        ds_phone_dict = get_ds_phone_dict("japanese_dict_full", g2p=False)
        vowels_set = get_vowels_set("japanese_dict_full")
        ace_phone_dict = get_ace_phone_plan(ACE_PHONE_DICT, language)
    elif language in LATIN_G2P_LANGUAGES:
        ds_phone_dict = {}
        ace_phone_dict = get_ace_phone_plan(ACE_PHONE_DICT, language)
        vowels_set = get_ace_vowels_set(ace_phone_dict)
    else:
        raise NotImplementedError(f"Language {language} is not supported yet")
    ds_project = DsProject.model_validate_json(in_path.read_text(encoding="utf-8"))
    notes = []
    await fetch_singers(client)
    seed_id = typer.prompt("Please input the seed id", default=DEFAULT_SEED)
    if render or local:
        mix_info_resp = await client.post(
            urljoin(
                ACE_API_BASE_URL,
                "/api/as/voice/mix/str/v3",
            ),
            json={"seeds_list": [
                [{"s": 1, "t": 1, "s_id": int(seed_id)}],
            ]},
        )
        if not mix_info_resp.is_error:
            mix_info_data = json.loads(mix_info_resp.text)
            if mix_info_data["code"] == 200:
                mix_info = mix_info_data["data"][
                    "mix_info_list"
                ][0]
            elif mix_info_data.get("error"):
                logger.error(mix_info_data["error"])
                return
        await fetch_router_config(client, echo=True)
        router_id = int(typer.prompt("Please input the router id", default=1))
    else:
        mix_info = ""
        router_id = 1
    params = AcesPieceParams()
    for ds_item in ds_project.root:
        cur_time = float(ds_item.offset)
        pitch_params = []
        if param and ds_item.f0_timestep and ds_item.f0_seq:
            pitch_param = AceParam(start_time=cur_time, hop_time=ds_item.f0_timestep, values=[
                hz2midi(f0) for f0 in ds_item.f0_seq
            ])
            pitch_params.append(pitch_param)
        ph_index = 0
        midi_key = 69
        for lyric_index, slur_group in enumerate(
            more_itertools.split_before(
                enumerate(ds_item.note_slur or []),
                lambda pair: pair[1] == 0,
            )
        ):
            if not ds_item.note_dur:
                break
            text = ds_item.text[lyric_index]
            if text == "SP":
                next_time = cur_time + sum(
                    ds_item.note_dur[note_index]
                    for note_index, is_slur in slur_group
                )
                ph_index += 1
            elif text == "AP":
                next_time = cur_time + sum(
                    ds_item.note_dur[note_index]
                    for note_index, is_slur in slur_group
                )
                notes.append(
                    AcesSimpleNote(
                        start_time=cur_time,
                        end_time=next_time,
                        pitch=midi_key,
                        type="br",
                    )
                )
                ph_index += 1
            else:
                phoneme_buf = []
                consonant_time_head = []
                next_time = cur_time
                for note_index, is_slur in slur_group:
                    note = ds_item.note_seq[note_index]
                    midi_key = note2midi(CENTS_RE.sub("", note))
                    note_dur = ds_item.note_dur[note_index]
                    if is_slur:
                        if len(phoneme_buf):
                            if len(phoneme_buf) == 1:
                                pronunciation = phoneme_buf[0]
                            else:
                                ds_phone = " ".join(phoneme_buf)
                                if ds_phone in ds_phone_dict:
                                    pronunciation = ds_phone_dict[ds_phone]
                                else:
                                    pronunciation = g2p(text, language)
                            syllable_alias = ace_phone_dict.get("syllable_alias", {})
                            if pronunciation in syllable_alias:
                                pronunciation = syllable_alias[pronunciation]
                            if language in LATIN_G2P_LANGUAGES:
                                phone = pronunciation.split()
                                pronunciation = text
                            elif language != "ch" and pronunciation in ace_phone_dict["dict"]:
                                phone = ace_phone_dict["dict"][pronunciation]
                            else:
                                phone = []
                            notes.append(
                                AcesSimpleNote(
                                    start_time=cur_time,
                                    end_time=next_time,
                                    pitch=midi_key,
                                    type="general",
                                    syllable=pronunciation,
                                    phone=phone,
                                    consonant_time_head=consonant_time_head,
                                    language=language,
                                )
                            )
                            cur_time = next_time
                            phoneme_buf.clear()
                            consonant_time_head.clear()
                        next_time += note_dur
                        notes.append(
                            AcesSimpleNote(
                                start_time=cur_time,
                                end_time=next_time,
                                pitch=midi_key,
                                type="slur",
                                syllable="?",
                            )
                        )
                    else:
                        while len(phoneme_buf) < 2:
                            phone = ds_item.ph_seq[ph_index].rsplit("/", 1)[-1]
                            if phone in vowels_set and not len(phoneme_buf) and ds_item.ph_dur is not None:
                                vowel_dur = ds_item.ph_dur[ph_index]
                                consonant_time_head.append(vowel_dur)
                                silence_time_start = cur_time - vowel_dur
                                pitch_params = [
                                    seg
                                    for pp in pitch_params
                                    for seg in pp.silent(silence_time_start, cur_time)
                                ]
                            ph_index += 1
                            phoneme_buf.append(phone)
                            if len(phoneme_buf) == 1 and phone not in vowels_set:
                                break
                        next_time += note_dur
                if len(phoneme_buf):
                    if len(phoneme_buf) == 1:
                        pronunciation = phoneme_buf[0]
                    else:
                        ds_phone = " ".join(phoneme_buf)
                        if ds_phone in ds_phone_dict:
                            pronunciation = ds_phone_dict[ds_phone]
                        else:
                            pronunciation = g2p(text, language)
                    syllable_alias = ace_phone_dict.get("syllable_alias", {})
                    if pronunciation in syllable_alias:
                        pronunciation = syllable_alias[pronunciation]
                    if language in LATIN_G2P_LANGUAGES:
                        phone = pronunciation.split()
                        pronunciation = text
                    elif language != "ch" and pronunciation in ace_phone_dict["dict"]:
                        phone = ace_phone_dict["dict"][pronunciation]
                    else:
                        phone = []
                    notes.append(
                        AcesSimpleNote(
                            start_time=cur_time,
                            end_time=next_time,
                            pitch=midi_key,
                            type="general",
                            syllable=pronunciation,
                            phone=phone,
                            consonant_time_head=consonant_time_head,
                            language=language,
                        )
                    )
                    phoneme_buf.clear()
                    consonant_time_head.clear()
            cur_time = next_time
        params.pitch.user.extend(pitch_params)
    output_dir.mkdir(parents=True, exist_ok=True)
    aces_file = output_dir / f"{in_path.stem}.aces"
    aces_file.write_text(
        AcesProject(
            piece_params=params,
            notes=notes,
            mix_info=mix_info,
        ).model_dump_json(by_alias=True, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Successfully saved to '{aces_file.absolute()}'")
    if render:
        await render_aces(client, aces_file, router_id)
    if local:
        await render_aces_local(client, aces_file, router_id, output_dir, mlaudio_dir, provider)

async def render_aces_local(
    client: httpx2.AsyncClient,
    aces_file: pathlib.Path,
    router_id: int,
    output_dir: pathlib.Path,
    mlaudio_dir: pathlib.Path | None = None,
    provider: Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
) -> None:
    """Render ACES using cloud tokens + local inference pipeline."""
    as_token = await fetch_as_token(client)
    if not as_token:
        raise ValueError("fetch as token failed")

    long_aces = json.loads(aces_file.read_text(encoding="utf-8"))
    long_aces = filter_short_note(long_aces)
    cutted_aces = cut_aces(long_aces)

    code2router_url = await fetch_router_config(client)
    if not code2router_url:
        return
    router_url, router_version = code2router_url[router_id]

    # Initialize the local inference pipeline
    resolved_mlaudio_dir = resolve_mlaudio_dir(mlaudio_dir)
    pipeline = Pipeline(resolved_mlaudio_dir, provider_names(provider))
    logger.info(f"Local pipeline initialized (mlaudio: {resolved_mlaudio_dir}, provider: {provider})")

    vocal_offset: float | None = None
    concat_audio: list[float] = []

    for aces in track(cutted_aces, description="Rendering ACE tokens locally..."):
        # Step 1: Get tokens from cloud API (response_type=1)
        pst, tokens = await one_piece_compose_tokens(
            client, as_token, aces, router_url, router_version,
        )
        logger.info(f"Got tokens: shape={tokens.shape}, pst={pst}")

        # Step 2: Run local inference pipeline
        result = pipeline.run_from_tokens(tokens)
        audio_data = result["audio"]

        if vocal_offset is None:
            vocal_offset = pst
        else:
            start_index = int((pst - vocal_offset) * 44100)
            if len(concat_audio) < start_index:
                concat_audio.extend([0.0] * (start_index - len(concat_audio)))
            else:
                concat_audio = concat_audio[:start_index]
        concat_audio.extend(audio_data.tolist())

    save_to_path = output_dir / f"{aces_file.stem}.wav"
    sf.write(save_to_path, concat_audio, 44100)
    logger.info(f"Local synthesis saved to {save_to_path}")
    logger.info(f"In relation to the project, the vocal offset is: {vocal_offset}")

    explanation = (
        "for example:  \n"
        "   If the accompaniment starts at 1.2 seconds relative to the project \n"
        "   The vocal starts at 2 seconds relative to the project \n"
        "   The vocal actually starts 0.5 seconds after the accompaniment."
    )
    logger.info(explanation)

@app.command()
def ds2aces(
    in_path: pathlib.Path = typer.Argument("input.ds", exists=True, dir_okay=False),
    output_dir: pathlib.Path = typer.Argument(
        ".", exists=False, dir_okay=True, file_okay=False
    ),
    param: bool = typer.Option(True),
    render: bool = typer.Option(False),
    language: Literal["ch", "en", "jp", "spa", "ko", "pt", "fr", "it"] = "ch",
    local: bool = typer.Option(False, "--local", help="Use cloud tokens + local inference pipeline"),
    mlaudio_dir: pathlib.Path | None = typer.Option(
        None,
        "--mlaudio-dir",
        help="ACE Studio mlaudio directory for --local; auto-detected when omitted",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    provider: Literal["cpu", "cuda", "dml", "coreml"] = typer.Option(
        "cpu",
        "--provider",
        help="ONNX Runtime provider for --local",
    ),
) -> None:
    if language not in ["ch", "en", "jp", "spa", "pt", "fr", "it"]: # TODO: support Korean
        raise NotImplementedError(
            "Only Chinese, Japanese, English, Spanish, Portuguese, French and Italian are supported"
        )
    anyio.run(
        ds_to_aces,
        in_path,
        output_dir,
        param,
        render,
        language,
        local,
        mlaudio_dir,
        provider,
    )


async def transcribe_asynchronous(
    song_path: pathlib.Path,
    language: Literal["ch", "en", "jp", "es", "ita", "fra", "por", "kor", "note", "unknown"] = "ch",
    version: str = "1.0",
) -> None:
    client = await pre_login()
    if client is None:
        return
    user_config = read_ace_user_info_config()
    user_id = json.loads(
        json.loads(user_config.get("user_info_group", "user_info_key"))
    )["uid"]
    resp = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/oss/sts",
        )
    )
    sts_token = resp.json()["data"]["sts_token"]
    auth = oss2.StsAuth(
        sts_token["access_key"],
        sts_token["access_Secret"],
        sts_token["security_token"],
    )
    resp = await client.get(urljoin(ACE_API_BASE_URL, "/api/as/ai/token/vt"))
    vt_api_url = resp.json()["data"]["vt"]["router"]
    vt_api_token = resp.json()["data"]["vt"]["token"]
    async with asyncio_oss.Bucket(
        auth, sts_token["endpoint"], sts_token["bucket"]
    ) as bucket:
        upload_prefix = "app/user/vt/audio"

        song = AudioSegment.from_file(song_path.absolute()).set_frame_rate(48000)
        ogg_buffer = io.BytesIO()
        song.export(ogg_buffer, format="ogg")
        song_hash = hashlib.md5(song_path.read_bytes()).hexdigest()
        upload_path = f"{upload_prefix}/{song_hash}.ogg"
        upload_result = await bucket.put_object(upload_path, ogg_buffer.getvalue())
        if upload_result.status != 200:
            raise ValueError("Upload failed")

    tim_listener: TranscriptionTimListener | None = None
    tim_manager = None
    request_id = f"{user_id}_{version}_{language}_{song_hash}"
    fallback_language = "zh" if language == "ch" else language
    fallback_url = (
        f"https://as-api.oss-cn-qingdao.aliyuncs.com/app/user/vt/file/"
        f"{fallback_language}/{song_hash}.ace"
    )
    if TIMManager is not None:
        try:
            im_sign_resp = await client.get(
                urljoin(ACE_API_BASE_URL, "/api/as/im/sign")
            )
            im_sign_data = im_sign_resp.json().get("data", {})
            sdk_appid = int(im_sign_data["app_id"])
            user_sig = im_sign_data["sign"]
            tim_listener = TranscriptionTimListener(
                request_id=request_id,
                song_hash=song_hash,
                fallback_url=fallback_url,
                download_dir=song_path.parent,
                client=client,
                loop=asyncio.get_running_loop(),
            )
            tim_manager = TIMManager(
                sdk_appid=sdk_appid,
                user_id=str(user_id),
                user_sig=user_sig,
                metadata={"request_id": request_id, "song_hash": song_hash},
                message_handler=tim_listener.handle_messages,
                login_handler=tim_listener.handle_login,
                log_handler=tim_listener.handle_log,
            )
            tim_manager.start()
        except Exception:
            logger.debug("TIM integration is unavailable for this transcription request")

    try:
        extra_kwargs = {
            "format": "json"
        } if version == "2.0" else {}
        resp = await client.post(
            vt_api_url,
            json={
                "app": "studio",
                "audio": f"https://as-api.oss-accelerate.aliyuncs.com/{upload_prefix}/{song_hash}.ogg",
                "request_id": request_id,
                "to_lan": language,
                "version": version,
                **extra_kwargs,
            },
            headers={
                "token": vt_api_token,
            },
        )
        resp_data = resp.json()
        if resp_data["data"].get("file_url"):
            logger.info("Transcription success")
            await (tim_listener.download_result(resp_data["data"]["file_url"]) if tim_listener is not None else _download_file(client, resp_data["data"]["file_url"], song_path.parent / f"{song_hash}.ace"))
        else:
            logger.info("Please wait for the transcription result")
            if tim_listener is not None:
                if await anyio.to_thread.run_sync(tim_listener.result_event.wait, 180):
                    if tim_listener.download_future is not None:
                        await anyio.to_thread.run_sync(
                            tim_listener.download_future.result, 180
                        )
                    elif tim_listener.result_url and tim_listener.downloaded_path is None:
                        await tim_listener.download_result(tim_listener.result_url)
                else:
                    logger.info(tim_listener.fallback_url)
            else:
                logger.info(fallback_url)
    finally:
        if tim_manager is not None:
            tim_manager.stop()
            tim_manager.join(timeout=15)
            if tim_manager.is_alive():
                logger.warning("TIM manager did not stop cleanly before process exit")


async def _download_file(
    client: httpx2.AsyncClient, url: str, target_path: pathlib.Path
) -> pathlib.Path:
    response = await client.get(url)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    logger.info(f"Downloaded transcription to {target_path}")
    return target_path


@app.command()
def transcribe(
    song_path: pathlib.Path,
    language: Literal["ch", "en", "jp", "es", "ita", "fra", "por", "kor", "note", "unknown"] = "ch",
    version: str = "1.0",
) -> None:
    anyio.run(
        transcribe_asynchronous,
        song_path,
        language,
        version,
    )
