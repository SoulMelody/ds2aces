from __future__ import annotations

import base64
import contextlib
import importlib
import platform
from typing import Literal

from pydantic import BaseModel, RootModel, Field

from ds2aces import __VERSION__
from ds2aces.ace.constants import MIN_NOTE_DURATION
from ds2aces.utils.search import find_index, find_last_index


for zstd_backend in (
    "compression.zstd",
    "backports.zstd",
    "zstd",
    "pyzstd",
    "zstandard",
    "cramjam",
    "numcodecs.zstd",
):
    with contextlib.suppress(ImportError):
        zstd = importlib.import_module(zstd_backend)
        if zstd_backend == "cramjam":
            zstd = zstd.zstd
        break


def default_os_version():
    return platform.win32_ver()[1] if platform.system() == "Windows" else "10.0.22000"



class AceDebugInfo(BaseModel):
    version: str = __VERSION__
    template_id: str | None = Field(None, alias="templateId")
    record_type: str = Field("create", alias="recordType")
    platform: str = "pc"
    os: str = Field(default_factory=default_os_version)
    device: str = "Windows(x86_64)_Administrator"
    build: str = "1400"


class AceGlobalInfo(BaseModel):
    time_scale_factor: float = 1.0
    type: str = ""


class AceSimpleSegment(BaseModel):
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self):
        return self.end_time - self.start_time


class AceEngineBody(BaseModel):
    compress_type: str | None = None
    ace_token: str
    pipeline_business: int
    flag: str


class AceParam(BaseModel):
    hop_time: float = MIN_NOTE_DURATION
    start_time: float = 0
    values: list[float] = Field(default_factory=list)

    def silent(self, start_time: float, end_time: float) -> None:
        time_list = [
            i * self.hop_time + self.start_time for i in range(len(self.values))
        ]
        start_index = find_last_index(time_list, lambda x: x < start_time)
        end_index = find_index(time_list, lambda x: x >= end_time)
        if start_index != -1 and end_index != -1:
            self.values[start_index + 1:end_index] = [0] * (end_index - start_index - 1)
        elif start_index != -1:
            self.values = self.values[:start_index + 1]
        elif end_index != -1:
            self.values[:end_index] = [0] * (end_index - 1)


class AcesSimpleNote(AceSimpleSegment):
    type_: Literal["general", "br", "sp", "slur"] = Field("general", alias="type")
    pitch: int = 0
    phone: list[str] = Field(default_factory=list)
    syllable: str | None = ""
    language: Literal["ch", "en", "jp", "spa"] = "ch"
    consonant_time_head: list[float] | None = None
    consonant_time_tail: list[float] | None = None


class AcesPadNotes(BaseModel):
    begin: AcesSimpleNote
    end: AcesSimpleNote

class AcesPieceParam(BaseModel):
    delta: list[AceParam] = Field(default_factory=list)
    envelope: list[AceParam] = Field(default_factory=list)
    user: list[AceParam] = Field(default_factory=list)

class AcesPieceParams(BaseModel):
    air: AcesPieceParam = Field(default_factory=AcesPieceParam, title="气声")
    energy: AcesPieceParam = Field(default_factory=AcesPieceParam, title="力度")
    falsetto: AcesPieceParam = Field(default_factory=AcesPieceParam, title="假声")
    gender: AcesPieceParam = Field(default_factory=AcesPieceParam, title="性别")
    pitch: AcesPieceParam = Field(default_factory=AcesPieceParam, title="音高")
    tension: AcesPieceParam = Field(default_factory=AcesPieceParam, title="张力")


class AcesProject(BaseModel):
    debug_info: AceDebugInfo = Field(default_factory=AceDebugInfo)
    global_info: AceGlobalInfo = Field(default_factory=AceGlobalInfo)
    notes: list[AcesSimpleNote] = Field(default_factory=list)
    piece_params: AcesPieceParams = Field(default_factory=AcesPieceParams)
    pad: AcesPadNotes | None = None
    version: float = 1.1
    mix_info: str


class AcesList(RootModel[list[AcesProject]]):
    pass


def decompress_ace_segment(src: bytes, raw: bool = False) -> str:
    raw_content = base64.b64decode(src.strip().strip(b'"'))
    decompressed = zstd.decompress(raw_content)
    return decompressed.decode("utf-8")


def compress_ace_segment(src: str) -> bytes:
    raw_content = src.encode("utf-8")
    compressed = zstd.compress(raw_content)
    return base64.b64encode(compressed)
