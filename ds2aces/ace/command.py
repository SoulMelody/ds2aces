from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import math
import os
import pathlib
import re
import time
import uuid
from urllib.parse import urljoin
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, TypeAlias

import asyncio_oss
import anyio
import httpx
import more_itertools
import oss2
import pypinyin
import soundfile as sf
import tenacity
import typer
import wanakana
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
from ds2aces.ace.constants import DEFAULT_SEED, MIN_NOTE_DURATION, MIN_SILENCE_DURATION, MAX_PIECE_DURATION
from ds2aces.ace.model import (
    AceParam,
    AceEngineBody,
    AcesPieceParams,
    AcesSimpleNote,
    AcesProject,
    compress_ace_segment,
)
from ds2aces.ds.ds_file import DsProject
from ds2aces.ds.phoneme_dict import get_ds_phone_dict, get_vowels_set
from ds2aces.utils.music_math import hz2midi, note2midi
from ds2aces.utils.search import find_last_index

if TYPE_CHECKING:
    import numpy as np

app = typer.Typer()

CENTS_RE = re.compile(r"[+-]\d+$")
AcesItem: TypeAlias = dict[str, Any]
AcesNoteItem: TypeAlias = dict[str, Any]

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

    def simple_cut(list_to_cut: list[AcesNoteItem]) -> list[list[AcesNoteItem]]:
        middle_time = (float(list_to_cut[0]["start_time"]) + float(list_to_cut[-1]["end_time"]))/2

        cut_index = 0
        cut_index = min(
            range(len(list_to_cut)),
            key=lambda x: abs(float(list_to_cut[x]["start_time"]) - middle_time)
        )
        if list_to_cut[cut_index]["type"] == "slur":
            cut_index = find_last_index(list_to_cut[:cut_index-1], lambda x: x["type"] != "slur")

        half_1 = list_to_cut[:cut_index]
        half_2 = list_to_cut[cut_index:]

        return [half_1, half_2]

    def fine_cut(list_to_cut: list[AcesNoteItem], max_length: float) -> list[list[AcesNoteItem]]:
        list_end = float(list_to_cut[-1].get("end_time"))
        list_start = float(list_to_cut[0].get("start_time"))
        list_length = list_end - list_start
        if list_length <= max_length:
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


async def render_aces(client: httpx.AsyncClient, aces_file: pathlib.Path, router_id: int) -> None:
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


async def fetch_router_config(client: httpx.AsyncClient, echo: bool = False) -> dict[int, str]:
    router_response = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/voice/seed/v3",
        ),
    )
    if not router_response.is_error:
        resp_data = router_response.json()
        if resp_data["code"] == 200:
            code2router_url = {
                router["id"]: router["router"]
                for router in resp_data["data"]["router_list"]
                if router["is_show"]
            }
            if echo:
                console = Console(color_system="256")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("ID", justify="left", style="cyan")
                table.add_column("Router Name", justify="left", style="cyan")
                table.add_column("URL", justify="left", style="cyan")
                table.add_column("Supported Languages", justify="left", style="cyan")
                for router in resp_data["data"]["router_list"]:
                    table.add_row(
                        str(router["id"]),
                        router["router_name"],
                        router["router"],
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
async def send_request(client: httpx.AsyncClient, compose_body: AceEngineBody, files: dict[str, tuple[str, BinaryIO, str]], router_url: str) -> httpx.Response:
    return await client.post(
        router_url,
        headers={
            "MIME-Version": "1.0"
        },
        data=compose_body.model_dump(mode="json"),
        files=files,
    )


async def download_and_open_audio(client: httpx.AsyncClient, url: str) -> tuple[np.ndarray, int]:
    response = await client.get(url)
    response.raise_for_status()

    with io.BytesIO(response.content) as file:
        data, samplerate = sf.read(file)
        return data, samplerate

async def one_piece_compose(client: httpx.AsyncClient, ace_token: str, aces: dict, router_url: str) -> tuple[float, np.ndarray, int]:
    user_config = read_ace_user_info_config()
    user_id = json.loads(
        json.loads(user_config.get("user_info_group", "user_info_key"))
    )["uid"]
    timestamp = int(time.time() * 1000)
    upload_path = f"{user_id}_{timestamp}.aces"

    compose_body = AceEngineBody(
        compress_type="zstd",
        flag=".aces",
        ace_token=ace_token,
        pipeline_business=2,
    )
    # logger.debug(aces)
    files = {
        "file": (
            upload_path,
            io.BytesIO(compress_ace_segment(json.dumps(aces))),
            "application/octet-stream",
        )
    }
    compose_resp = await send_request(
        client, compose_body, files, router_url
    )
    logger.debug(compose_resp.json())
    result = compose_resp.json()["data"][0]
    audio_url = result.get('audio')
    audio_data, samplerate = await download_and_open_audio(client, audio_url)
    pst = result.get('pst')

    return pst, audio_data, samplerate

async def rendering_ace_list(client: httpx.AsyncClient, ace_list: list[AcesItem], save_to_path: pathlib.Path, as_token: str, router_id: int) -> None:
    vocal_offset = None
    concat_audio = []

    code2router_url = await fetch_router_config(client)
    if not code2router_url:
        return
    router_url = code2router_url[router_id]

    if not ace_list:
        return
    for aces in track(ace_list, description="Rendering ACE Sequence files..."):
        pst, audio_data, samplerate = await one_piece_compose(client, as_token, aces, router_url)

        if vocal_offset is None:
            vocal_offset = pst
            concat_audio.extend(audio_data)
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


async def download_phoneme_data(client: httpx.AsyncClient) -> dict[str, Any]:
    config_resp = await client.get(
        urljoin(ACE_API_BASE_URL, "/api/as/conf/client"),
    )
    config_resp.raise_for_status()
    config_data = config_resp.json()
    multi_lan_plan_url = config_data["data"]["multi_lan_plan"]["url"]
    multi_lan_plan_path = ACE_CONFIG_ROOT / "ClientSetup" / os.path.basename(multi_lan_plan_url)
    if not multi_lan_plan_path.exists():
        with multi_lan_plan_path.open("wb") as f:
            async with client.stream("GET", multi_lan_plan_url) as stream:
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
    client = httpx.Client(
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



async def pre_login() -> httpx.AsyncClient | None:
    config_parser = read_ace_login_config()
    token = json.loads(
        json.loads(config_parser["login_token_group"]["login_token_key"])
    )["token"]
    client = httpx.AsyncClient(
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


async def fetch_as_token(client: httpx.AsyncClient) -> str:
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
    client = httpx.Client(follow_redirects=True, timeout=10, headers=default_headers())
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


async def fetch_singers(client: httpx.AsyncClient | None = None) -> None:
    if client is None:
        client = await pre_login()
    if client is None:
        return
    response = await client.get(
        urljoin(
            ACE_API_BASE_URL,
            "/api/as/singer/list/v2",
        ),
    )
    if not response.is_error:
        singers_data = response.json()
        if singers_data["code"] == 200:
            console = Console(color_system="256")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Seed", justify="left", style="cyan")
            table.add_column("Singer Name", justify="left", style="cyan")
            for singer_data in singers_data["data"]["official"]["singers"]:
                table.add_row(
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


def g2p(lyric: str, language: Literal["ch", "en", "jp", "spa"] = "ch") -> str:
    if language == "ch":
        return " ".join(pypinyin.lazy_pinyin(lyric, style=pypinyin.Style.NORMAL))
    elif language == "jp":
        return wanakana.to_romaji(lyric, custom_romaji_mapping={"っ": "cl"})
    else:
        raise NotImplementedError(f"Language {language} is not supported yet")


async def ds_to_aces(in_path: pathlib.Path, output_dir: pathlib.Path, param: bool = False, render: bool = False, language: Literal["ch", "en", "jp", "spa"] = "ch") -> None:
    client = await pre_login()
    if not client:
        logger.error("login failed")
        return
    ACE_PHONE_DICT = await download_phoneme_data(client)
    if language == "ch":
        ds_phone_dict = get_ds_phone_dict("opencpop-extension", g2p=False)
        vowels_set = get_vowels_set("opencpop-extension")
        ace_phone_dict = ACE_PHONE_DICT["plans"][0]
    elif language == "jp":
        ds_phone_dict = get_ds_phone_dict("japanese_dict_full", g2p=False)
        vowels_set = get_vowels_set("japanese_dict_full")
        ace_phone_dict = ACE_PHONE_DICT["plans"][1]
    else:
        raise NotImplementedError(f"Language {language} is not supported yet")
    ds_project = DsProject.model_validate_json(in_path.read_text(encoding="utf-8"))
    notes = []
    await fetch_singers(client)
    seed_id = typer.prompt("Please input the seed id", default=DEFAULT_SEED)
    if render:
        mix_info_resp = await client.post(
            urljoin(
                ACE_API_BASE_URL,
                "/api/as/voice/mix/str/v2",
            ),
            json={"seeds": [
                {"s": 1, "t": 1, "s_id": int(seed_id)},
            ]},
        )
        if not mix_info_resp.is_error:
            mix_info_data = json.loads(mix_info_resp.text)
            if mix_info_data["code"] == 200:
                mix_info = mix_info_data["data"][
                    "mix_info"
                ]
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
                silence_time = (cur_time + ds_item.ph_dur[ph_index]) if ds_item.ph_dur is not None else next_time
                for pitch_param in pitch_params:
                    pitch_param.silent(cur_time, silence_time)
                ph_index += 1
            elif text == "AP":
                next_time = cur_time + sum(
                    ds_item.note_dur[note_index]
                    for note_index, is_slur in slur_group
                )
                silence_time = (cur_time + ds_item.ph_dur[ph_index]) if ds_item.ph_dur is not None else next_time
                notes.append(
                    AcesSimpleNote(
                        start_time=cur_time,
                        end_time=next_time,
                        pitch=midi_key,
                        type="br",
                    )
                )
                for pitch_param in pitch_params:
                    pitch_param.silent(cur_time, silence_time)
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
                            if pronunciation in ace_phone_dict["syllable_alias"]:
                                pronunciation = ace_phone_dict["syllable_alias"][pronunciation]
                            if language != "ch" and pronunciation in ace_phone_dict["dict"]:
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
                                for pitch_param in pitch_params:
                                    pitch_param.silent(silence_time_start, cur_time)
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
                    if pronunciation in ace_phone_dict["syllable_alias"]:
                        pronunciation = ace_phone_dict["syllable_alias"][pronunciation]
                    if language != "ch" and pronunciation in ace_phone_dict["dict"]:
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

@app.command()
def ds2aces(
    in_path: pathlib.Path = typer.Argument("input.ds", exists=True, dir_okay=False),
    output_dir: pathlib.Path = typer.Argument(
        ".", exists=False, dir_okay=True, file_okay=False
    ),
    param: bool = typer.Option(True),
    render: bool = typer.Option(False),
    language: Literal["ch", "en", "jp", "spa"] = "ch",
) -> None:
    if language not in ["ch", "jp"]: # TODO: support other languages
        raise NotImplementedError("Only Chinese and Japanese are supported")
    anyio.run(ds_to_aces, in_path, output_dir, param, render, language)


async def transcribe_asynchronous(
    song_path: pathlib.Path,
    language: Literal["ch", "en", "jp"] = "ch",
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

    resp = await client.post(
        vt_api_url,
        json={
            "app": "studio",
            "audio": f"https://as-api.oss-accelerate.aliyuncs.com/{upload_prefix}/{song_hash}.ogg",
            "request_id": f"{user_id}_{version}_{language}_{song_hash}",
            "to_lan": language,
            "version": version,
        },
        headers={
            "token": vt_api_token,
        },
    )
    resp_data = resp.json()
    if resp_data["data"].get("file_url"):
        logger.info("Transcription success")
        logger.info(resp_data["data"]["file_url"])
    else:
        logger.info("Please wait for the transcription result")
        if language == "ch":
            language = "zh"
        logger.info(
            f"https://as-api.oss-cn-qingdao.aliyuncs.com/app/user/vt/file/{language}/{song_hash}.ace"
        )


@app.command()
def transcribe(
    song_path: pathlib.Path,
    language: Literal["ch", "en", "jp"] = "ch",
    version: str = "1.0",
) -> None:
    anyio.run(
        transcribe_asynchronous,
        song_path,
        language,
        version,
    )
