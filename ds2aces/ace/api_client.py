import json
import platform

from ds2aces import __VERSION__
from ds2aces.ace.config import get_ace_app_language, read_ace_user_info_config


def default_headers(token="-", os_version="11"):
    user_info_config = read_ace_user_info_config()
    system_full_name = platform.system().lower()
    if system_full_name == "windows":
        system_name = "win"
    elif system_full_name == "darwin":
        system_name = "mac"
    else:
        # system_name = "linux"
        system_full_name = "windows"  # hack for linux
        system_name = "win"
    region = "CN"
    if user_info_key := user_info_config.get("user_info_group", "user_info_key", fallback=None):
        region = json.loads(json.loads(user_info_key)).get("region", None) or region
    return {
        "User-Agent": f"{system_name};{__VERSION__};{system_full_name}-{os_version};",
        "channel": "GENERAL",
        "device": f"{system_full_name}-{os_version}",
        "platform": system_name,
        "region": region,
        "version": __VERSION__,
        "token": token,
        "lan": get_ace_app_language(),
    }