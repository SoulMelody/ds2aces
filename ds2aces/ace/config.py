import configparser
import pathlib

ACE_CONFIG_ROOT = pathlib.Path("~").expanduser() / "ACE_Studio/"

ace_studio_lock_file = ACE_CONFIG_ROOT / "ace-studio.lock"
ace_login_config_path = ACE_CONFIG_ROOT / "UserDefaults/login_ini"
ace_studio_config_path = ACE_CONFIG_ROOT / "user/config.ini"
ace_user_info_config_path = ACE_CONFIG_ROOT / "UserDefaults/user_info_ini"


def read_ace_login_config() -> configparser.ConfigParser:
    config_parser = configparser.ConfigParser()
    config_parser.read(ace_login_config_path, "utf-8")
    if not config_parser.has_section("login_record_group"):
        config_parser.add_section("login_record_group")
    if not config_parser.has_section("login_token_group"):
        config_parser.add_section("login_token_group")
    return config_parser


def read_ace_user_info_config() -> configparser.ConfigParser:
    config_parser = configparser.ConfigParser()
    config_parser.read(ace_user_info_config_path, "utf-8")
    if not config_parser.has_section("user_info_group"):
        config_parser.add_section("user_info_group")
    return config_parser


def get_ace_app_language() -> str:
    config_parser = configparser.ConfigParser()
    config_parser.read(ace_studio_config_path, "utf-8")
    if not config_parser.has_section("Common"):
        config_parser.add_section("Common")
    lang_value = config_parser["Common"].getint("Lan", 0)
    if lang_value == 1:
        return "ENG"
    elif lang_value == 2:
        return "JPN"
    return "CHN"


def write_ace_login_config(config_parser: configparser.ConfigParser):
    with ace_login_config_path.open("w", encoding="utf-8") as configfile:
        config_parser.write(configfile)


def write_ace_user_info_config(config_parser: configparser.ConfigParser):
    with ace_user_info_config_path.open("w", encoding="utf-8") as configfile:
        config_parser.write(configfile)


ACE_API_BASE_URL = "https://as-api.tdacestudio.com/"
