import json

from loguru import logger

from .enums import LogLevel
from .library import log_callback, tim_callback
from .utils import ptr2str


@log_callback
def tim_log(level, log, user_data):
    log_msg = ptr2str(log)
    if level == LogLevel.DEBUG:
        logger.debug(log_msg)
    elif level == LogLevel.INFO:
        logger.info(log_msg)
    elif level == LogLevel.WARNING:
        logger.warning(log_msg)
    elif level == LogLevel.ERROR:
        logger.error(log_msg)


@tim_callback
def login_callback(code, desc, json_param, user_data):
    logger.debug(code)
    logger.debug(ptr2str(desc))


@tim_callback
def list_msgs(code, desc, json_param, user_data):
    logger.debug(code)
    logger.debug(ptr2str(desc))
    logger.debug(ptr2str(json_param))


def on_recv_new_msg(json_msg_array, manager):
    msgs = json.loads(ptr2str(json_msg_array))
    logger.debug(msgs)
    if hasattr(manager, "_emit_messages"):
        manager._emit_messages(msgs)
