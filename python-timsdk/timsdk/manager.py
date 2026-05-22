import json
import os
import threading
import time
import contextlib
from collections.abc import Callable
from typing import Any

from loguru import logger

from .enums import LogLevel, LoginStatus
from .library import log_callback, module_dir, recv_msg_callback, tim_callback, tim_factory
from .utils import ptr2str, str2ptr

MessageHandler = Callable[[list[dict[str, Any]], dict[str, Any]], None]
LoginHandler = Callable[[int, str, str, dict[str, Any]], None]
LogHandler = Callable[[int, str, dict[str, Any]], None]


class TIMManager(threading.Thread):
    @property
    def sdk_config(self):
        return {
            "sdk_config_log_file_path": os.path.join(
                module_dir, "com_tencent_imsdk_log"
            ),
            "sdk_config_config_file_path": os.path.join(
                module_dir, "com_tencent_imsdk_data"
            ),
        }

    def __init__(
        self,
        sdk_appid: int,
        user_id: str,
        user_sig: str,
        metadata: dict,
        message_handler: MessageHandler | None = None,
        login_handler: LoginHandler | None = None,
        log_handler: LogHandler | None = None,
    ):
        super().__init__(daemon=True)
        self.sdk_appid = sdk_appid
        self.user_id = user_id
        self.user_sig = user_sig
        self.metadata = metadata
        self.message_handler = message_handler
        self.login_handler = login_handler
        self.log_handler = log_handler
        self.event = threading.Event()
        self.logged_in_event = threading.Event()
        self.login_failed_event = threading.Event()
        self.last_login_code: int | None = None
        self.last_login_desc = ""
        logger.add(
            os.path.join(module_dir, "com_tencent_imsdk_log/timsdk.log"), level="DEBUG"
        )

    def stop(self):
        self.event.set()

    def stopped(self):
        return self.event.is_set()

    def _emit_log(self, level: int, message: str) -> None:
        if level == LogLevel.DEBUG:
            logger.debug(message)
        elif level == LogLevel.INFO:
            logger.info(message)
        elif level == LogLevel.WARNING:
            logger.warning(message)
        elif level == LogLevel.ERROR:
            logger.error(message)
        else:
            logger.debug(message)
        if self.log_handler is not None:
            self.log_handler(level, message, self.metadata)

    def _emit_login(self, code: int, desc: str, json_param: str) -> None:
        self.last_login_code = code
        self.last_login_desc = desc
        logger.debug(f"TIM login callback code={code} desc={desc}")
        if json_param:
            logger.debug(f"TIM login callback payload={json_param}")
        if self.login_handler is not None:
            self.login_handler(code, desc, json_param, self.metadata)

    def _emit_messages(self, msgs: list[dict[str, Any]]) -> None:
        logger.debug(msgs)
        if self.message_handler is not None:
            self.message_handler(msgs, self.metadata)

    def run(self):
        @log_callback
        def _tim_log(level, log, _):
            self._emit_log(level, ptr2str(log))

        @tim_callback
        def _login_callback(code, desc, json_param, _):
            self._emit_login(code, ptr2str(desc), ptr2str(json_param))

        @recv_msg_callback
        def _on_recv_new_msg(json_msg_array, _):
            self._emit_messages(json.loads(ptr2str(json_msg_array)))

        try:
            with tim_factory() as tim_lib:
                tim_lib.TIMSetLogCallback(_tim_log, str2ptr(""))
                tim_lib.TIMInit(self.sdk_appid, str2ptr(json.dumps(self.sdk_config)))
                tim_lib.TIMAddRecvNewMsgCallback(_on_recv_new_msg, str2ptr(""))
                tim_lib.TIMLogin(
                    str2ptr(self.user_id),
                    str2ptr(self.user_sig),
                    _login_callback,
                    str2ptr(""),
                )
                logged_in = LoginStatus.LOGGED_OUT
                for _ in range(10):
                    logged_in = tim_lib.TIMGetLoginStatus()
                    if logged_in == LoginStatus.LOGGED_IN:
                        self.logged_in_event.set()
                        break
                    time.sleep(1)
                else:
                    if logged_in != LoginStatus.LOGGED_IN:
                        self.login_failed_event.set()
                        logger.warning("login failed!")
                        return
                while not self.stopped():
                    time.sleep(0.1)
                with contextlib.suppress(Exception):
                    getattr(tim_lib, "TIMLogout")(str2ptr(self.user_id))
        except Exception:
            self.login_failed_event.set()
            logger.exception("TIM manager crashed")
