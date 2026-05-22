import enum


class LoginStatus(enum.IntEnum):
    LOGGED_IN = 1
    LOGGING_IN = 2
    LOGGED_OUT = 3


class LogLevel(enum.IntEnum):
    NOTSET = 0
    DEBUG = 3
    INFO = 4
    WARNING = 5
    ERROR = 6


class ElementType(enum.IntEnum):
    TEXT = 1
    IMAGE = 2
    SOUND = 3
    CUSTOM = 4
    FILE = 5
    GROUP_TIPS = 6
    FACE = 7
    LOCATION = 8
    GROUP_REPORT = 9
    VIDEO = 10
    FRIEND_CHANGE = 11
    PROFILE_CHANGE = 12
    MERGE = 13
    INVALID = 14
