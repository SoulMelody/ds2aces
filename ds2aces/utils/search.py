from collections.abc import Callable
from typing import TypeVar

from more_itertools import locate, rlocate

T = TypeVar("T")


def find_index(obj_list: list[T], pred: Callable[[T], bool]) -> int:
    return next(locate(obj_list, pred), -1)


def find_last_index(obj_list: list[T], pred: Callable[[T], bool]) -> int:
    return next(rlocate(obj_list, pred), -1)
