import ctypes


def ptr2str(cstr: bytes) -> str:
    if not isinstance(cstr, str):
        text = cstr.decode()
    else:
        text = cstr
    return text


def str2ptr(text: str) -> ctypes.Array:
    if isinstance(text, str):
        encoded = text.encode()
    else:
        encoded = text
    return ctypes.create_string_buffer(encoded)
