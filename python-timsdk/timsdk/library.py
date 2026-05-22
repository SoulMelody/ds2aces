import contextlib
import ctypes
import os
import platform


module_dir = os.path.dirname(__file__)
operating_system = platform.system()
cpu_bits, _ = platform.architecture()

if operating_system == "Windows":
    func_type = ctypes.WINFUNCTYPE
else:
    func_type = ctypes.CFUNCTYPE


tim_callback = func_type(
    None,
    ctypes.c_int32,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
)

log_callback = func_type(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
recv_msg_callback = func_type(None, ctypes.c_char_p, ctypes.c_void_p)


@contextlib.contextmanager
def tim_factory():
    tim_lib = None
    if operating_system == "Windows":
        if cpu_bits == "64bit":
            tim_lib = ctypes.WinDLL(
                os.path.join(module_dir, "lib/windows/lib/Win64/ImSDK.dll")
            )
        elif cpu_bits == "32bit":
            tim_lib = ctypes.WinDLL(
                os.path.join(module_dir, "lib/windows/lib/Win32/ImSDK.dll")
            )
    elif operating_system == "Linux":
        tim_lib = ctypes.CDLL(os.path.join(module_dir, "lib/linux/lib/libImSDK.so"))
    elif operating_system in {"MacOS", "Darwin"}:
        tim_lib = ctypes.CDLL(
            os.path.join(module_dir, "lib/mac/Versions/A/ImSDKForMac.dylib")
        )
    yield tim_lib
    # Intentionally do not unload the native library explicitly.
    # The IMSDK may keep internal worker threads and callbacks alive beyond the
    # Python thread lifetime; forcing FreeLibrary/dlclose here can crash the
    # process during shutdown. Let the OS reclaim the library on process exit.
