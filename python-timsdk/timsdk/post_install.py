from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
NODE_MODULE_SDK_ROOT = PROJECT_ROOT / "node_modules" / "im_electron_sdk"
TARGET_LIB_ROOT = PACKAGE_ROOT / "lib"

PLATFORM_LIB_DIR = {
    "Windows": "windows",
    "Linux": "linux",
    "Darwin": "mac",
}


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _ensure_nodeenv() -> None:
    if NODE_MODULE_SDK_ROOT.exists():
        return
    _run(["nodeenv", "-p"])


def _ensure_sdk_package() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "npm was not found in PATH. Install Node.js or activate the nodeenv environment first."
        )
    _run(
        [
            npm,
            "install",
            "im_electron_sdk",
            "--registry=https://registry.npmmirror.com",
            "--no-save",
            "--ignore-scripts",
        ]
    )


def _resolve_platform_source() -> tuple[str, Path]:
    system = platform.system()
    platform_dir = PLATFORM_LIB_DIR.get(system)
    if platform_dir is None:
        supported = ", ".join(sorted(PLATFORM_LIB_DIR))
        raise RuntimeError(f"Unsupported platform: {system}. Supported platforms: {supported}")

    source = NODE_MODULE_SDK_ROOT / "lib" / platform_dir
    if not source.exists():
        raise FileNotFoundError(f"Expected SDK library directory does not exist: {source}")
    return platform_dir, source


def _copy_sdk(platform_dir: str, source: Path) -> None:
    destination = TARGET_LIB_ROOT / platform_dir
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _write_manifest(platform_dir: str) -> None:
    manifest = {
        "platform": platform.system(),
        "copied_dir": platform_dir,
    }
    (TARGET_LIB_ROOT / ".setup.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + os.linesep,
        encoding="utf-8",
    )


def main() -> None:
    _ensure_nodeenv()
    _ensure_sdk_package()
    platform_dir, source = _resolve_platform_source()
    _copy_sdk(platform_dir, source)
    _write_manifest(platform_dir)


if __name__ == "__main__":
    main()
