# Python-TIMSDK

# Build Instructions

1. Prerequisites: install `uv`, `node`, and ensure `npm` is available in `PATH`.
2. From the workspace root, run `uv sync --all-packages --dev`.
3. Run `uv run post-setup-workspace`.

# Workspace Integration

`python-timsdk` is configured as a `uv` workspace member.

- Root project dependency resolution uses the workspace package source automatically.
- The old `setup_imsdk_lib.sh` flow has been replaced by `uv run post-setup`, `uv run timsdk-post-setup`, or `uv run post-setup-workspace`.
- If your editor or workspace manager supports a `post_setup` hook, point it at `uv run post-setup-workspace`.
- The setup implementation now lives in `timsdk/post_install.py`; it is still invoked explicitly via the commands above.

# Platform Support

`timsdk` only supports Windows, Linux, and macOS.
