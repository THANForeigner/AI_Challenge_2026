"""
Chỉnh stdout/stderr sang UTF-8 một cách an toàn (chủ yếu cho Windows).

Gọi configure_stdio() ở đầu mỗi script. Hàm này không bao giờ ném
lỗi: stream không hỗ trợ reconfigure (IDE, stream bị wrap, None...)
thì lặng lẽ bỏ qua.
"""

import sys


def _reconfigure(stream):
    if stream is None:
        return

    reconfigure = getattr(stream, "reconfigure", None)

    if reconfigure is None:
        return

    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        # io.UnsupportedOperation, stream đã detach, buffer đặc biệt...
        pass


def configure_stdio():
    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)
