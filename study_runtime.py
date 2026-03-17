"""Console formatting and personal-path redaction for study runners."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def redact_personal_text(text: str) -> str:
    s = re.sub(r"/Users/[^/\s]+", "/Users/$USER", str(text))
    s = re.sub(r"/home/[^/\s]+", "/home/$USER", s)
    home = str(Path.home())
    return s.replace(home, "~") if home != "/" else s


def redact_personal_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_personal_text(value)
    if isinstance(value, (list, tuple)):
        return [redact_personal_obj(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_personal_obj(item) for key, item in value.items()}
    return value


def safe_name(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    cleaned = "".join(out).strip("-")
    return cleaned or "exp"


def duration_parts(sec: float) -> tuple[int, int, int]:
    total = int(round(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return h, m, s


def format_duration(sec: float) -> str:
    h, m, s = duration_parts(sec)
    if h == 0 and m == 0:
        return f"{float(sec):.1f}s"

    parts: list[str] = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    if s > 0:
        parts.append(f"{s:02d}s")
    if not parts:
        return "0s"
    return "".join(parts)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {redact_personal_text(message)}")
