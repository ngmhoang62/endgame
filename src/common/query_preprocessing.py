"""Deterministic query text handling for ENDGAME.

Important separation:
- model_text(): conservative normalization used for model/retriever input.
- *_match_key(): progressively stronger transforms used ONLY for diagnostics.

Never collapse, overwrite, or train on the stronger diagnostic keys.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def model_text(text: Any) -> str:
    """NFC + whitespace compaction only. Preserve case, accents and punctuation."""
    value = unicodedata.normalize("NFC", str(text or ""))
    return _WS.sub(" ", value).strip()


def strict_match_key(text: Any) -> str:
    """Case-insensitive exact-content key; still preserves accents/punctuation."""
    return model_text(text).casefold()


def punctuation_match_key(text: Any) -> str:
    """Diagnostic-only key: punctuation/symbols -> spaces, accents preserved."""
    text = unicodedata.normalize("NFKC", model_text(text)).casefold()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        chars.append(" " if cat.startswith(("P", "S")) else ch)
    return _WS.sub(" ", "".join(chars)).strip()


def accent_fold(text: Any) -> str:
    """Diagnostic-only Vietnamese accent folding."""
    text = punctuation_match_key(text).replace("đ", "d")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def accent_match_key(text: Any) -> str:
    return _WS.sub(" ", accent_fold(text)).strip()


def word_count(text: Any) -> int:
    return len(_WORD.findall(model_text(text)))


def suspicious_unicode(text: Any) -> dict[str, bool]:
    raw = str(text or "")
    return {
        "replacement_char": "\ufffd" in raw,
        "nul": "\x00" in raw,
        "control_char": any(
            unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in raw
        ),
        "non_nfc": raw != unicodedata.normalize("NFC", raw),
        "leading_or_trailing_ws": raw != raw.strip(),
        "multi_whitespace": bool(re.search(r"\s{2,}", raw)),
        "html_like": bool(re.search(r"<[^>]{1,120}>", raw)),
    }
