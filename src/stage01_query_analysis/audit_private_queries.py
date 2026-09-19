#!/usr/bin/env python
"""Stage 01A: label-free audit of DSC2026 private queries.

Default:
    python src/stage01_query_analysis/audit_private_queries.py

Input defaults to ./private-official.json.
Outputs are small, Git-friendly artifacts under reports/stage01_private_query_audit/.

This script intentionally uses only the Python standard library so it can run
immediately after cloning, before model downloads or GPU setup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "private-official.json"
DEFAULT_OUT = ROOT / "reports" / "stage01_private_query_audit"

QID_KEYS = ("qid", "query_id", "id", "question_id")
TEXT_KEYS = ("question", "query", "text", "content")

RE_WS = re.compile(r"\s+")
RE_WORD = re.compile(r"\w+", re.UNICODE)
RE_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*%|\s*(?:đồng|triệu|tỷ|ngày|tháng|năm|tuổi|km|m2|m²))?", re.I)

LEGAL_PATTERNS = {
    "dieu": re.compile(r"\bđiều\s+\d+[a-zA-Z]?\b", re.I),
    "khoan": re.compile(r"\bkhoản\s+\d+[a-zA-Z]?\b", re.I),
    "diem": re.compile(r"\bđiểm\s+[a-zA-Zđ]\b", re.I),
    "chuong": re.compile(r"\bchương\s+[IVXLCDM\d]+\b", re.I),
    "muc": re.compile(r"\bmục\s+\d+\b", re.I),
}

INSTRUMENT_PATTERNS = {
    "luat": re.compile(r"\bluật\b", re.I),
    "bo_luat": re.compile(r"\bbộ\s+luật\b", re.I),
    "nghi_dinh": re.compile(r"\bnghị\s+định\b", re.I),
    "thong_tu": re.compile(r"\bthông\s+tư\b", re.I),
    "quyet_dinh": re.compile(r"\bquyết\s+định\b", re.I),
    "nghi_quyet": re.compile(r"\bnghị\s+quyết\b", re.I),
    "phap_lenh": re.compile(r"\bpháp\s+lệnh\b", re.I),
    "hien_phap": re.compile(r"\bhiến\s+pháp\b", re.I),
}

RE_INSTRUMENT_NUMBER = re.compile(
    r"\b\d{1,4}/\d{4}/(?:NĐ-CP|ND-CP|TT-[A-ZĐ]+|TT-[A-Z]+|QĐ-[A-ZĐ]+|QD-[A-Z]+|NQ-[A-ZĐ]+|NQ-[A-Z]+|[A-ZĐ]{1,8}(?:-[A-ZĐ]{1,8})*)\b",
    re.I,
)
RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
RE_VERSION_CUE = re.compile(
    r"\b(?:hiện\s+hành|mới\s+nhất|sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|"
    r"có\s+hiệu\s+lực|hết\s+hiệu\s+lực|áp\s+dụng\s+từ|trước\s+ngày|sau\s+ngày)\b",
    re.I,
)
RE_CROSS_REF = re.compile(
    r"\b(?:theo|căn\s+cứ|quy\s+định\s+tại|dẫn\s+chiếu|được\s+quy\s+định\s+tại)\b",
    re.I,
)

ARCHETYPE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("SANCTION", re.compile(r"\b(?:phạt|xử\s+phạt|mức\s+phạt|phạt\s+tiền|truy\s+cứu|hình\s+phạt|vi\s+phạm)\b", re.I)),
    ("PROCEDURE", re.compile(r"\b(?:thủ\s+tục|hồ\s+sơ|trình\s+tự|các\s+bước|nộp|đăng\s+ký|cấp|gia\s+hạn|giải\s+quyết)\b", re.I)),
    ("CONDITION", re.compile(r"\b(?:điều\s+kiện|đủ\s+điều\s+kiện|được\s+phép|có\s+được|yêu\s+cầu|tiêu\s+chuẩn|đối\s+tượng)\b", re.I)),
    ("DEFINITION", re.compile(r"\b(?:là\s+gì|được\s+hiểu\s+là|khái\s+niệm|định\s+nghĩa|thế\s+nào\s+là)\b", re.I)),
    ("NUMERICAL_THRESHOLD", re.compile(r"\b(?:bao\s+nhiêu|mức|tối\s+đa|tối\s+thiểu|ít\s+nhất|không\s+quá|thời\s+hạn|bao\s+lâu|tỷ\s+lệ)\b", re.I)),
    ("ELIGIBILITY", re.compile(r"\b(?:ai\s+được|ai\s+phải|trường\s+hợp\s+nào|đối\s+tượng\s+nào|được\s+hưởng|được\s+miễn|được\s+giảm)\b", re.I)),
]

ENTITY_HINTS = re.compile(
    r"\b(?:công\s+ty|doanh\s+nghiệp|cơ\s+quan|ủy\s+ban|bộ\s+[A-ZÀ-Ỹ]|"
    r"người\s+lao\s+động|người\s+sử\s+dụng\s+lao\s+động|cá\s+nhân|tổ\s+chức|"
    r"hộ\s+kinh\s+doanh|chủ\s+đầu\s+tư|nhà\s+thầu)\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return RE_WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _extract_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        for key in ("data", "queries", "questions", "items"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        else:
            if obj and all(isinstance(v, (str, dict)) for v in obj.values()):
                records = []
                for k, v in obj.items():
                    if isinstance(v, str):
                        records.append({"qid": k, "question": v})
                    else:
                        row = dict(v)
                        row.setdefault("qid", k)
                        records.append(row)
            else:
                raise ValueError("Could not identify query list in top-level JSON object.")
    else:
        raise ValueError(f"Unsupported top-level JSON type: {type(obj).__name__}")

    if not all(isinstance(x, dict) for x in records):
        raise ValueError("Query collection must contain JSON objects.")
    return records


def pick_field(row: dict[str, Any], candidates: Iterable[str]) -> tuple[str | None, Any]:
    for key in candidates:
        if key in row and row[key] is not None:
            return key, row[key]
    return None, None


def classify_archetype(text: str, flags: dict[str, Any]) -> str:
    if flags["has_explicit_legal_locator"] and (
        re.search(r"\b(?:theo|tại|quy\s+định|nội\s+dung|quy\s+định\s+gì)\b", text, re.I)
        or len(text.split()) <= 35
    ):
        return "DIRECT_REF"
    for name, pattern in ARCHETYPE_RULES:
        if pattern.search(text):
            return name
    if flags["has_cross_ref_cue"] and flags["instrument_type_count"] >= 1:
        return "CROSS_REFERENCE"
    return "GENERAL_SEMANTIC"


def pct(values: list[int | float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def describe_numeric(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "p25": pct(values, 0.25),
        "median": pct(values, 0.50),
        "mean": float(statistics.fmean(values)),
        "p75": pct(values, 0.75),
        "p90": pct(values, 0.90),
        "p95": pct(values, 0.95),
        "max": float(max(values)),
    }


def audit(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records = _extract_records(raw)

    rows: list[dict[str, Any]] = []
    qid_key_counts: Counter[str] = Counter()
    text_key_counts: Counter[str] = Counter()
    archetypes: Counter[str] = Counter()
    instrument_counts: Counter[str] = Counter()
    schema_keys: Counter[str] = Counter()
    normalized_text_to_qids: defaultdict[str, list[str]] = defaultdict(list)

    for idx, rec in enumerate(records):
        schema_keys.update(rec.keys())
        qid_key, qid_value = pick_field(rec, QID_KEYS)
        text_key, text_value = pick_field(rec, TEXT_KEYS)
        if qid_key is None:
            qid = f"__row_{idx:06d}"
        else:
            qid = str(qid_value)
            qid_key_counts[qid_key] += 1
        if text_key is None:
            raise ValueError(f"Row {idx} ({qid}) has no question text. Keys={sorted(rec)}")
        text_key_counts[text_key] += 1

        text = normalize_text(str(text_value))
        words = RE_WORD.findall(text)
        loc_counts = {name: len(p.findall(text)) for name, p in LEGAL_PATTERNS.items()}
        inst = {name: bool(p.search(text)) for name, p in INSTRUMENT_PATTERNS.items()}
        for k, present in inst.items():
            if present:
                instrument_counts[k] += 1

        has_locator = any(loc_counts.values())
        flags = {
            "has_explicit_legal_locator": has_locator,
            "has_instrument_number": bool(RE_INSTRUMENT_NUMBER.search(text)),
            "has_year": bool(RE_YEAR.search(text)),
            "has_version_cue": bool(RE_VERSION_CUE.search(text)),
            "has_cross_ref_cue": bool(RE_CROSS_REF.search(text)),
            "instrument_type_count": sum(inst.values()),
            "has_numeric_expression": bool(RE_NUMBER.search(text)),
            "has_entity_hint": bool(ENTITY_HINTS.search(text)),
        }
        archetype = classify_archetype(text, flags)
        archetypes[archetype] += 1

        normalized_key = text.casefold()
        normalized_text_to_qids[normalized_key].append(qid)

        row = {
            "row_index": idx,
            "qid": qid,
            "question": text,
            "char_len": len(text),
            "word_len": len(words),
            **{f"count_{k}": v for k, v in loc_counts.items()},
            **{f"instrument_{k}": int(v) for k, v in inst.items()},
            **{k: int(v) if isinstance(v, bool) else v for k, v in flags.items()},
            "archetype": archetype,
        }
        rows.append(row)

    duplicate_groups = [
        {"question": text, "qids": qids, "count": len(qids)}
        for text, qids in normalized_text_to_qids.items()
        if len(qids) > 1
    ]
    duplicate_groups.sort(key=lambda x: (-x["count"], x["question"]))

    char_lengths = [r["char_len"] for r in rows]
    word_lengths = [r["word_len"] for r in rows]

    def count_flag(name: str) -> int:
        return sum(int(r[name]) for r in rows)

    n = len(rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "top_level_type": type(raw).__name__,
        "n_queries": n,
        "schema": {
            "record_keys_frequency": dict(schema_keys.most_common()),
            "qid_keys_detected": dict(qid_key_counts),
            "text_keys_detected": dict(text_key_counts),
        },
        "length": {
            "chars": describe_numeric(char_lengths),
            "words": describe_numeric(word_lengths),
        },
        "legal_reference": {
            "explicit_locator_queries": count_flag("has_explicit_legal_locator"),
            "instrument_number_queries": count_flag("has_instrument_number"),
            "year_queries": count_flag("has_year"),
            "version_cue_queries": count_flag("has_version_cue"),
            "cross_reference_cue_queries": count_flag("has_cross_ref_cue"),
        },
        "query_signals": {
            "numeric_expression_queries": count_flag("has_numeric_expression"),
            "entity_hint_queries": count_flag("has_entity_hint"),
        },
        "instrument_types": dict(instrument_counts.most_common()),
        "archetypes": dict(archetypes.most_common()),
        "duplicates": {
            "exact_normalized_duplicate_groups": len(duplicate_groups),
            "queries_inside_duplicate_groups": sum(x["count"] for x in duplicate_groups),
            "groups": duplicate_groups,
        },
    }

    for section_name in ("legal_reference", "query_signals"):
        section = summary[section_name]
        for k, v in list(section.items()):
            section[k + "_rate"] = (v / n) if n else 0.0
    summary["archetype_rates"] = {k: (v / n if n else 0.0) for k, v in archetypes.items()}
    summary["instrument_type_rates"] = {k: (v / n if n else 0.0) for k, v in instrument_counts.items()}
    return summary, rows


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "PRIVATE_QUERY_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / "PRIVATE_QUERY_FEATURES.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["qid", "question"])
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "PRIVATE_QUERY_FEATURES.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = summary["n_queries"]
    legal = summary["legal_reference"]
    md = [
        "# Private Query Audit",
        "",
        f"- Queries: **{n}**",
        f"- SHA256: `{summary['input_sha256']}`",
        f"- Median words: **{summary['length']['words']['median']:.1f}**",
        f"- P95 words: **{summary['length']['words']['p95']:.1f}**",
        f"- Explicit Điều/Khoản/Điểm/etc.: **{legal['explicit_locator_queries']} ({legal['explicit_locator_queries_rate']:.1%})**",
        f"- Explicit instrument number: **{legal['instrument_number_queries']} ({legal['instrument_number_queries_rate']:.1%})**",
        f"- Version/temporal cue: **{legal['version_cue_queries']} ({legal['version_cue_queries_rate']:.1%})**",
        "",
        "## Archetypes (heuristic, label-free)",
        "",
        "| Archetype | Count | Rate |",
        "|---|---:|---:|",
    ]
    for name, count in summary["archetypes"].items():
        md.append(f"| {name} | {count} | {count / n:.1%} |" if n else f"| {name} | {count} | 0% |")

    md += ["", "## Instrument mentions", "", "| Type | Count | Rate |", "|---|---:|---:|"]
    for name, count in summary["instrument_types"].items():
        md.append(f"| {name} | {count} | {count / n:.1%} |" if n else f"| {name} | {count} | 0% |")

    md += [
        "",
        "## Notes",
        "",
        "- Archetypes are deterministic heuristics for distribution analysis, not gold labels.",
        "- Exact duplicate detection uses NFC + whitespace normalization + casefold.",
        "- Embedding similarity / train-public comparison is intentionally deferred to the next Stage-01 script.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()

    if not input_path.exists():
        raise SystemExit(
            f"Input not found: {input_path}\n"
            "Expected private-official.json at repo root or pass --input PATH."
        )

    summary, rows = audit(input_path)
    write_outputs(summary, rows, out_dir)

    print("=== PRIVATE QUERY AUDIT COMPLETE ===")
    print(f"input    : {input_path}")
    print(f"queries  : {summary['n_queries']}")
    print(f"sha256   : {summary['input_sha256']}")
    print(f"out_dir  : {out_dir}")
    print("archetypes:")
    for name, count in summary["archetypes"].items():
        rate = count / summary["n_queries"] if summary["n_queries"] else 0.0
        print(f"  {name:20s} {count:5d}  {rate:7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
