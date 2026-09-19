#!/usr/bin/env python
"""Stage 01B-preflight: query data contract + preprocessing audit.

This stage MUST pass before semantic/private distribution diagnostics.

It does NOT modify source data. It creates multiple diagnostic match keys but
keeps model text conservative (NFC + whitespace compaction only).

Run:
  python src/stage01_query_analysis/preflight_query_data.py \
    --sota-root D:/Study/DSC2026/sota \
    --legalir-root D:/Study/DSC2026/LegalIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.query_preprocessing import (
    model_text, strict_match_key, punctuation_match_key, accent_match_key,
    word_count, suspicious_unicode,
)

DEFAULT_SOTA = Path("D:/Study/DSC2026/sota")
DEFAULT_LEGALIR = Path("D:/Study/DSC2026/LegalIR")
DEFAULT_OUT = ROOT / "reports/stage01_query_preflight"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{kind}: top-level must be object, got {type(raw).__name__}")
    out = {}
    for raw_qid, value in raw.items():
        qid = str(raw_qid)
        if qid in out:
            raise ValueError(f"{kind}: duplicate qid after str(): {qid}")
        if isinstance(value, str):
            row = {"question": value}
        elif isinstance(value, dict):
            row = dict(value)
        else:
            raise ValueError(f"{kind}/{qid}: unsupported row type {type(value).__name__}")
        if "question" not in row:
            raise ValueError(f"{kind}/{qid}: missing question")
        out[qid] = row
    return out


def group_keys(rows: dict[str, dict[str, Any]], key_fn: Callable[[str], str]):
    groups = defaultdict(list)
    for qid, row in rows.items():
        groups[key_fn(row["question"])].append(qid)
    return {k: v for k, v in groups.items() if k and len(v) > 1}


def cross_overlap(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    key_fn: Callable[[str], str],
    limit_examples: int = 50,
):
    by_key = defaultdict(list)
    for qid, row in b.items():
        key = key_fn(row["question"])
        if key:
            by_key[key].append(qid)
    hits = []
    for qid, row in a.items():
        key = key_fn(row["question"])
        if key in by_key:
            hits.append({
                "left_qid": qid,
                "right_qids": by_key[key],
                "question": model_text(row["question"]),
            })
    return {
        "count": len(hits),
        "rate_left": len(hits) / max(1, len(a)),
        "examples": hits[:limit_examples],
    }


def dataset_audit(name: str, rows: dict[str, dict[str, Any]], expect_answer: str):
    issues = Counter()
    words = []
    chars = []
    answer_sizes = []
    null_answers = 0
    nonnull_answers = 0

    for qid, row in rows.items():
        raw_q = row["question"]
        q = model_text(raw_q)
        if not q:
            issues["empty_question"] += 1
        wc = word_count(q)
        words.append(wc)
        chars.append(len(q))
        if wc <= 2:
            issues["question_le_2_words"] += 1
        if wc >= 80:
            issues["question_ge_80_words"] += 1
        for issue, active in suspicious_unicode(raw_q).items():
            if active:
                issues[issue] += 1

        answer = row.get("answer", "__MISSING__")
        if answer is None:
            null_answers += 1
        else:
            nonnull_answers += 1
        if expect_answer == "train":
            if not isinstance(answer, list):
                issues["train_answer_not_list"] += 1
            else:
                answer_sizes.append(len(answer))
                if not answer:
                    issues["train_empty_answer"] += 1
                if len(set(map(str, answer))) != len(answer):
                    issues["duplicate_doc_id_within_answer"] += 1
        elif expect_answer == "unlabeled":
            if isinstance(answer, list) and len(answer) > 0:
                issues["UNEXPECTED_NONEMPTY_TEST_LABEL"] += 1
            elif answer not in (None, "__MISSING__"):
                issues["unexpected_test_answer_type"] += 1

    def desc(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        def p(frac):
            pos = (len(xs)-1)*frac
            lo = int(pos); hi = min(lo+1, len(xs)-1)
            t = pos-lo
            return xs[lo]*(1-t)+xs[hi]*t
        return {
            "min": xs[0], "median": p(.5), "p95": p(.95),
            "max": xs[-1], "mean": sum(xs)/len(xs),
        }

    return {
        "name": name,
        "n": len(rows),
        "question_chars": desc(chars),
        "question_words": desc(words),
        "answer_sizes": desc(answer_sizes),
        "null_answers": null_answers,
        "nonnull_answers": nonnull_answers,
        "issues": dict(issues),
        "within_duplicates": {
            "strict": len(group_keys(rows, strict_match_key)),
            "punctuation": len(group_keys(rows, punctuation_match_key)),
            "accent": len(group_keys(rows, accent_match_key)),
        },
    }


def safe_exists(path: Path) -> tuple[bool, str | None]:
    """Return existence without letting Windows ACL/junction errors kill preflight."""
    try:
        return path.exists(), None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def optional_legacy_preprocess_audit(legalir_root: Path):
    """Audit legacy preprocessing only when accessible.

    This is intentionally OPTIONAL. ENDGAME's mandatory preflight inputs are
    PRIVATE/TRAIN/PUBLIC from the sota dataset tree. A missing, stale, moved,
    ACL-protected, junction-backed, or otherwise unreadable historical LegalIR
    cache must never invalidate those mandatory checks.
    """
    prep = legalir_root / "cache/final_preprocessed_v2"
    manifest = prep / "manifest.json"
    exclusions = prep / "exclusions.json"
    impact = prep / "train_label_impact.jsonl"

    manifest_ok, manifest_err = safe_exists(manifest)
    exclusions_ok, exclusions_err = safe_exists(exclusions)

    if not (manifest_ok and exclusions_ok):
        return {
            "available": False,
            "required_for_stage01b": False,
            "path": str(prep),
            "reason": "legacy preprocessing artifacts unavailable or inaccessible",
            "access_errors": {
                "manifest": manifest_err,
                "exclusions": exclusions_err,
            },
        }

    try:
        m = read_json(manifest)
        ex = read_json(exclusions)

        impact_ok, impact_err = safe_exists(impact)
        reason_counts = Counter(
            reason for row in ex for reason in row.get("reasons", [])
        )
        payload = {
            "available": True,
            "required_for_stage01b": False,
            "path": str(prep),
            "manifest_sha256": sha256(manifest),
            "exclusions_sha256": sha256(exclusions),
            "impact_sha256": sha256(impact) if impact_ok else None,
            "impact_access_error": impact_err,
            "manifest": m,
            "exclusion_rows": len(ex),
            "reason_counts": dict(reason_counts),
            "duplicate_mapping": {
                str(row["doc_id"]): str(row["duplicate_retained_id"])
                for row in ex
                if row.get("duplicate_retained_id")
            },
        }
        return payload
    except OSError as exc:
        return {
            "available": False,
            "required_for_stage01b": False,
            "path": str(prep),
            "reason": "legacy preprocessing became inaccessible during read/hash",
            "access_errors": {
                "read": f"{type(exc).__name__}: {exc}",
            },
        }
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "required_for_stage01b": False,
            "path": str(prep),
            "reason": "legacy preprocessing artifacts are malformed",
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sota-root", type=Path, default=DEFAULT_SOTA)
    p.add_argument("--legalir-root", type=Path, default=DEFAULT_LEGALIR)
    p.add_argument("--private", type=Path, default=ROOT/"private-official.json")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    sota = args.sota_root.resolve()
    legalir = args.legalir_root.resolve()
    private_path = args.private.resolve()
    data = sota / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    train_path = data/"train.json"
    public_path = data/"public-official.json"

    required = [private_path, train_path, public_path]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit("Missing required input(s):\n" + "\n".join(missing))

    private = load_dataset(private_path, "PRIVATE")
    train = load_dataset(train_path, "TRAIN")
    public = load_dataset(public_path, "PUBLIC")

    audits = {
        "PRIVATE": dataset_audit("PRIVATE", private, "unlabeled"),
        "TRAIN": dataset_audit("TRAIN", train, "train"),
        "PUBLIC": dataset_audit("PUBLIC", public, "unlabeled"),
    }

    overlaps = {}
    for right_name, right in [("TRAIN", train), ("PUBLIC", public)]:
        overlaps[f"PRIVATE_vs_{right_name}"] = {
            "qid_intersection_count": len(set(private) & set(right)),
            "qid_intersection": sorted(set(private) & set(right))[:200],
            "strict": cross_overlap(private, right, strict_match_key),
            "punctuation": cross_overlap(private, right, punctuation_match_key),
            "accent": cross_overlap(private, right, accent_match_key),
        }
    overlaps["PUBLIC_vs_TRAIN"] = {
        "qid_intersection_count": len(set(public) & set(train)),
        "strict": cross_overlap(public, train, strict_match_key),
        "punctuation": cross_overlap(public, train, punctuation_match_key),
        "accent": cross_overlap(public, train, accent_match_key),
    }

    critical = []
    for name in ("PRIVATE", "PUBLIC"):
        issues = audits[name]["issues"]
        if issues.get("empty_question", 0):
            critical.append(f"{name}: empty_question")
        if issues.get("UNEXPECTED_NONEMPTY_TEST_LABEL", 0):
            critical.append(f"{name}: test labels unexpectedly exposed")
        if issues.get("replacement_char", 0) or issues.get("nul", 0):
            critical.append(f"{name}: broken Unicode/control data")
    if audits["TRAIN"]["issues"].get("train_answer_not_list", 0):
        critical.append("TRAIN: malformed answer schema")

    legacy = optional_legacy_preprocess_audit(legalir)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "dsc2026.endgame.stage01b_query_preflight.v2.1",
        "status": "PASS" if not critical else "FAIL",
        "private_labels_used": False,
        "normalization_contract": {
            "model_text": "NFC + whitespace compaction only",
            "strict_match": "model_text + casefold; diagnostics only",
            "punctuation_match": "NFKC + casefold + punctuation/symbol folding; diagnostics only",
            "accent_match": "punctuation_match + Vietnamese accent folding; diagnostics only",
            "prohibition": "diagnostic keys MUST NOT replace model text or collapse query identities",
        },
        "inputs": {
            "PRIVATE": {"path": str(private_path), "sha256": sha256(private_path)},
            "TRAIN": {"path": str(train_path), "sha256": sha256(train_path)},
            "PUBLIC": {"path": str(public_path), "sha256": sha256(public_path)},
        },
        "datasets": audits,
        "cross_population_overlap": overlaps,
        "legacy_corpus_preprocessing_reference": legacy,
        "critical_failures": critical,
    }
    report_path = out/"QUERY_PREFLIGHT.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Stage 01B Query Preflight",
        "",
        f"**Status: {payload['status']}**",
        "",
        "| Population | N | Median words | Empty | Strict dup groups | Accent-fold dup groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("PRIVATE", "TRAIN", "PUBLIC"):
        x = audits[name]
        md.append(
            f"| {name} | {x['n']} | {x['question_words'].get('median', 0):.1f} | "
            f"{x['issues'].get('empty_question',0)} | "
            f"{x['within_duplicates']['strict']} | {x['within_duplicates']['accent']} |"
        )

    md += ["", "## Cross-population overlap", ""]
    for name, x in overlaps.items():
        md.append(f"### {name}")
        md.append(f"- QID intersection: {x.get('qid_intersection_count', 0)}")
        for level in ("strict", "punctuation", "accent"):
            if level in x:
                md.append(f"- {level}: {x[level]['count']} ({x[level]['rate_left']:.2%})")
        md.append("")

    md += ["## Legacy preprocessing reference", ""]
    if legacy.get("available"):
        md.append(f"- Available: yes")
        md.append(f"- Path: `{legacy.get('path')}`")
        md.append(f"- Exclusion rows: {legacy.get('exclusion_rows')}")
    else:
        md.append("- Available: no (**non-blocking**)")
        md.append(f"- Reason: {legacy.get('reason')}")
        errors = legacy.get("access_errors") or {}
        for k, v in errors.items():
            if v:
                md.append(f"- {k}: `{v}`")

    md += [
        "",
        "## Important contract",
        "",
        "- Model input text is **not aggressively normalized**.",
        "- Accent/punctuation folding exists only to detect contamination/duplicates.",
        "- No query is removed or merged by this stage.",
        "- Legacy LegalIR preprocessing is optional historical evidence, never a Stage 01B dependency.",
        "- PRIVATE labels used: **false**.",
        "",
    ]
    (out/"REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "private": len(private),
        "train": len(train),
        "public": len(public),
        "legacy_preprocessing_available": legacy.get("available", False),
        "legacy_preprocessing_reason": legacy.get("reason"),
        "critical_failures": critical,
        "out": str(out),
    }, ensure_ascii=False, indent=2))

    if critical:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
