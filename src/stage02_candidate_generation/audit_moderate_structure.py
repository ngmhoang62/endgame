#!/usr/bin/env python
"""Stage 02B0 — moderate structural geometry audit.

This stage does NOT load an embedding model and does NOT retrieve.

Goal:
Build a deliberately moderate, model-agnostic structural representation:
- recognize Chapter / Section / Article / Annex headings conservatively;
- Article and Annex are retrieval regions;
- Chapter/Section are metadata context, NOT standalone subtree chunks;
- uncovered source spans become fallback/gap regions so source coverage is exact;
- NO Clause / Point nodes;
- NO paragraph packing;
- NO sliding windows yet;
- NO metadata-rich retrieval text yet.

This directly tests whether we can start from a much smaller structural geometry
than historical structural-v3 (~343k chunks) before model-specific packing.

Run:
    python src/stage02_candidate_generation/audit_moderate_structure.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "evaluation_v2"
CACHE = ROOT / "cache" / "stage02b_structure"
OUT = ROOT / "reports" / "stage02b0_structure_geometry"

CHAPTER_RE = re.compile(
    r"(?mi)^[ \t]*Chương[ \t]+(?P<label>(?:[IVXLCDM]+|\d+))\b(?P<tail>[^\r\n]*)"
)
SECTION_RE = re.compile(
    r"(?mi)^[ \t]*Mục[ \t]+(?P<label>\d+[a-zđ]?)(?P<tail>[^\r\n]*)"
)
ARTICLE_RE = re.compile(
    r"(?mi)^[ \t]*Điều[ \t]+(?P<label>\d+[a-zđ]?)(?:[ \t]*[\.:])?"
    r"(?=[ \t\r\n]|$)(?P<tail>[^\r\n]*)"
)
SPLIT_ARTICLE_RE = re.compile(
    r"(?mi)^[ \t]*Điều[ \t]*(?:\r?\n[ \t]*){1,3}"
    r"(?P<label>\d+[a-zđ]?)(?:[ \t]*[\.:])?"
    r"(?=[ \t\r\n]|$)(?P<tail>[^\r\n]*)"
)
ANNEX_RE = re.compile(
    r"(?mi)^[ \t]*(?P<label>PHỤ[ \t]+LỤC|DANH[ \t]+MỤC|MẪU)"
    r"(?P<tail>[^\r\n]{0,140})"
)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values):
    x = np.asarray(list(values), dtype=np.float64)
    if not len(x):
        return {}
    return {
        "min": float(x.min()),
        "p25": float(np.quantile(x, .25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.quantile(x, .75)),
        "p90": float(np.quantile(x, .90)),
        "p95": float(np.quantile(x, .95)),
        "p99": float(np.quantile(x, .99)),
        "max": float(x.max()),
    }


def line_end(text: str, start: int) -> int:
    n = text.find("\n", start)
    return len(text) if n < 0 else n


def marker_rows(text: str):
    rows = []
    for kind, regex in (
        ("chapter", CHAPTER_RE),
        ("section", SECTION_RE),
        ("article", ARTICLE_RE),
        ("article_split", SPLIT_ARTICLE_RE),
        ("annex", ANNEX_RE),
    ):
        for m in regex.finditer(text):
            heading = compact(m.group(0))
            if kind == "annex" and heading != heading.upper():
                continue
            rows.append({
                "kind": "article" if kind == "article_split" else kind,
                "start": int(m.start()),
                "heading_end": int(m.end()),
                "label": compact(m.group("label")),
                "heading": heading,
                "split_article_heading": kind == "article_split",
            })

    priority = {"annex": 0, "chapter": 1, "section": 2, "article": 3}
    rows.sort(key=lambda x: (x["start"], priority[x["kind"]]))

    # Same source location may match ordinary + split article regex.
    dedup = []
    occupied = set()
    for row in rows:
        if row["start"] in occupied:
            continue
        occupied.add(row["start"])
        dedup.append(row)
    return dedup


def build_regions(doc_id: str, text: str):
    markers = marker_rows(text)
    major = [m for m in markers if m["kind"] in {"article", "annex"}]

    # Metadata state at each major retrieval region.
    def context_before(pos):
        chapter = section = ""
        for m in markers:
            if m["start"] > pos:
                break
            if m["kind"] == "chapter":
                chapter = m["heading"]
                section = ""
            elif m["kind"] == "section":
                section = m["heading"]
            elif m["kind"] == "annex":
                chapter = section = ""
        return chapter, section

    regions = []
    covered = []

    for i, m in enumerate(major):
        end = major[i + 1]["start"] if i + 1 < len(major) else len(text)
        start = m["start"]
        if end <= start:
            continue
        chapter, section = context_before(start)
        raw = text[start:end]
        if raw.strip():
            regions.append({
                "doc_id": doc_id,
                "region_id": f"{doc_id}:{m['kind']}:{start:09d}:{end:09d}",
                "kind": m["kind"],
                "start": start,
                "end": end,
                "heading": m["heading"],
                "label": m["label"],
                "chapter_heading": chapter,
                "section_heading": section,
                "raw_text": raw,
                "split_article_heading": bool(m.get("split_article_heading")),
            })
            covered.append((start, end))

    # Add gaps so raw source coverage is exact. We deliberately do not split
    # these gaps into paragraphs: that would already push us toward fine chunking.
    cursor = 0
    gap_index = 0
    for start, end in sorted(covered):
        if cursor < start:
            raw = text[cursor:start]
            if raw.strip():
                regions.append({
                    "doc_id": doc_id,
                    "region_id": f"{doc_id}:gap:{cursor:09d}:{start:09d}",
                    "kind": "gap",
                    "start": cursor,
                    "end": start,
                    "heading": "",
                    "label": "",
                    "chapter_heading": "",
                    "section_heading": "",
                    "raw_text": raw,
                    "split_article_heading": False,
                })
                gap_index += 1
        cursor = max(cursor, end)
    if cursor < len(text):
        raw = text[cursor:]
        if raw.strip():
            regions.append({
                "doc_id": doc_id,
                "region_id": f"{doc_id}:gap:{cursor:09d}:{len(text):09d}",
                "kind": "gap",
                "start": cursor,
                "end": len(text),
                "heading": "",
                "label": "",
                "chapter_heading": "",
                "section_heading": "",
                "raw_text": raw,
                "split_article_heading": False,
            })

    if not regions and text.strip():
        regions = [{
            "doc_id": doc_id,
            "region_id": f"{doc_id}:fallback:000000000:{len(text):09d}",
            "kind": "fallback",
            "start": 0,
            "end": len(text),
            "heading": "",
            "label": "",
            "chapter_heading": "",
            "section_heading": "",
            "raw_text": text,
            "split_article_heading": False,
        }]

    regions.sort(key=lambda x: (x["start"], x["end"], x["kind"]))

    # Strong integrity: after merging region coverage there must be no uncovered
    # non-whitespace source content.
    mask = bytearray(len(text))
    for r in regions:
        for j in range(r["start"], r["end"]):
            mask[j] = 1
    uncovered_non_ws = [
        i for i, ch in enumerate(text)
        if not ch.isspace() and not mask[i]
    ]
    if uncovered_non_ws:
        raise RuntimeError(
            f"{doc_id}: {len(uncovered_non_ws)} non-whitespace chars uncovered"
        )

    return markers, regions


def main():
    corpus_path = EVAL / "retrieval_corpus_8512.jsonl"
    gold_path = EVAL / "primary_golds_6991.json"
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)

    golds = {
        str(q): [str(d) for d in ds]
        for q, ds in read_json(gold_path).items()
    }
    unique_gold_docs = {d for ds in golds.values() for d in ds}

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    region_path = CACHE / "moderate_structure_regions.jsonl"

    docs = 0
    total_regions = 0
    kind_counts = Counter()
    docs_with_article = 0
    docs_with_annex = 0
    docs_fallback_only = 0
    split_heading_docs = 0
    region_counts_per_doc = []
    chars = []
    words = []
    article_chars = []
    article_words = []
    gap_chars = []
    gold_doc_region_counts = []
    gold_docs_with_article = 0
    largest_docs = []

    with corpus_path.open("r", encoding="utf-8") as src, \
         region_path.open("w", encoding="utf-8", newline="\n") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["id"])
            text = str(row["passage"])
            markers, regions = build_regions(doc_id, text)

            docs += 1
            total_regions += len(regions)
            region_counts_per_doc.append(len(regions))
            kinds = Counter(r["kind"] for r in regions)
            kind_counts.update(kinds)
            docs_with_article += int(kinds["article"] > 0)
            docs_with_annex += int(kinds["annex"] > 0)
            docs_fallback_only += int(
                len(regions) == 1 and regions[0]["kind"] in {"gap", "fallback"}
            )
            split_heading_docs += int(any(m["split_article_heading"] for m in markers))

            for r in regions:
                c = len(r["raw_text"])
                w = len(WORD_RE.findall(r["raw_text"]))
                r["char_count"] = c
                r["word_count"] = w
                chars.append(c)
                words.append(w)
                if r["kind"] == "article":
                    article_chars.append(c)
                    article_words.append(w)
                if r["kind"] == "gap":
                    gap_chars.append(c)
                out.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

            if doc_id in unique_gold_docs:
                gold_doc_region_counts.append(len(regions))
                gold_docs_with_article += int(kinds["article"] > 0)

            largest_docs.append((len(regions), len(text), doc_id, kinds["article"], kinds["gap"]))

            if docs % 1000 == 0 or docs == 8512:
                print(f"[geometry] {docs}/8512", flush=True)

    if docs != 8512:
        raise RuntimeError(f"Expected 8512 docs, got {docs}")

    largest_docs.sort(reverse=True)

    historical_chunks = 343_347
    report = {
        "schema_version": "dsc2026.endgame.stage02b0.moderate_structure.v1",
        "status": "PASS",
        "claim_boundary": "geometry only; no embedding/retrieval conclusions",
        "source_corpus_sha256": sha256(corpus_path),
        "contract": {
            "chapter_section_role": "metadata only",
            "article_annex_role": "retrieval regions",
            "clause_point_nodes": False,
            "paragraph_packing": False,
            "sliding_windows": False,
            "metadata_prefix_rendering": False,
            "exact_source_offsets": True,
            "gap_regions_preserve_uncovered_source": True,
        },
        "documents": {
            "count": docs,
            "with_article": docs_with_article,
            "with_article_rate": docs_with_article / docs,
            "with_annex": docs_with_annex,
            "fallback_or_gap_only": docs_fallback_only,
            "split_article_heading_docs": split_heading_docs,
        },
        "regions": {
            "count": total_regions,
            "kind_counts": dict(kind_counts),
            "per_document": describe(region_counts_per_doc),
            "char_length": describe(chars),
            "word_length": describe(words),
            "article_char_length": describe(article_chars),
            "article_word_length": describe(article_words),
            "gap_char_length": describe(gap_chars),
        },
        "gold_documents": {
            "unique_count": len(unique_gold_docs),
            "with_article": gold_docs_with_article,
            "with_article_rate": gold_docs_with_article / len(unique_gold_docs),
            "regions_per_document": describe(gold_doc_region_counts),
        },
        "historical_comparison": {
            "structural_v3_chunk_count": historical_chunks,
            "moderate_region_count": total_regions,
            "region_count_ratio_vs_structural_v3": total_regions / historical_chunks,
            "structural_v3_to_moderate_compression": historical_chunks / total_regions,
            "note": (
                "Not an apples-to-apples retrieval metric. This only measures "
                "base geometry before model-specific token packing."
            ),
        },
        "largest_region_count_documents": [
            {
                "doc_id": doc_id,
                "region_count": n_regions,
                "char_count": n_chars,
                "article_regions": n_articles,
                "gap_regions": n_gaps,
            }
            for n_regions, n_chars, doc_id, n_articles, n_gaps in largest_docs[:30]
        ],
        "artifact": {
            "regions_path": str(region_path.resolve()),
            "regions_sha256": sha256(region_path),
        },
    }

    (OUT / "STRUCTURE_GEOMETRY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = [
        "# Stage 02B0 — Moderate Structure Geometry",
        "",
        "**Geometry only — no retrieval/model winner is implied.**",
        "",
        "## Contract",
        "",
        "- Chapter/Mục: metadata context only",
        "- Điều/Phụ lục: retrieval regions",
        "- Clause/Point nodes: **disabled**",
        "- Paragraph packing: **disabled**",
        "- Sliding windows: **not applied yet**",
        "- Exact source offsets: preserved",
        "",
        "## Headline",
        "",
        f"- Documents: **{docs}**",
        f"- Moderate regions: **{total_regions:,}**",
        f"- Historical structural-v3 chunks: **{historical_chunks:,}**",
        f"- Geometry compression vs structural-v3: **{historical_chunks/total_regions:.2f}× fewer base regions**",
        f"- Docs with article structure: **{docs_with_article} ({docs_with_article/docs:.2%})**",
        f"- Unique gold docs with article structure: **{gold_docs_with_article}/{len(unique_gold_docs)} ({gold_docs_with_article/len(unique_gold_docs):.2%})**",
        "",
        "## Regions per document",
        "",
        f"- Median: **{report['regions']['per_document']['median']:.1f}**",
        f"- P90: **{report['regions']['per_document']['p90']:.1f}**",
        f"- P95: **{report['regions']['per_document']['p95']:.1f}**",
        f"- P99: **{report['regions']['per_document']['p99']:.1f}**",
        f"- Max: **{report['regions']['per_document']['max']:.0f}**",
        "",
        "## Article size in words",
        "",
        f"- Median: **{report['regions']['article_word_length'].get('median',0):.1f}**",
        f"- P90: **{report['regions']['article_word_length'].get('p90',0):.1f}**",
        f"- P95: **{report['regions']['article_word_length'].get('p95',0):.1f}**",
        f"- P99: **{report['regions']['article_word_length'].get('p99',0):.1f}**",
        "",
        "Next: choose model-specific packing budgets only after inspecting this geometry.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "documents": docs,
        "moderate_regions": total_regions,
        "historical_structural_v3_chunks": historical_chunks,
        "compression_vs_structural_v3": historical_chunks / total_regions,
        "docs_with_article_rate": docs_with_article / docs,
        "gold_docs_with_article_rate": gold_docs_with_article / len(unique_gold_docs),
        "median_regions_per_doc": report["regions"]["per_document"]["median"],
        "p95_regions_per_doc": report["regions"]["per_document"]["p95"],
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
