#!/usr/bin/env python
"""Stage 00B — Evaluation Population Forensics.

This stage DOES NOT create folds.

It audits:
1. raw official label population;
2. corpus/document identity and data-quality issues;
3. exact/punctuation/accent query duplicates;
4. semantic near-duplicate candidates using the fresh ENDGAME VietLegal-E5;
5. gold-frequency / answer-count strata;
6. threshold sensitivity of near-duplicate graph components.

Scientific rule:
- Raw official document IDs are preserved.
- Exact duplicate document content is NOT canonicalized away.
- Empty-passage golds are NOT silently removed from the primary population.
- Any alternative filtered/canonicalized evaluation is secondary diagnostic only.

Run:
    python src/stage00_audit/audit_evaluation_population.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.query_preprocessing import (
    model_text,
    strict_match_key,
    punctuation_match_key,
    accent_match_key,
)

os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "cache" / "huggingface" / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

DATA = ROOT / "data" / "official_v1"
MODEL_DIR = ROOT / "models" / "vietlegal-e5"
MODEL_MANIFEST = ROOT / "reports" / "stage00_model_materialization" / "vietlegal-e5" / "MODEL_MANIFEST.json"
OUT = ROOT / "reports" / "stage00b_evaluation_population"
CACHE = ROOT / "cache" / "stage00b_evaluation_population"

SEMANTIC_THRESHOLDS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
TOPK_NEIGHBORS = 12
FINAL_NORM_TOL = 2e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("Invalid embedding norms.")
    x = x / norms
    err = float(np.max(np.abs(np.linalg.norm(x, axis=1) - 1.0)))
    if err > FINAL_NORM_TOL:
        raise RuntimeError(f"FP32 normalization failed: {err}")
    return x


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self):
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return list(groups.values())


def verify_snapshot():
    manifest_path = DATA / "DATA_SNAPSHOT_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing official data snapshot. Run snapshot_official_data.py first.")
    manifest = read_json(manifest_path)
    for filename in ("train.json", "public-official.json", "private-official.json"):
        p = DATA / filename
        if sha256(p) != manifest["top_level_sha256"][filename]:
            raise RuntimeError(f"Snapshot drift: {filename}")
    return manifest


def verify_model():
    if not MODEL_MANIFEST.exists() or not MODEL_DIR.exists():
        raise RuntimeError("Fresh ENDGAME VietLegal-E5 materialization missing.")
    m = read_json(MODEL_MANIFEST)
    if m.get("status") != "PASS" or m.get("historical_model_reused") is not False:
        raise RuntimeError("Fresh model contract invalid.")
    return m


def load_train():
    raw = read_json(DATA / "train.json")
    qids = []
    questions = {}
    answers = {}
    for raw_qid, row in raw.items():
        qid = str(raw_qid)
        qids.append(qid)
        questions[qid] = model_text(row.get("question", ""))
        ans = [str(x) for x in row.get("answer", [])]
        answers[qid] = ans
    return qids, questions, answers


def audit_contexts(answers):
    context_dir = DATA / "selected-contexts"
    files = sorted(context_dir.glob("context_*.json"))
    docs = {}
    passage_hash_groups = defaultdict(list)
    empty_ids = set()
    malformed = []
    missing_name = 0

    for i, path in enumerate(files, 1):
        try:
            row = read_json(path)
            doc_id = str(row.get("id", path.stem[len("context_"):]))
            if doc_id in docs:
                raise RuntimeError(f"Duplicate document id: {doc_id}")
            passage = str(row.get("passage") or "")
            name = str(row.get("name") or "")
            link = str(row.get("link") or "")
            docs[doc_id] = {
                "file": path.name,
                "passage": passage,
                "name": name,
                "link": link,
            }
            if not passage.strip():
                empty_ids.add(doc_id)
            else:
                ph = hashlib.sha256(passage.encode("utf-8")).hexdigest()
                passage_hash_groups[ph].append(doc_id)
            if not name.strip():
                missing_name += 1
        except Exception as exc:
            malformed.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
        if i % 1500 == 0 or i == len(files):
            print(f"[corpus] {i}/{len(files)}", flush=True)

    duplicate_groups = [
        sorted(ids) for ids in passage_hash_groups.values() if len(ids) > 1
    ]
    duplicate_groups.sort(key=lambda x: (-len(x), x))

    gold_ids = {d for vals in answers.values() for d in vals}
    missing_gold_ids = sorted(gold_ids - set(docs))
    gold_empty = sorted(gold_ids & empty_ids)
    duplicate_doc_ids = {d for g in duplicate_groups for d in g}
    gold_duplicate_content_ids = sorted(gold_ids & duplicate_doc_ids)

    q_with_missing_gold = []
    q_with_empty_gold = []
    q_with_duplicate_content_gold = []
    for qid, gold in answers.items():
        gs = set(gold)
        if gs & set(missing_gold_ids):
            q_with_missing_gold.append(qid)
        if gs & empty_ids:
            q_with_empty_gold.append(qid)
        if gs & duplicate_doc_ids:
            q_with_duplicate_content_gold.append(qid)

    return {
        "document_count": len(docs),
        "file_count": len(files),
        "malformed_count": len(malformed),
        "malformed": malformed[:50],
        "empty_passage_count": len(empty_ids),
        "empty_passage_ids": sorted(empty_ids),
        "missing_name_count": missing_name,
        "exact_duplicate_passage_group_count": len(duplicate_groups),
        "exact_duplicate_passage_groups": duplicate_groups,
        "gold_unique_doc_count": len(gold_ids),
        "missing_gold_doc_count": len(missing_gold_ids),
        "missing_gold_doc_ids": missing_gold_ids,
        "gold_empty_doc_count": len(gold_empty),
        "gold_empty_doc_ids": gold_empty,
        "gold_duplicate_content_doc_count": len(gold_duplicate_content_ids),
        "gold_duplicate_content_doc_ids": gold_duplicate_content_ids,
        "queries_with_missing_gold": q_with_missing_gold,
        "queries_with_empty_gold": q_with_empty_gold,
        "queries_with_duplicate_content_gold": q_with_duplicate_content_gold,
    }


def duplicate_groups(qids, questions, key_fn):
    by_key = defaultdict(list)
    for qid in qids:
        key = key_fn(questions[qid])
        if key:
            by_key[key].append(qid)
    groups = [sorted(v) for v in by_key.values() if len(v) > 1]
    groups.sort(key=lambda x: (-len(x), x))
    return groups


def encode_train(qids, questions, model_manifest, batch_size):
    import torch
    from sentence_transformers import SentenceTransformer

    CACHE.mkdir(parents=True, exist_ok=True)
    vec_path = CACHE / "train_queries.f32.npy"
    meta_path = CACHE / "train_queries.json"
    train_sha = sha256(DATA / "train.json")
    contract = {
        "schema": "endgame.stage00b.train_query_embeddings.v1",
        "train_sha256": train_sha,
        "qids": qids,
        "model_sha": model_manifest["resolved_revision_sha"],
        "model_content_fingerprint": model_manifest["content_fingerprint"],
        "prefix": "query: ",
        "fix_mistral_regex": True,
        "postprocess": "float32_explicit_l2",
    }
    ch = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if vec_path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("contract_hash") == ch:
            vec = np.asarray(np.load(vec_path), dtype=np.float32)
            if vec.shape[0] == len(qids):
                err = float(np.max(np.abs(np.linalg.norm(vec, axis=1) - 1.0)))
                if err <= FINAL_NORM_TOL:
                    print("[embed] Stage00B cache hit", flush=True)
                    return vec

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Stage00B embedding audit.")

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
        processor_kwargs={"fix_mistral_regex": True},
    )
    texts = ["query: " + questions[qid] for qid in qids]
    raw = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vec = normalize_f32(raw)
    np.save(vec_path, vec)
    meta_path.write_text(
        json.dumps({
            "contract_hash": ch,
            "contract": contract,
            "shape": list(vec.shape),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return vec


def semantic_topk(vectors, k=TOPK_NEIGHBORS):
    import torch

    device = torch.device("cuda")
    ref = torch.from_numpy(vectors).to(device)
    n = len(vectors)
    all_scores = np.empty((n, k), dtype=np.float32)
    all_idx = np.empty((n, k), dtype=np.int32)

    block = 256
    with torch.inference_mode():
        for start in range(0, n, block):
            q = ref[start:start + block]
            scores = q @ ref.T
            rows = torch.arange(q.shape[0], device=device)
            cols = torch.arange(start, start + q.shape[0], device=device)
            scores[rows, cols] = -2.0
            vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            all_scores[start:start + q.shape[0]] = vals.float().cpu().numpy()
            all_idx[start:start + q.shape[0]] = idx.int().cpu().numpy()
            print(f"[semantic-nn] {min(start+block,n)}/{n}", flush=True)

    return all_scores, all_idx


def build_lexical_matrix(qids, questions):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform([questions[q] for q in qids])
    return vectorizer, matrix


def pair_lexical_sim(matrix, i, j):
    return float(matrix[i].multiply(matrix[j]).sum())


def make_pair_rows(qids, questions, answers, scores, indices, lexical):
    seen = set()
    rows = []
    for i, qid in enumerate(qids):
        gold_i = set(answers[qid])
        for rank in range(scores.shape[1]):
            j = int(indices[i, rank])
            qj = qids[j]
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            sim = float(scores[i, rank])
            if sim < min(SEMANTIC_THRESHOLDS):
                continue
            gold_j = set(answers[qj])
            rows.append({
                "qid_a": qid,
                "qid_b": qj,
                "embedding_cosine": sim,
                "lexical_cosine": pair_lexical_sim(lexical, i, j),
                "same_strict_text": int(strict_match_key(questions[qid]) == strict_match_key(questions[qj])),
                "same_punctuation_text": int(punctuation_match_key(questions[qid]) == punctuation_match_key(questions[qj])),
                "same_accent_text": int(accent_match_key(questions[qid]) == accent_match_key(questions[qj])),
                "gold_overlap_count": len(gold_i & gold_j),
                "gold_jaccard": (
                    len(gold_i & gold_j) / len(gold_i | gold_j)
                    if gold_i | gold_j else 0.0
                ),
                "question_a": questions[qid],
                "question_b": questions[qj],
            })
    rows.sort(key=lambda r: (-r["embedding_cosine"], -r["lexical_cosine"], r["qid_a"], r["qid_b"]))
    return rows


def graph_threshold_report(qids, pair_rows, threshold):
    uf = UnionFind(qids)
    edges = [r for r in pair_rows if r["embedding_cosine"] >= threshold]
    for r in edges:
        uf.union(r["qid_a"], r["qid_b"])
    comps = [c for c in uf.components() if len(c) > 1]
    sizes = sorted((len(c) for c in comps), reverse=True)
    return {
        "threshold": threshold,
        "edge_count": len(edges),
        "component_count_gt1": len(comps),
        "queries_in_components_gt1": sum(sizes),
        "max_component_size": max(sizes, default=1),
        "component_size_p95": (
            float(np.quantile(sizes, .95)) if sizes else 1.0
        ),
        "edge_same_strict_text_rate": (
            sum(r["same_strict_text"] for r in edges) / len(edges) if edges else 0.0
        ),
        "edge_gold_overlap_rate": (
            sum(r["gold_overlap_count"] > 0 for r in edges) / len(edges) if edges else 0.0
        ),
        "edge_lexical_median": (
            float(np.median([r["lexical_cosine"] for r in edges])) if edges else 0.0
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    snapshot_manifest = verify_snapshot()
    model_manifest = verify_model()
    qids, questions, answers = load_train()

    if len(qids) != 7000:
        raise RuntimeError(f"Expected 7000 train queries, got {len(qids)}")
    if any(not q for q in questions.values()):
        raise RuntimeError("Empty train question after conservative preprocessing.")
    if any(not isinstance(a, list) or not a for a in answers.values()):
        raise RuntimeError("Every raw official train query must retain >=1 raw gold ID.")

    print("=== corpus/document audit ===", flush=True)
    corpus = audit_contexts(answers)

    print("=== query duplicate audit ===", flush=True)
    exact_groups = duplicate_groups(qids, questions, strict_match_key)
    punctuation_groups = duplicate_groups(qids, questions, punctuation_match_key)
    accent_groups = duplicate_groups(qids, questions, accent_match_key)

    print("=== fresh semantic duplicate audit ===", flush=True)
    vectors = encode_train(qids, questions, model_manifest, args.batch_size)
    scores, indices = semantic_topk(vectors)

    print("=== lexical matrix ===", flush=True)
    vectorizer, lexical = build_lexical_matrix(qids, questions)
    pair_rows = make_pair_rows(qids, questions, answers, scores, indices, lexical)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "NEAR_DUPLICATE_CANDIDATES.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "qid_a", "qid_b", "embedding_cosine", "lexical_cosine",
            "same_strict_text", "same_punctuation_text", "same_accent_text",
            "gold_overlap_count", "gold_jaccard", "question_a", "question_b",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_rows)

    threshold_report = [
        graph_threshold_report(qids, pair_rows, t)
        for t in SEMANTIC_THRESHOLDS
    ]

    answer_size = Counter(len(v) for v in answers.values())
    gold_freq = Counter(d for vals in answers.values() for d in set(vals))
    max_gold_freq_by_q = {
        qid: max(gold_freq[d] for d in set(answers[qid]))
        for qid in qids
    }

    nearest = scores[:, 0]
    query_rows = []
    for i, qid in enumerate(qids):
        query_rows.append({
            "qid": qid,
            "question": questions[qid],
            "answer_size": len(answers[qid]),
            "max_gold_frequency": max_gold_freq_by_q[qid],
            "nearest_embedding_cosine": float(nearest[i]),
            "nearest_qid": qids[int(indices[i, 0])],
            "word_count": len(questions[qid].split()),
        })
    with (OUT / "QUERY_AUDIT.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(query_rows[0]))
        w.writeheader()
        w.writerows(query_rows)

    primary_population = {
        "policy": "raw_official_literal_doc_ids_v1",
        "query_count": len(qids),
        "all_raw_answers_nonempty": all(bool(answers[q]) for q in qids),
        "drop_empty_gold_queries": False,
        "canonicalize_duplicate_doc_ids": False,
        "rationale": (
            "Primary OOF should mirror official literal-ID evaluation. "
            "Corpus/data-quality exceptions are reported as slices, not silently removed."
        ),
    }

    report = {
        "schema_version": "dsc2026.endgame.stage00b_evaluation_population.v1",
        "status": "COMPLETE_NO_FOLDS_CREATED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_fingerprint": snapshot_manifest["content_fingerprint"],
        "model_sha": model_manifest["resolved_revision_sha"],
        "primary_population_contract": primary_population,
        "train": {
            "query_count": len(qids),
            "answer_size_counts": dict(sorted(answer_size.items())),
            "unique_gold_doc_count": len(gold_freq),
            "gold_doc_frequency": {
                "min": min(gold_freq.values()),
                "median": float(np.median(list(gold_freq.values()))),
                "p95": float(np.quantile(list(gold_freq.values()), .95)),
                "max": max(gold_freq.values()),
            },
        },
        "corpus": corpus,
        "query_duplicates": {
            "strict_group_count": len(exact_groups),
            "strict_groups": exact_groups,
            "punctuation_group_count": len(punctuation_groups),
            "punctuation_groups": punctuation_groups,
            "accent_group_count": len(accent_groups),
            "accent_groups": accent_groups,
        },
        "semantic_near_duplicate_audit": {
            "topk_per_query": TOPK_NEIGHBORS,
            "candidate_pair_count_ge_0.90": len(pair_rows),
            "thresholds": threshold_report,
            "nearest_neighbor_cosine": {
                "min": float(nearest.min()),
                "p25": float(np.quantile(nearest, .25)),
                "median": float(np.median(nearest)),
                "p75": float(np.quantile(nearest, .75)),
                "p90": float(np.quantile(nearest, .90)),
                "p95": float(np.quantile(nearest, .95)),
                "p99": float(np.quantile(nearest, .99)),
                "max": float(nearest.max()),
            },
        },
        "fold_creation": {
            "created": False,
            "reason": (
                "Near-duplicate thresholds and grouping policy must be selected "
                "after inspecting this forensic report/candidate edge table."
            ),
        },
    }
    (OUT / "EVALUATION_POPULATION_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 00B — Evaluation Population Forensics",
        "",
        "**No folds are created in this stage.**",
        "",
        "## Primary population proposal",
        "",
        f"- Raw official train queries: **{len(qids)}**",
        "- Raw official literal document IDs preserved: **yes**",
        "- Duplicate document IDs canonicalized: **no**",
        "- Empty-gold queries dropped: **no**",
        "",
        "## Corpus audit",
        "",
        f"- Documents: **{corpus['document_count']}**",
        f"- Empty passages: **{corpus['empty_passage_count']}**",
        f"- Exact duplicate-passage groups: **{corpus['exact_duplicate_passage_group_count']}**",
        f"- Missing gold document IDs: **{corpus['missing_gold_doc_count']}**",
        f"- Queries touching empty gold docs: **{len(corpus['queries_with_empty_gold'])}**",
        f"- Queries touching duplicate-content gold docs: **{len(corpus['queries_with_duplicate_content_gold'])}**",
        "",
        "## Query duplicates",
        "",
        f"- Strict duplicate groups: **{len(exact_groups)}**",
        f"- Punctuation-fold duplicate groups: **{len(punctuation_groups)}**",
        f"- Accent-fold duplicate groups: **{len(accent_groups)}**",
        "",
        "## Near-duplicate threshold sensitivity",
        "",
        "| Emb cosine | Edges | Components | Queries grouped | Max comp | Gold-overlap edge rate | Median lexical |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in threshold_report:
        lines.append(
            f"| {x['threshold']:.2f} | {x['edge_count']} | "
            f"{x['component_count_gt1']} | {x['queries_in_components_gt1']} | "
            f"{x['max_component_size']} | {x['edge_gold_overlap_rate']:.2%} | "
            f"{x['edge_lexical_median']:.4f} |"
        )
    lines += [
        "",
        "## Next decision",
        "",
        "Inspect `NEAR_DUPLICATE_CANDIDATES.csv` and choose the grouping rule.",
        "Only after that will Stage 00C materialize the new ENDGAME folds.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "query_count": len(qids),
        "document_count": corpus["document_count"],
        "strict_duplicate_groups": len(exact_groups),
        "near_duplicate_pairs_ge_0.90": len(pair_rows),
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
