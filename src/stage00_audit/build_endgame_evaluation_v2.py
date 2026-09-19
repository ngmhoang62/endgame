#!/usr/bin/env python
"""Stage 00C v2 — rebuild retrievable evaluation universe and freeze primary folds.

Key scientific decision:
- Empty-passage documents are impossible for a content retriever to retrieve
  meaningfully, so they are excluded from the retrieval universe.
- Gold occurrences pointing to empty-passage docs are removed.
- Queries with no remaining retrievable gold are non-evaluable and excluded
  from PRIMARY OOF.
- Exact duplicate-content documents are NOT deduplicated. Literal document IDs
  remain distinct because the official scorer is ID-literal and the corpus
  contains only five duplicate-excluded IDs historically, so deduplication gives
  negligible efficiency benefit while introducing alias-risk.

Expected result from the frozen official snapshot:
- RAW_TRAIN_7000
- RAW_CORPUS_8532
- RETRIEVAL_CORPUS_8512 (= 8532 - 20 empty passages)
- PRIMARY_EVAL_6991
- NON_EVALUABLE_9

No historical LegalIR exclusion artifact is trusted to build this universe.
Everything is derived again from raw ENDGAME data.

Leakage grouping (label-free):
- exact/accent-normalized duplicate, OR
- embedding cosine >= 0.95, OR
- embedding cosine >= 0.94 AND lexical cosine >= 0.80

Run:
    python src/stage00_audit/build_endgame_evaluation_v2.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "official_v1"
FORENSICS = ROOT / "reports" / "stage00b_evaluation_population"
OUT_DATA = ROOT / "data" / "evaluation_v2"
OUT_REPORT = ROOT / "reports" / "stage00c_evaluation_protocol_v2"

N_SPLITS = 5
DEFAULT_SEED_SCAN = 512

EMB_HIGH = 0.95
EMB_BORDER = 0.94
LEX_BORDER = 0.80

EXPECTED_RAW_QUERIES = 7000
EXPECTED_RAW_DOCS = 8532
EXPECTED_EMPTY_DOCS = 20
EXPECTED_RETRIEVAL_DOCS = 8512
EXPECTED_PRIMARY_QUERIES = 6991
EXPECTED_NON_EVALUABLE = 9

# Gates are deliberately tight but not absurdly exact.
SIZE_MAX_REL_DEV_GATE = 0.012
ANSWER_RATE_MAX_DEV_GATE = 0.006
FREQ_RATE_MAX_DEV_GATE = 0.012
LENGTH_RATE_MAX_DEV_GATE = 0.015
DENSITY_RATE_MAX_DEV_GATE = 0.015


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x):
        r = x
        while r != self.parent[r]:
            r = self.parent[r]
        while x != r:
            nxt = self.parent[x]
            self.parent[x] = r
            x = nxt
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self):
        d = defaultdict(list)
        for x in self.parent:
            d[self.find(x)].append(x)
        return list(d.values())


def load_raw_train():
    raw = read_json(DATA / "train.json")
    qids = list(map(str, raw.keys()))
    questions = {str(q): str(row.get("question", "")).strip() for q, row in raw.items()}
    answers = {str(q): [str(x) for x in row.get("answer", [])] for q, row in raw.items()}
    if len(qids) != EXPECTED_RAW_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_RAW_QUERIES} raw queries, got {len(qids)}")
    if any(not answers[q] for q in qids):
        raise RuntimeError("Raw official train contains an empty answer list.")
    return qids, questions, answers


def rebuild_retrieval_corpus_and_labels(raw_qids, raw_answers):
    context_dir = DATA / "selected-contexts"
    paths = sorted(context_dir.glob("context_*.json"))
    if len(paths) != EXPECTED_RAW_DOCS:
        raise RuntimeError(
            f"Expected {EXPECTED_RAW_DOCS} context files, got {len(paths)}. "
            "selected-contexts is local-only but must exist for Stage00C."
        )

    docs = {}
    empty_ids = set()
    passage_hash_groups = defaultdict(list)
    corpus_rows = []

    for i, path in enumerate(paths, 1):
        row = read_json(path)
        doc_id = str(row.get("id", path.stem.replace("context_", "", 1)))
        if doc_id in docs:
            raise RuntimeError(f"Duplicate raw doc ID: {doc_id}")
        passage = str(row.get("passage") or "")
        name = str(row.get("name") or "")
        link = str(row.get("link") or "")
        docs[doc_id] = {"passage": passage, "name": name, "link": link}
        if passage.strip():
            ph = hashlib.sha256(passage.encode("utf-8")).hexdigest()
            passage_hash_groups[ph].append(doc_id)
            corpus_rows.append({
                "id": doc_id,
                "name": name,
                "link": link,
                "passage": passage,
                "source_file": path.name,
            })
        else:
            empty_ids.add(doc_id)

        if i % 1500 == 0 or i == len(paths):
            print(f"[rebuild-corpus] {i}/{len(paths)}", flush=True)

    if len(empty_ids) != EXPECTED_EMPTY_DOCS:
        raise RuntimeError(
            f"Expected {EXPECTED_EMPTY_DOCS} empty docs, got {len(empty_ids)}"
        )
    if len(corpus_rows) != EXPECTED_RETRIEVAL_DOCS:
        raise RuntimeError(
            f"Expected {EXPECTED_RETRIEVAL_DOCS} retrievable docs, got {len(corpus_rows)}"
        )

    duplicate_groups = [
        sorted(ids) for ids in passage_hash_groups.values() if len(ids) > 1
    ]
    duplicate_groups.sort(key=lambda g: (-len(g), g))

    missing_gold = sorted(
        {d for vals in raw_answers.values() for d in vals} - set(docs)
    )
    if missing_gold:
        raise RuntimeError(f"Raw gold IDs missing from corpus: {missing_gold[:20]}")

    primary_golds = {}
    non_evaluable = {}
    affected_evaluable = {}
    removed_occurrences = 0

    for qid in raw_qids:
        raw = list(raw_answers[qid])
        kept = [d for d in raw if d not in empty_ids]
        removed = [d for d in raw if d in empty_ids]
        removed_occurrences += len(removed)

        if kept:
            primary_golds[qid] = kept
            if removed:
                affected_evaluable[qid] = {
                    "raw_gold_ids": raw,
                    "primary_gold_ids": kept,
                    "removed_empty_gold_ids": removed,
                }
        else:
            non_evaluable[qid] = {
                "raw_gold_ids": raw,
                "removed_empty_gold_ids": removed,
                "reason": "all_gold_documents_have_empty_passage",
            }

    if len(primary_golds) != EXPECTED_PRIMARY_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY_QUERIES} primary evaluable queries, "
            f"got {len(primary_golds)}"
        )
    if len(non_evaluable) != EXPECTED_NON_EVALUABLE:
        raise RuntimeError(
            f"Expected {EXPECTED_NON_EVALUABLE} non-evaluable queries, "
            f"got {len(non_evaluable)}"
        )

    return {
        "docs": docs,
        "retrieval_rows": corpus_rows,
        "empty_ids": sorted(empty_ids),
        "duplicate_groups": duplicate_groups,
        "primary_golds": primary_golds,
        "non_evaluable": non_evaluable,
        "affected_evaluable": affected_evaluable,
        "removed_gold_occurrences": removed_occurrences,
    }


def write_retrieval_corpus_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_query_audit():
    path = FORENSICS / "QUERY_AUDIT.csv"
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row["qid"])] = {
                "nearest_embedding_cosine": float(row["nearest_embedding_cosine"]),
                "word_count": int(row["word_count"]),
            }
    return rows


def load_pairs():
    path = FORENSICS / "NEAR_DUPLICATE_CANDIDATES.csv"
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "qid_a": str(row["qid_a"]),
                "qid_b": str(row["qid_b"]),
                "embedding_cosine": float(row["embedding_cosine"]),
                "lexical_cosine": float(row["lexical_cosine"]),
                "same_accent_text": int(row["same_accent_text"]),
                # audit-only fields; NEVER used for edge selection
                "gold_overlap_count": int(row["gold_overlap_count"]),
            })
    return rows


def build_leakage_groups(primary_qids, forensic, pair_rows):
    allowed = set(primary_qids)
    uf = UnionFind(primary_qids)

    for group in forensic["query_duplicates"]["accent_groups"]:
        members = [str(q) for q in group if str(q) in allowed]
        for q in members[1:]:
            uf.union(members[0], q)

    selected_edges = []
    for row in pair_rows:
        a, b = row["qid_a"], row["qid_b"]
        if a not in allowed or b not in allowed:
            continue
        e, l = row["embedding_cosine"], row["lexical_cosine"]
        selected = (
            row["same_accent_text"] == 1
            or e >= EMB_HIGH
            or (e >= EMB_BORDER and l >= LEX_BORDER)
        )
        if selected:
            uf.union(a, b)
            selected_edges.append(row)

    components = uf.components()
    order = {q: i for i, q in enumerate(primary_qids)}
    components = [sorted(c, key=lambda q: order[q]) for c in components]
    components.sort(key=lambda c: order[c[0]])

    group_by_qid = {}
    for i, comp in enumerate(components):
        gid = f"g{i:05d}"
        for q in comp:
            group_by_qid[q] = gid
    gt1 = [c for c in components if len(c) > 1]
    return group_by_qid, components, gt1, selected_edges


def answer_bucket(n):
    return "a1" if n == 1 else ("a2" if n == 2 else "a3p")


def freq_bucket(n):
    if n == 1:
        return "f1"
    if n <= 3:
        return "f2_3"
    if n <= 7:
        return "f4_7"
    if n <= 15:
        return "f8_15"
    return "f16p"


def density_bucket(x):
    if x < 0.60:
        return "d_lt60"
    if x < 0.75:
        return "d_60_75"
    if x < 0.90:
        return "d_75_90"
    return "d_90p"


def length_bounds(values):
    q = np.quantile(np.asarray(values, dtype=np.float64), [.25, .50, .75])
    return [int(math.floor(x)) for x in q]


def length_bucket(n, b):
    if n <= b[0]:
        return "l_q1"
    if n <= b[1]:
        return "l_q2"
    if n <= b[2]:
        return "l_q3"
    return "l_q4"


def category_rates(qids, mapping):
    c = Counter(mapping[q] for q in qids)
    n = len(qids)
    return {k: v / n for k, v in c.items()}


def rate_dev(folds, mapping, global_rates):
    worst = 0.0
    by_fold = {}
    for name, ids in folds.items():
        rates = category_rates(ids, mapping)
        dev = max(
            abs(rates.get(k, 0.0) - global_rates.get(k, 0.0))
            for k in global_rates
        )
        by_fold[name] = dev
        worst = max(worst, dev)
    return worst, by_fold


def collapse_rare_strata(labels, min_count=N_SPLITS):
    counts = Counter(labels)
    out = []
    for label in labels:
        if counts[label] >= min_count:
            out.append(label)
        else:
            # Preserve answer-cardinality signal, collapse only sparse frequency combo.
            out.append(label.split("|", 1)[0] + "|rare")
    counts2 = Counter(out)
    too_small = {k: v for k, v in counts2.items() if v < min_count}
    if too_small:
        out = [x.split("|", 1)[0] for x in out]
    return out


def make_folds(qids, strata, groups, seed):
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=seed
    )
    X = np.zeros((len(qids), 1), dtype=np.uint8)
    y = np.asarray(strata, dtype=object)
    g = np.asarray(groups, dtype=object)
    folds = {}
    for i, (_, val_idx) in enumerate(splitter.split(X, y, groups=g)):
        val_set = set(map(int, val_idx))
        folds[f"fold_{i}"] = [q for j, q in enumerate(qids) if j in val_set]
    return folds


def score_folds(folds, qids, maps):
    target = len(qids) / N_SPLITS
    size_devs = {
        name: abs(len(ids) - target) / target
        for name, ids in folds.items()
    }
    out = {
        "size_max_relative_deviation": max(size_devs.values()),
        "size_relative_deviation_by_fold": size_devs,
    }
    objective = 12.0 * out["size_max_relative_deviation"]
    weights = {"answer": 8.0, "freq": 4.0, "length": 2.0, "density": 2.0}
    for key, w in weights.items():
        global_rates = category_rates(qids, maps[key])
        worst, by_fold = rate_dev(folds, maps[key], global_rates)
        out[f"{key}_max_rate_deviation"] = worst
        out[f"{key}_rate_deviation_by_fold"] = by_fold
        out[f"{key}_global_rates"] = global_rates
        objective += w * worst
    out["objective"] = objective
    return out


def verify_group_isolation(folds, group_by_qid):
    fold_for = {}
    for fold, ids in folds.items():
        for q in ids:
            if q in fold_for:
                raise RuntimeError(f"QID repeated across folds: {q}")
            fold_for[q] = fold
    group_folds = defaultdict(set)
    for q, gid in group_by_qid.items():
        group_folds[gid].add(fold_for[q])
    bad = {gid: sorted(v) for gid, v in group_folds.items() if len(v) > 1}
    return fold_for, bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-scan", type=int, default=DEFAULT_SEED_SCAN)
    args = p.parse_args()

    required = [
        DATA / "train.json",
        FORENSICS / "EVALUATION_POPULATION_AUDIT.json",
        FORENSICS / "QUERY_AUDIT.csv",
        FORENSICS / "NEAR_DUPLICATE_CANDIDATES.csv",
        DATA / "selected-contexts",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisite(s):\n" + "\n".join(missing))

    raw_qids, questions, raw_answers = load_raw_train()

    print("=== rebuild retrieval universe from RAW official snapshot ===", flush=True)
    rebuilt = rebuild_retrieval_corpus_and_labels(raw_qids, raw_answers)
    primary_golds = rebuilt["primary_golds"]
    primary_qids = [q for q in raw_qids if q in primary_golds]

    # Build local corpus artifact. Keep duplicate-content IDs distinct.
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DATA / "retrieval_corpus_8512.jsonl"
    write_retrieval_corpus_jsonl(rebuilt["retrieval_rows"], corpus_path)

    # Recompute label-frequency strata from PRIMARY labels, not raw labels.
    gold_frequency = Counter(
        d for q in primary_qids for d in set(primary_golds[q])
    )
    query_audit = load_query_audit()
    forensic = read_json(FORENSICS / "EVALUATION_POPULATION_AUDIT.json")
    pair_rows = load_pairs()

    group_by_qid, all_groups, leakage_groups, selected_edges = build_leakage_groups(
        primary_qids, forensic, pair_rows
    )

    lb = length_bounds([query_audit[q]["word_count"] for q in primary_qids])
    maps = {
        "answer": {q: answer_bucket(len(set(primary_golds[q]))) for q in primary_qids},
        "freq": {
            q: freq_bucket(max(gold_frequency[d] for d in set(primary_golds[q])))
            for q in primary_qids
        },
        "length": {
            q: length_bucket(query_audit[q]["word_count"], lb)
            for q in primary_qids
        },
        "density": {
            q: density_bucket(query_audit[q]["nearest_embedding_cosine"])
            for q in primary_qids
        },
    }

    raw_strata = [f"{maps['answer'][q]}|{maps['freq'][q]}" for q in primary_qids]
    strata = collapse_rare_strata(raw_strata)
    groups = [group_by_qid[q] for q in primary_qids]

    best = None
    top_candidates = []
    for seed in range(args.seed_scan):
        folds = make_folds(primary_qids, strata, groups, seed)
        score = score_folds(folds, primary_qids, maps)
        row = {"seed": seed, "score": score, "folds": folds}
        if best is None or score["objective"] < best["score"]["objective"]:
            best = row
        top_candidates.append(row)

    folds = best["folds"]
    score = best["score"]
    fold_for_qid, bad_groups = verify_group_isolation(folds, group_by_qid)

    gates = {
        "population_6991": len(primary_qids) == EXPECTED_PRIMARY_QUERIES,
        "non_evaluable_9": len(rebuilt["non_evaluable"]) == EXPECTED_NON_EVALUABLE,
        "retrieval_docs_8512": len(rebuilt["retrieval_rows"]) == EXPECTED_RETRIEVAL_DOCS,
        "group_isolation": not bad_groups,
        "fold_coverage": set(fold_for_qid) == set(primary_qids),
        "size_balance": score["size_max_relative_deviation"] <= SIZE_MAX_REL_DEV_GATE,
        "answer_balance": score["answer_max_rate_deviation"] <= ANSWER_RATE_MAX_DEV_GATE,
        "freq_balance": score["freq_max_rate_deviation"] <= FREQ_RATE_MAX_DEV_GATE,
        "length_balance": score["length_max_rate_deviation"] <= LENGTH_RATE_MAX_DEV_GATE,
        "density_balance": score["density_max_rate_deviation"] <= DENSITY_RATE_MAX_DEV_GATE,
    }
    status = "PASS" if all(gates.values()) else "FAIL"

    duplicate_doc_ids = {
        d for group in rebuilt["duplicate_groups"] for d in group
    }
    duplicate_gold_queries = [
        q for q in primary_qids if set(primary_golds[q]) & duplicate_doc_ids
    ]

    special = {
        "single_gold": [q for q in primary_qids if len(set(primary_golds[q])) == 1],
        "multi_gold": [q for q in primary_qids if len(set(primary_golds[q])) > 1],
        "three_plus_gold": [q for q in primary_qids if len(set(primary_golds[q])) >= 3],
        "empty_gold_affected_but_evaluable": sorted(rebuilt["affected_evaluable"]),
        "duplicate_content_gold_literal_ids": duplicate_gold_queries,
        "near_duplicate_group_member": [q for g in leakage_groups for q in g],
        "semantic_dense_ge_0_90": [
            q for q in primary_qids
            if query_audit[q]["nearest_embedding_cosine"] >= 0.90
        ],
        "rare_gold_only": [
            q for q in primary_qids
            if max(gold_frequency[d] for d in set(primary_golds[q])) == 1
        ],
        "frequent_gold_16p": [
            q for q in primary_qids
            if max(gold_frequency[d] for d in set(primary_golds[q])) >= 16
        ],
    }

    # Write authoritative data artifacts.
    (OUT_DATA / "primary_golds_6991.json").write_text(
        json.dumps(primary_golds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "non_evaluable_9.json").write_text(
        json.dumps(rebuilt["non_evaluable"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "empty_gold_affected_evaluable.json").write_text(
        json.dumps(rebuilt["affected_evaluable"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "empty_document_ids.json").write_text(
        json.dumps(rebuilt["empty_ids"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "exact_duplicate_passage_groups.json").write_text(
        json.dumps(rebuilt["duplicate_groups"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "folds_v2.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "qid_to_fold.json").write_text(
        json.dumps(fold_for_qid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "group_id_by_qid.json").write_text(
        json.dumps(group_by_qid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DATA / "stress_slices.json").write_text(
        json.dumps(special, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep retrieval corpus local-only via .gitignore, but fingerprint it.
    corpus_sha = sha256(corpus_path)

    edge_gold_overlap_rate = (
        sum(r["gold_overlap_count"] > 0 for r in selected_edges) / len(selected_edges)
        if selected_edges else 0.0
    )

    contract = {
        "schema_version": "dsc2026.endgame.evaluation_contract.v2",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "populations": {
            "RAW_TRAIN_7000": {
                "query_count": len(raw_qids),
                "label_policy": "untouched official literal IDs",
                "role": "secondary raw-official diagnostic",
            },
            "PRIMARY_EVAL_6991": {
                "query_count": len(primary_qids),
                "label_policy": "drop only gold IDs whose raw passage is empty; preserve all other literal IDs",
                "role": "primary model-development OOF",
            },
            "RAW_CORPUS_8532": {"document_count": len(rebuilt["docs"])},
            "RETRIEVAL_CORPUS_8512": {
                "document_count": len(rebuilt["retrieval_rows"]),
                "policy": "exclude empty-passage docs only; exact duplicate-content IDs preserved",
                "corpus_jsonl_sha256": corpus_sha,
            },
        },
        "document_identity_policy": {
            "literal_ids_preserved": True,
            "exact_duplicate_content_docs_deduplicated": False,
            "alias_mapping": False,
            "rationale": (
                "Official scorer is literal-ID based. Only five historical duplicate exclusions "
                "exist, so retaining IDs avoids alias-risk at negligible compute cost."
            ),
        },
        "empty_passage_policy": {
            "empty_document_count": len(rebuilt["empty_ids"]),
            "gold_occurrences_removed": rebuilt["removed_gold_occurrences"],
            "affected_but_evaluable_queries": len(rebuilt["affected_evaluable"]),
            "non_evaluable_queries": len(rebuilt["non_evaluable"]),
        },
        "primary_metric": {
            "name": "macro_recall_at_5",
            "query_formula": "hits / number_of_primary_retrievable_literal_gold_ids",
            "aggregate": "mean over PRIMARY_EVAL_6991",
        },
        "secondary_metric": {
            "name": "macro_precision_at_5",
            "query_formula": "hits / returned_doc_count for 1..5 returned docs",
        },
        "folds": {
            "count": N_SPLITS,
            "selected_seed": best["seed"],
            "builder": "StratifiedGroupKFold seed scan",
            "seed_scan": args.seed_scan,
            "stratification": "primary answer-cardinality x primary gold-frequency, rare strata collapsed",
            "leakage_rule": {
                "accent_normalized_exact_duplicate": True,
                "embedding_cosine_ge": EMB_HIGH,
                "border": {
                    "embedding_cosine_ge": EMB_BORDER,
                    "lexical_cosine_ge": LEX_BORDER,
                },
                "gold_used_to_select_edge": False,
            },
        },
        "historical_legalir_comparison": {
            "expected_same_primary_query_count": 6991,
            "important_difference": (
                "ENDGAME v2 does NOT exclude/remap exact duplicate-content document IDs."
            ),
        },
    }
    (OUT_DATA / "EVALUATION_CONTRACT.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    # Fold assignment CSV.
    group_sizes = Counter(group_by_qid.values())
    with (OUT_REPORT / "FOLD_ASSIGNMENTS.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "qid", "fold", "group_id", "group_size", "answer_size",
            "answer_bucket", "gold_freq_bucket", "length_bucket",
            "density_bucket", "max_primary_gold_frequency",
            "nearest_embedding_cosine", "question",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in primary_qids:
            w.writerow({
                "qid": q,
                "fold": fold_for_qid[q],
                "group_id": group_by_qid[q],
                "group_size": group_sizes[group_by_qid[q]],
                "answer_size": len(set(primary_golds[q])),
                "answer_bucket": maps["answer"][q],
                "gold_freq_bucket": maps["freq"][q],
                "length_bucket": maps["length"][q],
                "density_bucket": maps["density"][q],
                "max_primary_gold_frequency": max(
                    gold_frequency[d] for d in set(primary_golds[q])
                ),
                "nearest_embedding_cosine": f"{query_audit[q]['nearest_embedding_cosine']:.8f}",
                "question": questions[q],
            })

    fold_stats = {}
    for name, ids in folds.items():
        ids_set = set(ids)
        fold_stats[name] = {
            "n": len(ids),
            "answer_counts": dict(Counter(maps["answer"][q] for q in ids)),
            "freq_counts": dict(Counter(maps["freq"][q] for q in ids)),
            "special_slice_counts": {
                k: len(ids_set & set(v)) for k, v in special.items()
            },
        }

    audit = {
        "schema_version": "dsc2026.endgame.stage00c_eval_v2.audit",
        "status": status,
        "gates": gates,
        "selected_seed": best["seed"],
        "selected_objective": score["objective"],
        "balance": score,
        "fold_sizes": {k: len(v) for k, v in folds.items()},
        "fold_stats": fold_stats,
        "population_audit": {
            "raw_queries": len(raw_qids),
            "primary_queries": len(primary_qids),
            "non_evaluable_queries": len(rebuilt["non_evaluable"]),
            "raw_docs": len(rebuilt["docs"]),
            "empty_docs": len(rebuilt["empty_ids"]),
            "retrieval_docs": len(rebuilt["retrieval_rows"]),
            "duplicate_passage_groups_preserved": len(rebuilt["duplicate_groups"]),
            "empty_gold_affected_but_evaluable": len(rebuilt["affected_evaluable"]),
            "removed_empty_gold_occurrences": rebuilt["removed_gold_occurrences"],
        },
        "leakage_groups": {
            "groups_gt1": len(leakage_groups),
            "queries_in_groups_gt1": sum(len(g) for g in leakage_groups),
            "max_group_size": max(map(len, leakage_groups), default=1),
            "selected_semantic_edges": len(selected_edges),
            "selected_edge_gold_overlap_rate_posthoc_only": edge_gold_overlap_rate,
            "gold_used_for_edge_selection": False,
        },
        "stress_slice_sizes": {k: len(v) for k, v in special.items()},
        "artifacts": {
            "retrieval_corpus_sha256": corpus_sha,
            "primary_golds_sha256": sha256(OUT_DATA / "primary_golds_6991.json"),
            "folds_sha256": sha256(OUT_DATA / "folds_v2.json"),
            "evaluation_contract_sha256": sha256(OUT_DATA / "EVALUATION_CONTRACT.json"),
        },
    }
    (OUT_REPORT / "FOLD_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top = sorted(
        [{"seed": x["seed"], "score": x["score"]} for x in top_candidates],
        key=lambda x: x["score"]["objective"],
    )[:12]
    (OUT_REPORT / "SEED_SEARCH.json").write_text(
        json.dumps(top, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 00C v2 — ENDGAME Evaluation Protocol",
        "",
        f"**Status: {status}**",
        "",
        "## Rebuilt populations",
        "",
        f"- RAW_TRAIN: **{len(raw_qids)}**",
        f"- PRIMARY_EVAL: **{len(primary_qids)}**",
        f"- NON_EVALUABLE: **{len(rebuilt['non_evaluable'])}**",
        f"- RAW_CORPUS: **{len(rebuilt['docs'])}**",
        f"- EMPTY_DOCS excluded: **{len(rebuilt['empty_ids'])}**",
        f"- RETRIEVAL_CORPUS: **{len(rebuilt['retrieval_rows'])}**",
        "",
        "## Identity policy",
        "",
        "- Empty passage docs excluded: **yes**",
        "- Exact duplicate-content docs deduplicated: **no**",
        "- Literal document IDs preserved: **yes**",
        "- Alias/remap labels: **no**",
        "",
        "## Empty-gold impact",
        "",
        f"- Removed empty-gold occurrences: **{rebuilt['removed_gold_occurrences']}**",
        f"- Queries still evaluable after removal: **{len(rebuilt['affected_evaluable'])}**",
        f"- Queries with no retrievable gold left: **{len(rebuilt['non_evaluable'])}**",
        "",
        "## Leakage grouping",
        "",
        f"- Rule: emb >= {EMB_HIGH:.2f} OR (emb >= {EMB_BORDER:.2f} AND lexical >= {LEX_BORDER:.2f}) + exact/accent duplicates",
        "- Gold used for grouping: **no**",
        f"- Groups >1: **{len(leakage_groups)}**",
        f"- Queries in grouped components: **{sum(len(g) for g in leakage_groups)}**",
        f"- Post-hoc edge gold-overlap: **{edge_gold_overlap_rate:.2%}**",
        "",
        "## Fold sizes",
        "",
        "| Fold | N | Single | Multi | Empty-gold-affected | Duplicate-content literal gold | Near-dup members |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(folds):
        st = fold_stats[name]
        single = st["answer_counts"].get("a1", 0)
        multi = st["n"] - single
        ss = st["special_slice_counts"]
        lines.append(
            f"| {name} | {st['n']} | {single} | {multi} | "
            f"{ss['empty_gold_affected_but_evaluable']} | "
            f"{ss['duplicate_content_gold_literal_ids']} | "
            f"{ss['near_duplicate_group_member']} |"
        )

    lines += ["", "## Gates", ""]
    for k, v in gates.items():
        lines.append(f"- {k}: **{'PASS' if v else 'FAIL'}**")

    lines += [
        "",
        "## Historical LegalIR comparison",
        "",
        "- Same intended evaluable-query count: **6991**.",
        "- Same empty-passage principle: **yes**.",
        "- Different duplicate policy: ENDGAME keeps all non-empty duplicate-content IDs literal.",
        "",
    ]
    (OUT_REPORT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": status,
        "raw_queries": len(raw_qids),
        "primary_queries": len(primary_qids),
        "non_evaluable": len(rebuilt["non_evaluable"]),
        "raw_docs": len(rebuilt["docs"]),
        "empty_docs": len(rebuilt["empty_ids"]),
        "retrieval_docs": len(rebuilt["retrieval_rows"]),
        "duplicate_passage_groups_preserved": len(rebuilt["duplicate_groups"]),
        "selected_seed": best["seed"],
        "fold_sizes": {k: len(v) for k, v in folds.items()},
        "gates": gates,
        "out_data": str(OUT_DATA),
        "out_report": str(OUT_REPORT),
    }, ensure_ascii=False, indent=2))

    if status != "PASS":
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
