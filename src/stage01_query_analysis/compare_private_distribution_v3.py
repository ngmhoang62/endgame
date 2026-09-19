#!/usr/bin/env python
"""Stage 01B v3.1 — fresh-model private distribution diagnostics.

Canonical embedding representation:
1. SentenceTransformer.encode(..., normalize_embeddings=True)
2. cast to float32
3. explicit float32 L2 normalization
4. cache only the final float32 vectors

Run:
    python src/stage01_query_analysis/compare_private_distribution_v3.py \
      --sota-root D:/Study/DSC2026/sota
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.query_preprocessing import model_text, strict_match_key

os.environ["HF_HOME"] = str(ROOT / "cache" / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "cache" / "huggingface" / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

PREFLIGHT = ROOT / "reports/stage01_query_preflight/QUERY_PREFLIGHT.json"
MODEL_MANIFEST = ROOT / "reports/stage00_model_materialization/vietlegal-e5/MODEL_MANIFEST.json"
MODEL_DIR = ROOT / "models/vietlegal-e5"
OUT = ROOT / "reports/stage01_private_distribution_v3"
CACHE = ROOT / "cache/stage01_query_analysis/private_distribution_v3"

FINAL_NORM_MAX_ERROR = 2e-6

HIST_CAL = {
    "HIST_CAL_A": (750, 850),
    "HIST_CAL_B": (1250, 1350),
    "HIST_CAL_C": (1350, 1450),
    "HIST_CAL_D": (1450, 1750),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_f32(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("Invalid embedding norms before f32 normalization.")
    x = x / norms
    final = np.linalg.norm(x, axis=1)
    err = float(np.max(np.abs(final - 1.0)))
    if err > FINAL_NORM_MAX_ERROR:
        raise RuntimeError(f"Float32 normalization failed: max error={err:.8g}")
    return x


def load_questions(path: Path, require_answer=False):
    raw = read_json(path)
    out = {}
    for qid, row in raw.items():
        if isinstance(row, str):
            if require_answer:
                continue
            text = row
        else:
            if require_answer and not row.get("answer"):
                continue
            text = row.get("question", "")
        text = model_text(text)
        if text:
            out[str(qid)] = text
    return out


def verify_contract(private_path: Path, train_path: Path, public_path: Path):
    if not PREFLIGHT.exists():
        raise RuntimeError("Missing Stage 01B query preflight.")
    pre = read_json(PREFLIGHT)
    if pre.get("status") != "PASS":
        raise RuntimeError("Query preflight is not PASS.")

    for name, path in {
        "PRIVATE": private_path,
        "TRAIN": train_path,
        "PUBLIC": public_path,
    }.items():
        current = sha256(path)
        frozen = pre["inputs"][name]["sha256"]
        if current != frozen:
            raise RuntimeError(
                f"{name} input changed after preflight: {frozen} -> {current}"
            )

    if not MODEL_MANIFEST.exists():
        raise RuntimeError("Fresh model manifest missing. Run materialize_hf_model.py first.")
    model_manifest = read_json(MODEL_MANIFEST)
    if model_manifest.get("status") != "PASS":
        raise RuntimeError("Fresh model materialization did not PASS.")
    if model_manifest.get("historical_model_reused") is not False:
        raise RuntimeError("Model manifest does not prove fresh materialization.")
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Fresh ENDGAME model directory missing: {MODEL_DIR}")

    emb_contract = model_manifest.get("embedding_contract", {})
    if emb_contract.get("canonical_postprocess") != "astype_float32_then_explicit_l2_normalize":
        raise RuntimeError("Model manifest lacks ENDGAME v3.1 float32 normalization contract.")

    return pre, model_manifest


def exact_overlap(left, right):
    ref = {}
    for qid, text in right.items():
        ref.setdefault(strict_match_key(text), []).append(qid)
    hits = []
    for qid, text in left.items():
        key = strict_match_key(text)
        if key in ref:
            hits.append((qid, ref[key]))
    return len(hits), hits


def lexical_space(private, train, public):
    texts = list(private.values()) + list(train.values()) + list(public.values())
    n_private, n_train = len(private), len(train)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    x = vectorizer.fit_transform(texts)
    a = n_private
    b = a + n_train
    return x[:a], x[a:b], x[b:], len(vectorizer.vocabulary_)


def top1_sparse(target, ref):
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(ref)
    dist, idx = nn.kneighbors(target)
    return (1.0 - dist[:, 0]).astype(np.float32), idx[:, 0]


def load_sentence_transformer(model_manifest):
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
        processor_kwargs={"fix_mistral_regex": True},
    )

    expected_dim = int(model_manifest["model_contract"]["embedding_dimension"])
    expected_max = int(model_manifest["model_contract"]["max_seq_length"])
    if int(model.get_sentence_embedding_dimension()) != expected_dim:
        raise RuntimeError("Embedding dimension drift from materialization manifest.")
    if int(model.max_seq_length) != expected_max:
        raise RuntimeError("max_seq_length drift from materialization manifest.")
    return model


def cached_encode(name, qmap, model, model_manifest, input_sha, batch_size):
    CACHE.mkdir(parents=True, exist_ok=True)
    npy = CACHE / f"{name}.f32.npy"
    meta_path = CACHE / f"{name}.json"

    contract = {
        "schema": "endgame.embedding_cache.v3.1",
        "population": name,
        "input_sha256": input_sha,
        "qids": list(qmap),
        "model_hub_sha": model_manifest["resolved_revision_sha"],
        "model_content_fingerprint": model_manifest["content_fingerprint"],
        "interface": "SentenceTransformer.encode",
        "prefix": "query: ",
        "sentence_transformer_normalize_embeddings": True,
        "canonical_postprocess": "astype_float32_then_explicit_l2_normalize",
        "fix_mistral_regex": True,
        "max_seq_length": int(model.max_seq_length),
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if npy.exists() and meta_path.exists():
        old = read_json(meta_path)
        if old.get("contract_hash") == contract_hash:
            arr = np.asarray(np.load(npy), dtype=np.float32)
            if arr.shape[0] == len(qmap):
                norms = np.linalg.norm(arr, axis=1)
                if float(np.max(np.abs(norms - 1.0))) <= FINAL_NORM_MAX_ERROR:
                    print(f"[embed] v3.1 cache hit: {name}", flush=True)
                    return arr
                print(f"[embed] rejecting stale/noncanonical cache: {name}", flush=True)

    texts = ["query: " + x for x in qmap.values()]
    print(f"[embed] fresh v3.1 encode: {name} n={len(texts)}", flush=True)
    raw = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    raw = np.asarray(raw, dtype=np.float32)
    raw_norms = np.linalg.norm(raw, axis=1)
    raw_max_error = float(np.max(np.abs(raw_norms - 1.0)))

    arr = normalize_f32(raw)
    final_norms = np.linalg.norm(arr, axis=1)
    final_max_error = float(np.max(np.abs(final_norms - 1.0)))

    np.save(npy, arr)
    meta_path.write_text(
        json.dumps(
            {
                "contract_hash": contract_hash,
                "contract": contract,
                "shape": list(arr.shape),
                "raw_norm_min": float(raw_norms.min()),
                "raw_norm_max": float(raw_norms.max()),
                "raw_max_abs_norm_error": raw_max_error,
                "final_norm_min": float(final_norms.min()),
                "final_norm_max": float(final_norms.max()),
                "final_max_abs_norm_error": final_max_error,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return arr


def top1_dense(a, b, block=256):
    bt = np.ascontiguousarray(b.T)
    sims = np.empty(len(a), dtype=np.float32)
    idx = np.empty(len(a), dtype=np.int64)
    for start in range(0, len(a), block):
        x = np.ascontiguousarray(a[start:start + block])
        scores = x @ bt
        local_idx = scores.argmax(axis=1)
        sims[start:start + len(local_idx)] = scores[np.arange(len(local_idx)), local_idx]
        idx[start:start + len(local_idx)] = local_idx
    return sims, idx


def describe(a):
    x = np.asarray(a, dtype=np.float64)
    return {
        "min": float(x.min()),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
        "ge_0.90": int((x >= 0.90).sum()),
        "ge_0.95": int((x >= 0.95).sum()),
        "ge_0.99": int((x >= 0.99).sum()),
    }


def cluster_private(vectors, qids, questions):
    candidates = [8, 12, 16, 24]
    scores, runs = {}, {}
    for k in candidates:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=2026, n_init=10, batch_size=256
        )
        labels = km.fit_predict(vectors)
        score = silhouette_score(
            vectors,
            labels,
            metric="cosine",
            sample_size=min(1200, len(vectors)),
            random_state=2026,
        )
        scores[str(k)] = float(score)
        runs[k] = (km, labels)

    best = max(candidates, key=lambda k: scores[str(k)])
    km, labels = runs[best]
    centers = normalize_f32(km.cluster_centers_.astype(np.float32))

    clusters = []
    for cid in range(best):
        members = np.where(labels == cid)[0]
        sims = vectors[members] @ centers[cid]
        representative = members[np.argsort(-sims)[:5]]
        clusters.append({
            "cluster": cid,
            "size": int(len(members)),
            "representatives": [
                {
                    "qid": qids[i],
                    "question": questions[qids[i]],
                    "centroid_cosine": float(vectors[i] @ centers[cid]),
                }
                for i in representative
            ],
        })
    clusters.sort(key=lambda row: -row["size"])
    return {"selected_k": best, "silhouette": scores, "clusters": clusters}, labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sota-root", type=Path, default=Path("D:/Study/DSC2026/sota"))
    p.add_argument("--private", type=Path, default=ROOT / "private-official.json")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    sota = args.sota_root.resolve()
    data = sota / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path = args.private.resolve()
    train_path = data / "train.json"
    public_path = data / "public-official.json"

    _, model_manifest = verify_contract(private_path, train_path, public_path)

    private = load_questions(private_path)
    train = load_questions(train_path)
    train_eval = load_questions(train_path, require_answer=True)
    public = load_questions(public_path)

    eval_qids = list(train_eval)
    hist = {}
    for name, (lo, hi) in HIST_CAL.items():
        qids = eval_qids[lo:hi]
        hist[name] = {qid: train_eval[qid] for qid in qids}

    expected_hist = {
        "HIST_CAL_A": 100, "HIST_CAL_B": 100,
        "HIST_CAL_C": 100, "HIST_CAL_D": 300,
    }
    if {k: len(v) for k, v in hist.items()} != expected_hist:
        raise RuntimeError("Historical CAL coordinate contract mismatch.")

    OUT.mkdir(parents=True, exist_ok=True)

    private_lex, train_lex, public_lex, vocab = lexical_space(private, train, public)
    train_index = {qid: i for i, qid in enumerate(train)}
    lexical_refs = {"TRAIN": (train, train_lex), "PUBLIC": (public, public_lex)}
    for name, qmap in hist.items():
        rows = np.asarray([train_index[qid] for qid in qmap], dtype=np.int64)
        lexical_refs[name] = (qmap, train_lex[rows])

    lexical = {}
    for name, (qmap, matrix) in lexical_refs.items():
        sims, idx = top1_sparse(private_lex, matrix)
        lexical[name] = (sims, idx, list(qmap))

    model = load_sentence_transformer(model_manifest)
    private_vec = cached_encode("private", private, model, model_manifest, sha256(private_path), args.batch_size)
    train_vec = cached_encode("train", train, model, model_manifest, sha256(train_path), args.batch_size)
    public_vec = cached_encode("public", public, model, model_manifest, sha256(public_path), args.batch_size)

    embedding_refs = {"TRAIN": (train, train_vec), "PUBLIC": (public, public_vec)}
    for name, qmap in hist.items():
        rows = np.asarray([train_index[qid] for qid in qmap], dtype=np.int64)
        embedding_refs[name] = (qmap, train_vec[rows])

    embedding = {}
    for name, (qmap, matrix) in embedding_refs.items():
        sims, idx = top1_dense(private_vec, matrix)
        embedding[name] = (sims, idx, list(qmap))

    clusters, cluster_labels = cluster_private(private_vec, list(private), private)
    (OUT / "PRIVATE_CLUSTERS.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exact = {
        name: exact_overlap(private, qmap)[0]
        for name, qmap in {"TRAIN": train, "PUBLIC": public, **hist}.items()
    }

    report = {
        "schema_version": "dsc2026.endgame.stage01b_private_distribution.v3.1",
        "status": "COMPLETE_FRESH_MODEL_DIAGNOSTIC",
        "private_labels_used": False,
        "evaluation_protocol_claim": "NONE; diagnostic only",
        "supersedes_embedding_results_from": "stage01_private_distribution/v2",
        "preflight_sha256": sha256(PREFLIGHT),
        "model_manifest_sha256": sha256(MODEL_MANIFEST),
        "model": {
            "id": model_manifest["model_id"],
            "resolved_revision_sha": model_manifest["resolved_revision_sha"],
            "content_fingerprint": model_manifest["content_fingerprint"],
            "local_path": str(MODEL_DIR.resolve()),
            "historical_model_reused": False,
            "interface": "SentenceTransformer",
            "fix_mistral_regex": True,
            "query_prefix": "query: ",
            "sentence_transformer_normalize_embeddings": True,
            "canonical_postprocess": "astype_float32_then_explicit_l2_normalize",
        },
        "population": {
            "PRIVATE": len(private), "TRAIN": len(train), "PUBLIC": len(public),
            **{name: len(qmap) for name, qmap in hist.items()},
        },
        "lexical_vocabulary_size": vocab,
        "exact_text_overlap": exact,
        "nearest_similarity": {
            name: {
                "lexical": describe(lexical[name][0]),
                "embedding": describe(embedding[name][0]),
            }
            for name in lexical
        },
        "cluster_summary": {
            "selected_k": clusters["selected_k"],
            "silhouette": clusters["silhouette"],
            "sizes": {str(row["cluster"]): row["size"] for row in clusters["clusters"]},
        },
    }
    (OUT / "PRIVATE_DISTRIBUTION_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = ["private_qid", "question", "cluster_id"]
    for name in lexical_refs:
        tag = name.lower()
        fields += [
            f"{tag}_lex_qid", f"{tag}_lex_sim", f"{tag}_lex_question",
            f"{tag}_emb_qid", f"{tag}_emb_sim", f"{tag}_emb_question",
        ]

    with (OUT / "PRIVATE_NEAREST_NEIGHBORS.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_i, (qid, question) in enumerate(private.items()):
            row = {
                "private_qid": qid,
                "question": question,
                "cluster_id": int(cluster_labels[row_i]),
            }
            for name in lexical_refs:
                tag = name.lower()
                ls, li, lqids = lexical[name]
                es, ei, eqids = embedding[name]
                lqid = lqids[int(li[row_i])]
                eqid = eqids[int(ei[row_i])]
                row.update({
                    f"{tag}_lex_qid": lqid,
                    f"{tag}_lex_sim": f"{float(ls[row_i]):.8f}",
                    f"{tag}_lex_question": lexical_refs[name][0][lqid],
                    f"{tag}_emb_qid": eqid,
                    f"{tag}_emb_sim": f"{float(es[row_i]):.8f}",
                    f"{tag}_emb_question": embedding_refs[name][0][eqid],
                })
            writer.writerow(row)

    lines = [
        "# Stage 01B v3.1 — Private Distribution Diagnostic",
        "",
        "**Fresh ENDGAME model only. Diagnostic, not evaluation protocol.**",
        "",
        f"- Model: `{model_manifest['model_id']}`",
        f"- Hub SHA: `{model_manifest['resolved_revision_sha']}`",
        "- Historical model/cache reused: **false**",
        "- `fix_mistral_regex=True`: **enabled**",
        "- Canonical vector postprocess: **explicit float32 L2 normalization**",
        "",
        "## Nearest-neighbor summary",
        "",
        "| Reference | Lexical median | Embedding median | Embedding >= .95 |",
        "|---|---:|---:|---:|",
    ]
    for name in lexical:
        x = report["nearest_similarity"][name]
        lines.append(
            f"| {name} | {x['lexical']['median']:.4f} | "
            f"{x['embedding']['median']:.4f} | {x['embedding']['ge_0.95']} |"
        )
    lines += [
        "",
        "## Scientific status",
        "",
        "- Stage 01B v2 query preflight remains valid.",
        "- Stage 01B v2 exact/lexical diagnostics remain valid.",
        "- Stage 01B v2 embedding similarities and clusters remain invalidated.",
        "- This v3.1 report is the authoritative semantic-distribution diagnostic.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "out": str(OUT),
        "model_sha": model_manifest["resolved_revision_sha"],
        "cluster_k": clusters["selected_k"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
