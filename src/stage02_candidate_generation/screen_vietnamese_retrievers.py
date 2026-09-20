#!/usr/bin/env python
"""Stage 02B4 v1.3 — Vietnamese-first encoder screen on fixed moderate geometry.

Scientific isolation:
- EXACT SAME 179,737 region IDs for every model.
- One vector per region.
- No additional splitting.
- No overlap.
- No metadata/title prefix added by ENDGAME.
- Exact parent score = max cosine over that parent's fixed regions.
- Each model keeps its own documented tokenizer/prompt/pooling/context contract.
- Long regions are truncated, not split, during this MODEL screen.

This deliberately separates "which encoder stack is useful?" from the next
question, "how should we pack/split structure for that encoder?"

Models:
  vietlegal_e5
  vnlegal_lal
  aiteamvn_v1
  aiteamvn_v2
  vietnamese_legal_embedding
  vietlegal_harrier
Run all:
  python src/stage02_candidate_generation/screen_vietnamese_retrievers.py

Run selected:
  python src/stage02_candidate_generation/screen_vietnamese_retrievers.py \
    --models vietlegal_e5,aiteamvn_v2,vietnamese_legal_embedding

Large embeddings/rankings stay under cache/stage02b4_vi_screen/.
Small metrics/complementarity reports go under reports/stage02b4_vi_screen/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.evaluation import official_metrics

HF_ROOT = ROOT / "cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_ROOT / "modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / "cache" / "sentence_transformers")

GEOMETRY = ROOT / "cache" / "stage02b_structure" / "moderate_structure_regions.jsonl"
GEOMETRY_REPORT = ROOT / "reports" / "stage02b0_structure_geometry" / "STRUCTURE_GEOMETRY.json"
RAW = ROOT / "data" / "official_v1"
EVAL = ROOT / "data" / "evaluation_v2"
CACHE = ROOT / "cache" / "stage02b4_vi_screen"
OUT = ROOT / "reports" / "stage02b4_vi_screen"

KS = (1, 5, 10, 20, 50, 100)
TOPK = 100
NORM_TOL = 2e-6

LEGAL_INSTRUCTION = (
    "Given a Vietnamese legal question, retrieve relevant legal passages "
    "that answer the question"
)
LEGAL_Q_PREFIX = f"Instruct: {LEGAL_INSTRUCTION}\nQuery: "


@dataclass(frozen=True)
class Spec:
    key: str
    kind: str
    path: Path
    manifest: Path
    doc_max: int
    query_max: int
    doc_prefix: str = ""
    query_prefix: str = ""
    query_prompt_name: str | None = None
    trust_remote_code: bool = False
    fix_mistral_regex: bool = False
    batch_docs: int = 8
    batch_queries: int = 32


SPECS = {
    "vietlegal_e5": Spec(
        "vietlegal_e5", "st",
        ROOT/"models/vietlegal-e5",
        ROOT/"reports/stage00_model_materialization/vietlegal-e5/MODEL_MANIFEST.json",
        512, 512, "passage: ", "query: ",
        fix_mistral_regex=True, batch_docs=12, batch_queries=64,
    ),
    "vnlegal_lal": Spec(
        "vnlegal_lal", "lal",
        ROOT/"models/retrievers/vnlegal-lal",
        ROOT/"reports/stage02b1_model_materialization/vnlegal_lal/MODEL_MANIFEST.json",
        2048, 2048, "", LEGAL_Q_PREFIX,
        fix_mistral_regex=True, batch_docs=2, batch_queries=16,
    ),
    "aiteamvn_v1": Spec(
        "aiteamvn_v1", "st",
        ROOT/"models/retrievers_vi/aiteamvn-vietnamese-embedding",
        ROOT/"reports/stage02b3_vietnamese_model_materialization/aiteamvn_vietnamese_embedding/MODEL_MANIFEST.json",
        # Model card/training contract is 2048 even though current ST metadata
        # inherits BGE-M3's 8192.
        2048, 2048, batch_docs=4, batch_queries=32,
    ),
    "aiteamvn_v2": Spec(
        "aiteamvn_v2", "st",
        ROOT/"models/retrievers_vi/aiteamvn-vietnamese-embedding-v2",
        ROOT/"reports/stage02b3_vietnamese_model_materialization/aiteamvn_vietnamese_embedding_v2/MODEL_MANIFEST.json",
        # v2 training: 256 query / 2048 passage.
        2048, 256, batch_docs=4, batch_queries=64,
    ),
    "vietnamese_legal_embedding": Spec(
        "vietnamese_legal_embedding", "st",
        ROOT/"models/retrievers_vi/vietnamese-legal-embedding",
        ROOT/"reports/stage02b3_vietnamese_model_materialization/vietnamese_legal_embedding/MODEL_MANIFEST.json",
        512, 512, "passage: ", "query: ",
        trust_remote_code=True, batch_docs=32, batch_queries=128,
    ),
    "vietlegal_harrier": Spec(
        "vietlegal_harrier", "st",
        ROOT/"models/retrievers_vi/vietlegal-harrier-0.6b",
        ROOT/"reports/stage02b3_vietnamese_model_materialization/vietlegal_harrier_0_6b/MODEL_MANIFEST.json",
        512, 512, query_prompt_name="query",
        fix_mistral_regex=True, batch_docs=12, batch_queries=64,
    ),
}


def rj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def normalize_f32(a):
    x = np.asarray(a, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise RuntimeError(f"Invalid embedding array {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n <= 0) or not np.isfinite(n).all():
        raise RuntimeError("Invalid embedding norm")
    x = x / n
    err = float(np.max(np.abs(np.linalg.norm(x, axis=1) - 1.0)))
    if err > NORM_TOL:
        raise RuntimeError(f"Normalization error {err}")
    return x


def load_geometry():
    report = rj(GEOMETRY_REPORT)
    expected_sha = report["artifact"]["regions_sha256"]
    actual_sha = sha256(GEOMETRY)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Geometry hash drift: {actual_sha} != {expected_sha}")

    region_ids, parent_ids, kinds, texts = [], [], [], []
    with GEOMETRY.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            region_ids.append(str(row["region_id"]))
            parent_ids.append(str(row["doc_id"]))
            kinds.append(str(row["kind"]))
            texts.append(str(row["raw_text"]))

    if len(region_ids) != 179737:
        raise RuntimeError(f"Expected 179737 regions, got {len(region_ids)}")
    if len(set(region_ids)) != len(region_ids):
        raise RuntimeError("Duplicate region IDs")

    docs = []
    seen = set()
    for d in parent_ids:
        if d not in seen:
            seen.add(d)
            docs.append(d)
    if len(docs) != 8512:
        raise RuntimeError(f"Expected 8512 parents, got {len(docs)}")

    doc_to_idx = {d: i for i, d in enumerate(docs)}
    parent_index = np.asarray([doc_to_idx[d] for d in parent_ids], dtype=np.int64)
    return {
        "sha": actual_sha,
        "region_ids": region_ids,
        "parent_ids": parent_ids,
        "parent_index": parent_index,
        "doc_ids": docs,
        "kinds": kinds,
        "texts": texts,
    }


def load_eval():
    train = rj(RAW/"train.json")
    golds = {str(q): [str(d) for d in ds] for q, ds in rj(EVAL/"primary_golds_6991.json").items()}
    folds = {k: [str(q) for q in v] for k, v in rj(EVAL/"folds_v2.json").items()}
    stress = {k: [str(q) for q in v] for k, v in rj(EVAL/"stress_slices.json").items()}
    qids = [str(q) for q in train if str(q) in golds]
    questions = {q: str(train[q]["question"]).strip() for q in qids}
    if len(qids) != 6991:
        raise RuntimeError("Primary query count drift")
    return qids, questions, golds, folds, stress




def install_persistent_warning_log():
    """Persist Python/Hugging Face warnings so terminal scrollback is not required."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "RUN_WARNINGS.log"

    # Preserve warnings.warn output and append a durable copy.
    original_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"[PYTHON_WARNING] {category.__name__}: {message}\n"
                f"  at {filename}:{lineno}\n"
            )
        return original_showwarning(
            message, category, filename, lineno, file=file, line=line
        )

    warnings.showwarning = showwarning

    # Transformers/SentenceTransformers usually emit through logging.
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return path

def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Adapter:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.manifest = rj(spec.manifest)
        self.pyvi_version = None

        if spec.kind == "st":
            self._load_st()
        elif spec.kind == "lal":
            self._load_lal()
        else:
            raise RuntimeError(spec.kind)

    def _load_st(self):
        import torch
        from sentence_transformers import SentenceTransformer

        tok_kwargs = {}
        if self.spec.fix_mistral_regex:
            tok_kwargs["fix_mistral_regex"] = True

        self.model = SentenceTransformer(
            str(self.spec.path),
            device="cuda",
            local_files_only=True,
            trust_remote_code=self.spec.trust_remote_code,
            model_kwargs={"dtype": torch.float16},
            tokenizer_kwargs=tok_kwargs or None,
        )
        self.dim = int(self.model.get_embedding_dimension())

    def _load_lal(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.spec.path),
            local_files_only=True,
            use_fast=True,
            fix_mistral_regex=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.spec.path),
            local_files_only=True,
            dtype=torch.float16,
        ).cuda().eval()
        self.dim = int(self.model.config.hidden_size)

    def preprocess(self, texts, role):
        out = texts
        prefix = self.spec.query_prefix if role == "query" else self.spec.doc_prefix
        if prefix:
            out = [prefix + x for x in out]
        return out

    def encode_st(self, texts, role):
        max_len = self.spec.query_max if role == "query" else self.spec.doc_max
        self.model.max_seq_length = max_len
        kwargs = dict(
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if role == "query" and self.spec.query_prompt_name:
            kwargs["prompt_name"] = self.spec.query_prompt_name
        return normalize_f32(self.model.encode(texts, **kwargs))

    def encode_lal(self, texts, role):
        import torch
        import torch.nn.functional as F
        max_len = self.spec.query_max if role == "query" else self.spec.doc_max
        inp = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            out = self.model(**inp)
            last_idx = inp["attention_mask"].sum(dim=1) - 1
            emb = out.last_hidden_state[
                torch.arange(len(texts), device="cuda"), last_idx
            ]
            emb = F.normalize(emb.float(), p=2, dim=1)
        return normalize_f32(emb.cpu().numpy())

    def encode(self, texts, role):
        texts = self.preprocess(texts, role)
        return self.encode_st(texts, role) if self.spec.kind == "st" else self.encode_lal(texts, role)

    def close(self):
        del self.model
        clear_cuda()


def contract_hash(spec, manifest, geometry_sha, role, ids, pyvi_version):
    obj = {
        "schema": "stage02b4.fixed_geometry_embedding.v1",
        "model_key": spec.key,
        "model_sha": manifest.get("resolved_revision_sha"),
        "geometry_sha": geometry_sha if role == "document" else None,
        "role": role,
        "ids": ids if role == "query" else None,
        "doc_max": spec.doc_max,
        "query_max": spec.query_max,
        "doc_prefix": spec.doc_prefix,
        "query_prefix": spec.query_prefix,
        "query_prompt_name": spec.query_prompt_name,
        # BACKWARD COMPATIBILITY:
        # The original Stage02B4 cache contract included this field for every
        # model. All six retained models had False. Keep it forever for v1 cache
        # identity even though BKAI/Halong were removed from the screen.
        "word_segment": False,
        "pyvi_version": pyvi_version,
        "fix_mistral_regex": spec.fix_mistral_regex,
        "pooling_contract": manifest.get("module_summary") or manifest.get("adapter_contract"),
    }
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _validate_partial_embedding_cache(arr, completed, expected_total, expected_dim, key, role):
    """Validate a resumable prefix without scanning the entire potentially huge memmap."""
    if arr.shape != (expected_total, expected_dim):
        raise RuntimeError(
            f"[{key}] {role} cache shape mismatch: {arr.shape} != "
            f"{(expected_total, expected_dim)}. REFUSING TO OVERWRITE."
        )
    if not (0 <= completed <= expected_total):
        raise RuntimeError(
            f"[{key}] {role} invalid completed={completed}. REFUSING TO OVERWRITE."
        )
    if completed == 0:
        return

    # Check up to 2048 already-written rows, including the tail of the completed prefix.
    ncheck = min(completed, 2048)
    if completed <= ncheck:
        sample = np.asarray(arr[:completed], dtype=np.float32)
    else:
        # Deterministic coverage of the prefix plus its latest rows.
        idx = np.unique(np.concatenate([
            np.linspace(0, completed - 1, num=ncheck // 2, dtype=np.int64),
            np.arange(max(0, completed - ncheck // 2), completed, dtype=np.int64),
        ]))
        sample = np.asarray(arr[idx], dtype=np.float32)

    if not np.isfinite(sample).all():
        raise RuntimeError(
            f"[{key}] {role} partial cache contains non-finite values. "
            "REFUSING TO OVERWRITE."
        )
    norms = np.linalg.norm(sample, axis=1)
    max_err = float(np.max(np.abs(norms - 1.0)))
    if max_err > 5e-3:
        raise RuntimeError(
            f"[{key}] {role} partial cache fails unit-norm validation "
            f"(max_error={max_err:.6g}). REFUSING TO OVERWRITE."
        )


def encode_resumable(adapter: Adapter, texts, ids, role, geometry_sha):
    import torch

    spec = adapter.spec
    mdir = CACHE/spec.key
    mdir.mkdir(parents=True, exist_ok=True)
    stem = "region" if role == "document" else "query"
    arr_path = mdir/f"{stem}_embeddings.f32.npy"
    meta_path = mdir/f"{stem}_embeddings.json"
    ch = contract_hash(
        spec, adapter.manifest, geometry_sha, role,
        ids if role == "query" else None, adapter.pyvi_version
    )

    arr_exists = arr_path.exists()
    meta_exists = meta_path.exists()

    # Fail closed. Never silently destroy one half of an existing cache pair.
    if arr_exists != meta_exists:
        raise RuntimeError(
            f"[{spec.key}] {role} incomplete cache pair: "
            f"array_exists={arr_exists}, meta_exists={meta_exists}. "
            "REFUSING TO OVERWRITE. Inspect the cache manually."
        )

    completed = 0
    if arr_exists and meta_exists:
        meta = rj(meta_path)
        stored_hash = meta.get("contract_hash")
        if stored_hash != ch:
            raise RuntimeError(
                f"[{spec.key}] {role} cache contract mismatch. "
                f"stored={stored_hash}, expected={ch}. "
                "REFUSING TO OVERWRITE EXISTING EMBEDDINGS."
            )

        completed = int(meta.get("completed", 0))
        arr = np.lib.format.open_memmap(arr_path, mode="r+")
        _validate_partial_embedding_cache(
            arr, completed, len(texts), adapter.dim, spec.key, role
        )

        if completed == len(texts):
            meta["status"] = "PASS"
            print(
                f"[{spec.key}] {role} cache hit "
                f"{arr.shape} completed={completed}/{len(texts)}",
                flush=True,
            )
            return np.load(arr_path, mmap_mode="r"), meta

        print(
            f"[{spec.key}] {role} RESUME from "
            f"{completed}/{len(texts)} batch="
            f"{spec.batch_docs if role == 'document' else spec.batch_queries}",
            flush=True,
        )
    else:
        # Only create a new cache when neither file exists.
        arr = np.lib.format.open_memmap(
            arr_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), adapter.dim),
        )
        meta = {
            "contract_hash": ch,
            "completed": 0,
            "total": len(texts),
            "dimension": adapter.dim,
            "role": role,
            "status": "IN_PROGRESS",
            "pyvi_version": adapter.pyvi_version,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{spec.key}] {role} NEW cache 0/{len(texts)}",
            flush=True,
        )

    initial_batch = spec.batch_docs if role == "document" else spec.batch_queries
    batch = initial_batch
    started = time.perf_counter()
    oom_reductions = list(meta.get("oom_reductions", []))

    i = completed
    last_checkpoint = completed

    while i < len(texts):
        j = min(i + batch, len(texts))
        try:
            emb = adapter.encode(texts[i:j], role)
            if emb.shape != (j-i, adapter.dim):
                raise RuntimeError(f"Embedding shape drift {emb.shape}")

            arr[i:j] = emb
            arr.flush()
            i = j

            # Persist progress frequently enough that Ctrl+C loses at most ~1k rows.
            if i - last_checkpoint >= 1000 or i == len(texts):
                elapsed = time.perf_counter() - started
                meta = {
                    "contract_hash": ch,
                    "completed": i,
                    "total": len(texts),
                    "dimension": adapter.dim,
                    "role": role,
                    "current_batch_size": batch,
                    "initial_batch_size": initial_batch,
                    "oom_reductions": oom_reductions,
                    "elapsed_this_run_seconds": elapsed,
                    "pyvi_version": adapter.pyvi_version,
                    "status": "PASS" if i == len(texts) else "IN_PROGRESS",
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                last_checkpoint = i
                print(
                    f"[{spec.key}] {role} {i}/{len(texts)} batch={batch}",
                    flush=True,
                )

        except torch.cuda.OutOfMemoryError:
            clear_cuda()
            if batch <= 1:
                raise
            new_batch = max(1, batch // 2)
            oom_reductions.append({"at_index": i, "from": batch, "to": new_batch})
            batch = new_batch
            print(
                f"[{spec.key}] OOM -> batch {batch} at {role} index {i}",
                flush=True,
            )

    final_meta = rj(meta_path)
    if int(final_meta.get("completed", -1)) != len(texts):
        raise RuntimeError(
            f"[{spec.key}] {role} completion metadata drift after encode."
        )
    final_meta["status"] = "PASS"
    meta_path.write_text(
        json.dumps(final_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return np.load(arr_path, mmap_mode="r"), final_meta


def exact_parent_max_search(region_emb, query_emb, parent_index, doc_count):
    import torch

    clear_cuda()
    # region_emb is a read-only NumPy memmap on cache hits. PyTorch warns
    # when wrapping non-writable arrays even though this search path is read-only.
    # Make a writable contiguous copy before transferring to CUDA.
    rv_np = np.array(region_emb, dtype=np.float32, copy=True, order="C")
    rv = torch.from_numpy(rv_np).cuda()
    pidx = torch.from_numpy(parent_index).cuda().long()

    top_idx = np.empty((len(query_emb), TOPK), dtype=np.int32)
    top_scores = np.empty((len(query_emb), TOPK), dtype=np.float32)

    block = 32
    with torch.inference_mode():
        for start in range(0, len(query_emb), block):
            q_np = np.asarray(query_emb[start:start+block], dtype=np.float32)
            q = torch.from_numpy(np.ascontiguousarray(q_np)).cuda()
            scores = q @ rv.T
            parent_scores = torch.full(
                (len(q), doc_count),
                -torch.inf,
                device="cuda",
                dtype=torch.float32,
            )
            parent_scores.scatter_reduce_(
                1,
                pidx.unsqueeze(0).expand(len(q), -1),
                scores,
                reduce="amax",
                include_self=True,
            )
            vals, idx = torch.topk(
                parent_scores, k=TOPK, dim=1, largest=True, sorted=True
            )
            top_idx[start:start+len(q)] = idx.cpu().numpy().astype(np.int32)
            top_scores[start:start+len(q)] = vals.cpu().numpy().astype(np.float32)
            if start % 512 == 0 or start + len(q) == len(query_emb):
                print(f"[search] {min(start+block,len(query_emb))}/{len(query_emb)}", flush=True)

    del rv, rv_np, pidx
    clear_cuda()
    return top_idx, top_scores


def metric_curve(rank_doc_ids, qids, golds):
    curve = {}
    for k in KS:
        recalls = []
        full = 0
        for i, q in enumerate(qids):
            pred = set(rank_doc_ids[i][:k])
            g = set(golds[q])
            h = len(pred & g)
            recalls.append(h / len(g))
            full += int(h == len(g))
        curve[str(k)] = {
            "macro_recall": float(np.mean(recalls)),
            "full_gold_coverage_rate": full / len(qids),
        }
    return curve


def evaluate_model(key, rankings_idx, doc_ids, qids, golds, folds, stress):
    rank_doc_ids = [
        [doc_ids[int(j)] for j in rankings_idx[i]]
        for i in range(len(qids))
    ]
    preds5 = {q: rank_doc_ids[i][:5] for i, q in enumerate(qids)}
    curve = metric_curve(rank_doc_ids, qids, golds)

    primary = set(qids)
    qindex = {q:i for i,q in enumerate(qids)}
    per_fold = {}
    for name, ids in folds.items():
        per_fold[name] = {
            "at5": official_metrics(preds5, golds, ids),
            "recall_at100": float(np.mean([
                len(set(rank_doc_ids[qindex[q]][:100]) & set(golds[q]))
                / len(set(golds[q]))
                for q in ids
            ])),
        }

    stress_out = {}
    for name, ids in stress.items():
        use = [q for q in ids if q in primary]
        if not use:
            continue
        stress_out[name] = {
            "n": len(use),
            "recall_at5": official_metrics(preds5, golds, use)["recall_at_5"],
            "recall_at100": float(np.mean([
                len(set(rank_doc_ids[qindex[q]][:100]) & set(golds[q]))
                / len(set(golds[q]))
                for q in use
            ])),
        }

    return rank_doc_ids, {
        "curve": curve,
        "official_at5": official_metrics(preds5, golds, qids),
        "per_fold": per_fold,
        "stress": stress_out,
    }


def pairwise_complementarity(results, rankings, qids, golds):
    import csv
    keys = list(results)
    rows = []

    for a in keys:
        for b in keys:
            if a >= b:
                continue
            vals_union = []
            a_rescues_b = 0
            b_rescues_a = 0
            gold_a_not_b = 0
            gold_b_not_a = 0
            jaccards = []

            for i, q in enumerate(qids):
                A = set(rankings[a][i][:100])
                B = set(rankings[b][i][:100])
                G = set(golds[q])
                ra = len(A & G) / len(G)
                rb = len(B & G) / len(G)
                ru = len((A | B) & G) / len(G)
                vals_union.append(ru)
                a_rescues_b += int(ru > rb + 1e-12)
                b_rescues_a += int(ru > ra + 1e-12)
                gold_a_not_b += len((A & G) - B)
                gold_b_not_a += len((B & G) - A)
                jaccards.append(len(A & B) / len(A | B))

            rows.append({
                "model_a": a,
                "model_b": b,
                "union_oracle_recall_at100": float(np.mean(vals_union)),
                "union_full_gold_coverage_rate_at100": float(np.mean(np.asarray(vals_union) == 1.0)),
                "queries_a_rescues_b": a_rescues_b,
                "queries_b_rescues_a": b_rescues_a,
                "gold_occurrences_a_not_b": gold_a_not_b,
                "gold_occurrences_b_not_a": gold_b_not_a,
                "mean_top100_jaccard": float(np.mean(jaccards)),
            })

    rows.sort(key=lambda x: -x["union_oracle_recall_at100"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/"PAIRWISE_COMPLEMENTARITY.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    return rows


def run_model(spec, geom, qids, questions, golds, folds, stress):
    import torch

    print(f"\n=== {spec.key} ===", flush=True)
    if not spec.path.exists() or not spec.manifest.exists():
        raise FileNotFoundError(f"Missing model/materialization for {spec.key}")

    adapter = Adapter(spec)
    try:
        t0 = time.perf_counter()
        region_emb, region_meta = encode_resumable(
            adapter, geom["texts"], None, "document", geom["sha"]
        )
        query_texts = [questions[q] for q in qids]
        query_emb, query_meta = encode_resumable(
            adapter, query_texts, qids, "query", geom["sha"]
        )
        encode_seconds = time.perf_counter() - t0
    finally:
        adapter.close()

    model_cache = CACHE/spec.key
    rank_idx_path = model_cache/"parent_top100_idx.npy"
    rank_score_path = model_cache/"parent_top100_scores.npy"
    rank_meta_path = model_cache/"parent_top100.json"

    search_contract = {
        "schema": "stage02b4.parent_max_search.v1",
        "geometry_sha": geom["sha"],
        "region_embedding_contract": region_meta["contract_hash"],
        "query_embedding_contract": query_meta["contract_hash"],
        "aggregation": "exact_parent_amax_over_fixed_regions",
        "topk": TOPK,
    }
    sch = hashlib.sha256(
        json.dumps(search_contract, sort_keys=True).encode()
    ).hexdigest()

    if (
        rank_idx_path.exists()
        and rank_score_path.exists()
        and rank_meta_path.exists()
        and rj(rank_meta_path).get("contract_hash") == sch
    ):
        idx = np.load(rank_idx_path)
        scores = np.load(rank_score_path)
        print(f"[{spec.key}] ranking cache hit", flush=True)
    else:
        t1 = time.perf_counter()
        idx, scores = exact_parent_max_search(
            region_emb, query_emb, geom["parent_index"], len(geom["doc_ids"])
        )
        search_seconds = time.perf_counter() - t1
        np.save(rank_idx_path, idx)
        np.save(rank_score_path, scores)
        rank_meta_path.write_text(
            json.dumps({
                "contract_hash": sch,
                "contract": search_contract,
                "search_seconds": search_seconds,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    rankings, metrics = evaluate_model(
        spec.key, idx, geom["doc_ids"], qids, golds, folds, stress
    )

    manifest = rj(spec.manifest)
    result = {
        "model_key": spec.key,
        "model_id": manifest.get("model_id"),
        "model_sha": manifest.get("resolved_revision_sha"),
        "parameter_count": manifest.get("parameter_count"),
        "document_max_length": spec.doc_max,
        "query_max_length": spec.query_max,
        "document_prefix": spec.doc_prefix,
        "query_prefix": spec.query_prefix,
        "query_prompt_name": spec.query_prompt_name,
        "pyvi_version": region_meta.get("pyvi_version"),
        "geometry": {
            "regions": len(geom["region_ids"]),
            "parents": len(geom["doc_ids"]),
            "sha256": geom["sha"],
            "same_geometry_for_all_models": True,
            "long_region_policy": "truncate, never split",
        },
        "runtime": {
            "encode_wall_seconds_this_invocation": encode_seconds,
            "final_document_batch": region_meta.get("current_batch_size"),
            "document_oom_reductions": region_meta.get("oom_reductions", []),
        },
        "metrics": metrics,
    }
    mdir = OUT/spec.key
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir/"RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result, rankings


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warning_log = OUT / "RUN_WARNINGS.log"
    warning_log.write_text("", encoding="utf-8")
    install_persistent_warning_log()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        default="all",
        help="all or comma-separated model keys",
    )
    args = p.parse_args()

    if args.models == "all":
        selected = list(SPECS)
    else:
        selected = [x.strip() for x in args.models.split(",") if x.strip()]
        unknown = [x for x in selected if x not in SPECS]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")

    geom = load_geometry()
    qids, questions, golds, folds, stress = load_eval()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    results = {}
    rankings = {}
    failures = {}

    for key in selected:
        try:
            result, rank = run_model(
                SPECS[key], geom, qids, questions, golds, folds, stress
            )
            results[key] = result
            rankings[key] = rank
            m = result["metrics"]
            print(json.dumps({
                "model": key,
                "status": "COMPLETE",
                "recall_at5": m["official_at5"]["recall_at_5"],
                "recall_at20": m["curve"]["20"]["macro_recall"],
                "recall_at100": m["curve"]["100"]["macro_recall"],
                "full_coverage_at100": m["curve"]["100"]["full_gold_coverage_rate"],
            }, ensure_ascii=False, indent=2))
        except Exception as exc:
            import traceback
            failures[key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(json.dumps({
                "model": key, "status": "FAIL", "error": str(exc)
            }, ensure_ascii=False, indent=2))
            clear_cuda()

    pairwise = pairwise_complementarity(
        results, rankings, qids, golds
    ) if len(results) >= 2 else []

    summary = {
        "schema_version": "dsc2026.endgame.stage02b4_vi_screen.v1",
        "status": "COMPLETE" if not failures else "PARTIAL_FAIL",
        "scientific_claim_boundary": (
            "fixed moderate geometry encoder-stack screen only; "
            "no representation or final pipeline promotion"
        ),
        "geometry": {
            "region_count": len(geom["region_ids"]),
            "parent_count": len(geom["doc_ids"]),
            "sha256": geom["sha"],
            "fixed_across_models": True,
            "long_region_policy": "truncate_to_model_contract_no_split",
        },
        "results": {
            k: {
                "model_id": v["model_id"],
                "parameter_count": v["parameter_count"],
                "document_max_length": v["document_max_length"],
                "query_max_length": v["query_max_length"],
                "recall_at5": v["metrics"]["official_at5"]["recall_at_5"],
                "precision_at5": v["metrics"]["official_at5"]["precision_at_5"],
                "single_gold_recall_at5": v["metrics"]["official_at5"]["single_gold_recall_at_5"],
                "multi_gold_recall_at5": v["metrics"]["official_at5"]["multi_gold_recall_at_5"],
                "recall_at20": v["metrics"]["curve"]["20"]["macro_recall"],
                "recall_at100": v["metrics"]["curve"]["100"]["macro_recall"],
                "full_gold_coverage_at100": v["metrics"]["curve"]["100"]["full_gold_coverage_rate"],
                "per_fold": v["metrics"]["per_fold"],
            }
            for k, v in results.items()
        },
        "failures": failures,
        "warning_log": "RUN_WARNINGS.log",
        "pairwise_complementarity_file": (
            "PAIRWISE_COMPLEMENTARITY.csv" if pairwise else None
        ),
    }
    (OUT/"SCREEN_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Compact human report, no winner/ranking decision.
    lines = [
        "# Stage 02B4 — Vietnamese-first fixed-geometry encoder screen",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Same 179,737 moderate regions for every model; one vector per region;",
        "no extra splitting/overlap; exact max aggregation to parent.",
        "",
        "| Model | Params | Doc max | R@5 | R@20 | R@100 | Full@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["results"].items():
        pb = (
            row["parameter_count"]/1e9
            if row["parameter_count"] is not None else float("nan")
        )
        lines.append(
            f"| {key} | {pb:.3f}B | {row['document_max_length']} | "
            f"{row['recall_at5']:.6f} | {row['recall_at20']:.6f} | "
            f"{row['recall_at100']:.6f} | {row['full_gold_coverage_at100']:.4%} |"
        )
    if failures:
        lines += ["", "Failures:", ""]
        for k, v in failures.items():
            lines.append(f"- {k}: `{v['error_type']}: {v['error']}`")
    lines += [
        "",
        "No model is promoted automatically from this report.",
        "Next step uses standalone + pairwise rescue behavior to choose encoders",
        "for representation/packing experiments.",
        "",
    ]
    (OUT/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "completed_models": list(results),
        "failed_models": list(failures),
        "warning_log": str((OUT/"RUN_WARNINGS.log").resolve()),
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))

    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
