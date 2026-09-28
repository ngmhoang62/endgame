#!/usr/bin/env python3
"""Rebuild and audit the GOLD_PROP_AIT_W75_K5 local OOF path.

Only official raw data, pinned public model revisions, and repository source
are inputs. No private-set scoring or submission creation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/official_v1"
MANIFEST = DATA / "DATA_SNAPSHOT_MANIFEST.json"
AUDIT = ROOT / "reports/stage07y_final_clean_fusion/FUSION_AUDIT.json"

STAGES = [
    ("evaluation universe", "src/stage00_audit/build_endgame_evaluation_v2.py"),
    ("VietLegal E5 model", "src/stage00_audit/materialize_hf_model.py"),
    ("VNLegal LAL model", "src/stage02_candidate_generation/materialize_retriever_candidates.py", "--candidate", "vnlegal_lal"),
    ("AIT embedding model", "src/stage02_candidate_generation/materialize_vietnamese_retrievers.py", "--candidate", "aiteamvn_vietnamese_embedding"),
    ("article geometry", "src/stage02_candidate_generation/audit_moderate_structure.py"),
    ("E5/BM25 anchor", "src/stage02_candidate_generation/run_parent_anchor.py"),
    ("query and region embeddings", "src/stage02_candidate_generation/screen_vietnamese_retrievers.py", "--models", "vietlegal_e5,vnlegal_lal,aiteamvn_v1"),
    ("AIT representations", "src/stage02_candidate_generation/screen_aiteam_representations.py", "--representations", "atomic_split_2048,coarse_pack_1024"),
    ("LAL representations", "src/stage02_candidate_generation/screen_lal_representations.py", "--representations", "lal_atomic_split_2048,lal_coarse_pack_1024"),
    ("OOF shortlist", "src/stage03_rerank/prepare_oof_shortlist.py"),
    ("AIT reranker model", "src/stage03_rerank/materialize_aiteam_reranker.py"),
    ("AIT reranker OOF", "src/stage03_rerank/run_aiteam_reranker_oof.py"),
    ("raw AIT centroids", "src/stage07_local/materialize_gold_raw_centroids.py"),
    ("gold propensity OOF", "src/stage07_local/run_gold_doc_propensity_oof.py"),
    ("gold centroid head OOF", "src/stage07_local/run_gold_centroid_head_oof.py"),
]

EXPECTED = {
    "w_prop": 0.75,
    "R": 0.9505364039479332,
    "multi": 0.7680580762250454,
    "nested_weighted_R": 0.9502503218423687,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_data() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("schema_version") != "dsc2026.endgame.official_data_snapshot.v1":
        raise RuntimeError("Official data manifest is not the locked PASS snapshot")
    rows = manifest["files"]
    if len(rows) != 8535 or len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("Official data manifest population drift")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe manifest path: {relative}")
        target = DATA / relative
        if not target.is_file() or target.stat().st_size != row["size_bytes"] or sha256(target) != row["sha256"]:
            raise RuntimeError(f"Official data missing or differs from the locked snapshot: {relative}")
    print(f"OFFICIAL_DATA_PASS files={len(rows)}", flush=True)


def verify_cv() -> None:
    # Recompute the audit from generated OOF arrays. The committed historical
    # report alone is evidence, not proof that a fresh clone regenerated it.
    subprocess.run(
        [sys.executable, "src/stage07_local/run_final_clean_fusion_private.py", "--oof-only"],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("teacher_used") is not False or audit.get("distillation_used") is not False:
        raise RuntimeError("Fusion provenance drift")
    actual = {**audit["robust_choice"], "nested_weighted_R": audit["nested_weighted_R"]}
    for name, value in EXPECTED.items():
        if abs(float(actual[name]) - value) > 1e-12:
            raise RuntimeError(f"GOLD local CV differs: {name}={actual[name]} expected={value}")
    print("GOLD_LOCAL_CV_EXACT_PASS", {name: actual[name] for name in EXPECTED}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-data", "run", "verify-cv"))
    args = parser.parse_args()
    if args.action in ("check-data", "run"):
        check_data()
    if args.action == "run":
        for stage in STAGES:
            name, *command = stage
            print(f"\n=== {name} ===", flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    if args.action in ("run", "verify-cv"):
        verify_cv()


if __name__ == "__main__":
    main()
