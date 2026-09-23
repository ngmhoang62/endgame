"""One prespecified equal-weight check of three independently clean OOF arms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_local.run_gold_centroid_head_oof as ait

qids, _, golds, folds, stress, docs, _ = b1.load_world()
short, _ = b1.load_shortlist(len(qids))
paths = {
    "gold_document_propensity": ROOT / "reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy",
    "pointwise_raw_embedding": ROOT / "cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy",
    "setwise_raw_embedding": ROOT / "cache/stage07u_gold_embedding_setwise/setwise_centroid_oof_scores30.f32.npy",
}
scores = []
for name, path in paths.items():
    arr = np.load(path)
    if arr.shape != short.shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{name} OOF shape/value drift")
    scores.append(ait.zrows(arr))
rank = ait.rank(short, sum(scores) / len(scores))
metric = b1.eval_rank(rank, qids, golds, docs, folds, stress)
report = {"schema": "stage07v.fixed_clean_three_signal.v1", "teacher_used": False,
          "distillation_used": False, "weights": {name: 1 / 3 for name in paths},
          "sources": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in paths.items()},
          "metrics": metric, "decision": "PROMOTE_FOR_PRIVATE" if metric["overall"]["recall_at_5"] >= .95 else "KILL"}
out = ROOT / "reports/stage07v_fixed_clean_three_signal/OOF_REPORT.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(report["decision"], metric["overall"]["recall_at_5"],
      "multi", metric["overall"]["multi_gold_recall_at_5"])
