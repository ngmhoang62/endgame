"""One fixed equal-weight check for each clean semantic representation set."""
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
sources = {
    "ait_centroid": np.load(ROOT / "cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"),
    "lal_centroid": np.load(ROOT / "cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy"),
    "ait_witness": np.load(ROOT / "cache/stage07q_gold_witness_head/gold_witness_oof_scores30.f32.npy"),
}
sets = [("ait_centroid", "ait_witness"),
        ("lal_centroid", "ait_witness"),
        ("ait_centroid", "lal_centroid", "ait_witness")]
result = {}
for group in sets:
    score = sum(ait.zrows(sources[name]) for name in group) / len(group)
    rank = ait.rank(short, score)
    m = b1.eval_rank(rank, qids, golds, docs, folds, stress)
    result["+".join(group)] = {"overall": m["overall"], "per_fold": {
        f: m["per_fold"][f]["recall_at_5"] for f in folds}}
OUT = ROOT / "reports/stage07q_gold_witness_head/FIXED_ENSEMBLE_REPORT.json"
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for name, value in result.items():
    print(name, "R", value["overall"]["recall_at_5"],
          "multi", value["overall"]["multi_gold_recall_at_5"])
