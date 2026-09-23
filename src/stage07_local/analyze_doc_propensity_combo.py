"""Fixed, preregistered equal-weight check for new label-memory signal."""
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
scores = {
    "propensity": np.load(ROOT / "reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"),
    "ait": np.load(ROOT / "cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"),
    "lal": np.load(ROOT / "cache/stage07p_lal_gold_centroid/lal_gold_oof_scores30.f32.npy"),
}
groups = [("propensity", "ait"), ("propensity", "lal"),
          ("propensity", "ait", "lal")]
result = {}
for group in groups:
    s = sum(ait.zrows(scores[x]) for x in group) / len(group)
    rank = ait.rank(short, s)
    met = b1.eval_rank(rank, qids, golds, docs, folds, stress)
    result["+".join(group)] = {"overall": met["overall"],
                               "per_fold": {f: met["per_fold"][f]["recall_at_5"] for f in folds}}
OUT = ROOT / "reports/stage07r_gold_doc_propensity/FIXED_COMBO_REPORT.json"
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for name, value in result.items():
    print(name, value["overall"]["recall_at_5"],
          "multi", value["overall"]["multi_gold_recall_at_5"])
