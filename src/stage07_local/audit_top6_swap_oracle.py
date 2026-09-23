"""Cheap, fold-clean upper bound for zero-shot Qwen rank5/rank6 verification."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_breakthrough.run_residual_setwise_oof as sw

qids, _, golds, folds, _, docs, _ = b1.load_world()
short, _ = b1.load_shortlist(len(qids))
sources = b1.load_sources(len(qids))
ce = np.load(b1.CACHE / "oof_top30_ce_scores.f32.npy")
X, _, _, _ = b1.build_source_ce_features(short, ce, sources)
X = X.reshape(len(qids), 30, 45)
y = qg.labels_for(short, qids, golds, docs)
report = {}
for fn in folds:
    _, held, _, bhe = sw.inner_crossfit_prior(X, y, folds, qids, fn)
    base = np.asarray(bhe, np.float32)
    order = np.argsort(-base, axis=1, kind="stable")
    ordered = np.take_along_axis(y[held], order, axis=1)
    can_gain = (ordered[:, 4] == 0) & (ordered[:, 5] == 1)
    can_harm = (ordered[:, 4] == 1) & (ordered[:, 5] == 0)
    den = np.asarray([len(golds[qids[i]]) for i in held], np.float32)
    report[fn] = {"queries": len(held), "oracle_gain_queries": int(can_gain.sum()),
                  "risk_queries": int(can_harm.sum()),
                  "oracle_gain_recall": float(np.mean(can_gain / den)),
                  "rank6_gold_queries": int((ordered[:, 5] == 1).sum())}
print(json.dumps(report, indent=2))
