"""Authoritative ENDGAME evaluation utilities.

Two label views may exist:
- primary local OOF: retrievable literal gold IDs after empty-passage filtering;
- raw-official diagnostic: untouched official train labels.

Both preserve literal document IDs. Exact duplicate-content IDs are NOT
canonicalized or substituted.

Competition scoring contract:
- valid prediction length: 1..5
- query recall = hits / |gold|
- query precision = hits / |returned|
- aggregate = macro mean over queries
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Any
import numpy as np


def official_query_metrics(predictions: Sequence[str], gold: Iterable[str]) -> dict[str, float]:
    pred = [str(x) for x in predictions]
    gold_set = {str(x) for x in gold}
    if not gold_set:
        raise ValueError("Evaluation gold set must be non-empty.")
    if not (0 < len(pred) <= 5):
        return {"recall": 0.0, "precision": 0.0, "hits": 0.0}
    hits = len(set(pred) & gold_set)
    return {
        "recall": hits / len(gold_set),
        "precision": hits / len(pred),
        "hits": float(hits),
    }


def official_metrics(
    predictions: Mapping[str, Sequence[str]],
    golds: Mapping[str, Iterable[str]],
    qids: Sequence[str] | None = None,
) -> dict[str, float]:
    ids = list(qids) if qids is not None else list(golds)
    recalls, precisions, sizes = [], [], []
    single, multi = [], []

    for qid in ids:
        row = official_query_metrics(predictions.get(qid, []), golds[qid])
        recalls.append(row["recall"])
        precisions.append(row["precision"])
        sizes.append(len(predictions.get(qid, [])))
        if len(set(golds[qid])) == 1:
            single.append(row["recall"])
        else:
            multi.append(row["recall"])

    recall = float(np.mean(recalls)) if recalls else 0.0
    precision = float(np.mean(precisions)) if precisions else 0.0
    f2 = 0.0 if 4 * precision + recall == 0 else (
        5 * precision * recall / (4 * precision + recall)
    )
    return {
        "queries": len(ids),
        "recall_at_5": recall,
        "precision_at_5": precision,
        "f2_at_5": f2,
        "single_gold_recall_at_5": float(np.mean(single)) if single else 0.0,
        "multi_gold_recall_at_5": float(np.mean(multi)) if multi else 0.0,
        "mean_returned": float(np.mean(sizes)) if sizes else 0.0,
    }


def evaluate_folds(
    predictions: Mapping[str, Sequence[str]],
    golds: Mapping[str, Iterable[str]],
    folds: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    all_qids = [q for ids in folds.values() for q in ids]
    out = official_metrics(predictions, golds, all_qids)
    out["per_fold"] = {
        name: official_metrics(predictions, golds, list(ids))
        for name, ids in folds.items()
    }
    return out
