#!/usr/bin/env python3
"""Trace the first Stage07O training updates without changing its runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest_arrays(values) -> str:
    h = hashlib.sha256()
    for value in values:
        if value is not None:
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    print("TRACE_PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "<random>"), flush=True)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import torch
    import src.stage07_local.run_gold_centroid_head_oof as head

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    qids, _, folds, _, _, short, X45, X, y, qe, cent = head.make_world()
    name = list(folds)[args.fold]
    tr, held, btr, bhe = head.sw.inner_crossfit_prior(X45, y, folds, qids, name)
    for label, value in (("short", short), ("X45", X45), ("X", X),
                         ("y", y), ("qe", qe), ("cent", cent), ("btr", btr)):
        print("TRACE_INPUT", label, hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(), flush=True)

    original_make_model = head.make_model
    original_step = torch.optim.AdamW.step
    counter = 0

    def traced_make_model(dim):
        model = original_make_model(dim)
        print("TRACE_INITIAL", digest_arrays(model.parameters()), flush=True)
        return model

    def traced_step(optimizer, closure=None):
        nonlocal counter
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if counter < args.steps:
            print("TRACE_BEFORE", counter, digest_arrays(parameters),
                  digest_arrays(p.grad for p in parameters), flush=True)
        result = original_step(optimizer, closure=closure)
        if counter < args.steps:
            print("TRACE_AFTER", counter, digest_arrays(parameters), flush=True)
        counter += 1
        return result

    head.make_model = traced_make_model
    torch.optim.AdamW.step = traced_step
    params = type("Params", (), {"seed": 276, "device": "cuda", "epochs": 4,
                                  "batch_queries": 32, "lr": 8e-4})()
    residual, meta = head.fit_predict(args.fold, tr, held, btr, short, X, y, qe, cent, params)
    scores = head.zrows(bhe)
    scores[:, :head.DEPTH] += meta["alpha"] * residual
    print("TRACE_RESULT", json.dumps({
        "fold": name,
        "scores_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
        "alpha": meta["alpha"],
        "loss_tail": meta["loss_tail"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
