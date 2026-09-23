#!/usr/bin/env python
"""Fold-clean gold-supervised centroid reranker. No distillation inputs.

The parent centroids are a deterministic mean of the frozen AITeam region
embeddings. Their cache happens to live under stage07d, but this runner reads
only that raw-embedding derivative; it never reads a Stage07D prediction,
checkpoint, teacher target, or teacher score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage05_selector.run_qgate_selector_oof as qg
import src.stage07_breakthrough.run_residual_setwise_oof as sw

OUT = ROOT / "reports/stage07o_gold_centroid_head"
CACHE = ROOT / "cache/stage07o_gold_centroid_head"
CENTROID = ROOT / "cache/stage07d_teacher_distill_head/aiteam_parent_centroids.f32.npy"
CENTROID_META = CENTROID.parent / "aiteam_parent_centroids.json"
REGION = ROOT / "cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy"
DEPTH = 20
FULL = 30
BASE_R = 0.9425976255185238


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def zrows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    sd = a.std(axis=1, keepdims=True)
    return (a - a.mean(axis=1, keepdims=True)) / np.where(sd < 1e-6, 1, sd)


def load_raw_centroids(ndocs: int) -> np.ndarray:
    meta = json.loads(CENTROID_META.read_text(encoding="utf-8"))
    expected = "cache/stage02b4_vi_screen/aiteamvn_v1/region_embeddings.f32.npy"
    if meta.get("source") != expected or meta.get("aggregation") != "mean(region embeddings) then L2 normalize":
        raise RuntimeError("centroid source/aggregation drift")
    if not REGION.is_file():
        raise FileNotFoundError(REGION)
    c = np.load(CENTROID, mmap_mode="r")
    if c.shape != (ndocs, 1024) or not np.isfinite(c).all():
        raise RuntimeError("centroid shape or value drift")
    return np.asarray(c, np.float32)


def make_world():
    qids, questions, golds, folds, stress, docs, _ = b1.load_world()
    short, _ = b1.load_shortlist(len(qids))
    sources = b1.load_sources(len(qids))
    ce = np.load(b1.CACHE / "oof_top30_ce_scores.f32.npy")
    done = np.load(b1.CACHE / "oof_top30_done.u1.npy")
    if ce.shape != short.shape or int(done.sum()) != len(qids):
        raise RuntimeError("Stage03B1 CE cache incomplete")
    X45, _, _, _ = b1.build_source_ce_features(short, ce, sources)
    X45 = X45.reshape(len(qids), FULL, 45)
    title = qg.make_title_features(qids, questions, short, qg.load_names(docs))
    X = np.concatenate([X45, title], axis=2).astype(np.float32)
    y = qg.labels_for(short, qids, golds, docs).astype(np.float32)
    qe = qg.query_embed(qids)
    cent = load_raw_centroids(len(docs))
    return qids, golds, folds, stress, docs, short, X45, X, y, qe, cent


def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cdim: int):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.LayerNorm(128))
            self.d = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.LayerNorm(128))
            self.c = nn.Sequential(nn.Linear(cdim, 96), nn.GELU(), nn.LayerNorm(96))
            self.h = nn.Sequential(nn.Linear(608, 192), nn.GELU(), nn.Dropout(0.08),
                                   nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))
            self.raw_alpha = nn.Parameter(torch.tensor(-1.8))

        def forward(self, q, d, c):
            qq, dd, cc = self.q(q), self.d(d), self.c(c)
            return self.h(torch.cat((qq, dd, qq * dd, torch.abs(qq - dd), cc), -1)).squeeze(-1)

        def alpha(self):
            return 0.5 * torch.sigmoid(self.raw_alpha)

    return Head()


def fit_predict(outer, tr, held, btr, short, X, y, qe, cent, args, save_checkpoint=None):
    import torch
    import torch.nn.functional as F

    seed_all(args.seed + outer)
    device = torch.device(args.device)
    model = make_model(X.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    tq = torch.from_numpy(qe).to(device)
    td = torch.from_numpy(cent).to(device)
    tx = torch.from_numpy(X[:, :DEPTH]).to(device)
    ty = torch.from_numpy(y[:, :DEPTH]).to(device)
    base = torch.from_numpy(zrows(btr)[:, :DEPTH]).to(device)
    trloc = {int(q): i for i, q in enumerate(tr)}
    valid = np.asarray([int(q) for q in tr if y[int(q), :DEPTH].sum() > 0], np.int32)
    rng = np.random.default_rng(args.seed + outer)
    t0 = time.perf_counter()
    losses = []

    def forward(qis):
        n = len(qis)
        di = torch.as_tensor(short[qis, :DEPTH].reshape(-1), device=device)
        q = tq[qis][:, None, :].expand(-1, DEPTH, -1).reshape(n * DEPTH, 1024)
        d = td[di]
        c = tx[qis].reshape(n * DEPTH, -1)
        return model(q, d, c).reshape(n, DEPTH)

    for ep in range(args.epochs):
        order = valid.copy()
        rng.shuffle(order)
        model.train()
        ep_losses = []
        for start in range(0, len(order), args.batch_queries):
            qis = order[start:start + args.batch_queries]
            p = forward(qis)
            yy = ty[qis]
            bb = base[torch.as_tensor([trloc[int(i)] for i in qis], device=device)]
            total = bb + model.alpha() * p
            target = yy / yy.sum(1, keepdim=True).clamp_min(1)
            listwise = -(target * F.log_softmax(total, dim=1)).sum(1).mean()
            # The hardest six wrong candidates are chosen from a fold-clean prior.
            pair_terms = []
            for i in range(len(qis)):
                pos = torch.where(yy[i] > 0)[0]
                neg = torch.where(yy[i] == 0)[0]
                if len(pos) and len(neg):
                    hard = neg[torch.topk(bb[i, neg], min(6, len(neg))).indices]
                    pair_terms.append(F.softplus(0.4 - total[i, pos, None] + total[i, None, hard]).mean())
            pair = torch.stack(pair_terms).mean() if pair_terms else listwise * 0
            bce = F.binary_cross_entropy_with_logits(p, yy, reduction="none")
            pw = 0.5 / yy.sum(1, keepdim=True).clamp_min(1)
            nw = 0.5 / (1 - yy).sum(1, keepdim=True).clamp_min(1)
            bce = (bce * torch.where(yy > 0, pw, nw)).sum(1).mean()
            loss = 0.55 * listwise + 0.25 * pair + 0.20 * bce + 0.003 * p.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite gold-supervised loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
        losses = ep_losses
        print(f"[fold{outer}] ep={ep+1}/{args.epochs} loss={np.mean(losses):.5f} "
              f"alpha={float(model.alpha().detach().cpu()):.4f} "
              f"elapsed={(time.perf_counter()-t0)/60:.1f}m", flush=True)

    model.eval()
    if save_checkpoint is not None:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "candidate_dim": int(X.shape[-1]), "epochs": args.epochs,
                    "seed": args.seed + outer,
                    "supervision": "gold_labels_only",
                    "teacher_used": False}, save_checkpoint)
    pred = np.empty((len(held), DEPTH), np.float32)
    with torch.inference_mode():
        for start in range(0, len(held), args.batch_queries):
            qis = held[start:start + args.batch_queries]
            pred[start:start + len(qis)] = forward(qis).cpu().numpy()
    alpha = float(model.alpha().detach().cpu())
    meta = {"alpha": alpha, "train_queries": len(tr), "held_queries": len(held),
            "loss_tail": float(np.mean(losses[-50:])),
            "wall_seconds": time.perf_counter() - t0}
    return pred, meta


def rank(short, score):
    out = np.empty_like(short)
    for i in range(len(short)):
        out[i] = short[i, np.lexsort((short[i], -score[i]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=276)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    qids, golds, folds, stress, docs, short, X45, X, y, qe, cent = make_world()
    base_oof = np.zeros((len(qids), FULL), np.float32)
    score_oof = np.zeros_like(base_oof)
    fold_meta = {}
    for outer, fn in enumerate(folds):
        tr, held, btr, bhe = sw.inner_crossfit_prior(X45, y, folds, qids, fn)
        base_oof[held] = bhe
        pred, meta = fit_predict(outer, tr, held, btr, short, X, y, qe, cent, args)
        s = zrows(bhe)
        s[:, :DEPTH] += meta["alpha"] * pred
        score_oof[held] = s
        fold_meta[fn] = meta
    baseline = rank(short, base_oof)
    candidate = rank(short, score_oof)
    bm = b1.eval_rank(baseline, qids, golds, docs, folds, stress)
    cm = b1.eval_rank(candidate, qids, golds, docs, folds, stress)
    if abs(bm["overall"]["recall_at_5"] - BASE_R) > 3e-6:
        raise RuntimeError("CE-LR OOF parity failure")
    bq, cq = [], []
    for i, q in enumerate(qids):
        g = set(golds[q])
        bq.append(len(g & {docs[int(x)] for x in baseline[i, :5]}) / len(g))
        cq.append(len(g & {docs[int(x)] for x in candidate[i, :5]}) / len(g))
    bq, cq = np.asarray(bq), np.asarray(cq)
    delta = cm["overall"]["recall_at_5"] - bm["overall"]["recall_at_5"]
    folds_delta = {f: cm["per_fold"][f]["recall_at_5"] - bm["per_fold"][f]["recall_at_5"] for f in folds}
    wins, losses = int((cq > bq).sum()), int((cq < bq).sum())
    decision = "PROMOTE_FOR_PRIVATE" if delta >= 0.002 and wins >= losses and sum(v > 0 for v in folds_delta.values()) >= 3 and cm["overall"]["multi_gold_recall_at_5"] >= bm["overall"]["multi_gold_recall_at_5"] else "KILL_ON_OOF"
    np.save(CACHE / "gold_head_oof_scores30.f32.npy", score_oof)
    report = {"schema": "dsc2026.endgame.stage07o.gold_centroid_head.v1", "status": "COMPLETE",
              "teacher_used": False, "distillation_used": False,
              "supervision": "competition gold labels only", "prior": "inner crossfit CE-LR 45D",
              "raw_centroid_source": str(REGION.relative_to(ROOT)).replace("\\", "/"),
              "raw_centroid_sha256": sha256(CENTROID),
              "baseline": bm, "candidate": cm,
              "effect": {"delta_recall": delta, "wins": wins, "losses": losses,
                         "fold_deltas": folds_delta}, "fold_meta": fold_meta,
              "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "BASE", bm["overall"]["recall_at_5"],
          "CANDIDATE", cm["overall"]["recall_at_5"], "DELTA", delta,
          "W/L", wins, losses, flush=True)


if __name__ == "__main__":
    main()
