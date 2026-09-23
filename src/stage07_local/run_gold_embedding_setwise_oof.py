#!/usr/bin/env python
"""Gold-only setwise reranker over raw AIT document embeddings and CE-LR prior.

Unlike Stage07A, tokens contain frozen document vectors; unlike Stage07O,
candidate tokens attend to each other. No teacher, KD, or Stage07D score.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_breakthrough.run_residual_setwise_oof as sw
import src.stage07_local.run_gold_centroid_head_oof as ait

OUT = ROOT / "reports/stage07u_gold_embedding_setwise"
CACHE = ROOT / "cache/stage07u_gold_embedding_setwise"


def fit_predict(outer, tr, held, btr, bhe, short, x, y, qe, cent, args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed = args.seed + outer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = torch.device(args.device)
    k = ait.DEPTH

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            h = 128
            self.q = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.LayerNorm(64))
            self.d = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.LayerNorm(64))
            self.c = nn.Sequential(nn.Linear(x.shape[-1] + 1, 64), nn.GELU(), nn.LayerNorm(64))
            self.token = nn.Sequential(nn.Linear(64 * 4, h), nn.GELU(), nn.LayerNorm(h))
            layer = nn.TransformerEncoderLayer(h, 4, h * 2, dropout=.08,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(h))
            self.out = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 1))
            self.raw_alpha = nn.Parameter(torch.tensor(-1.8))

        def forward(self, q, d, c, prior):
            qq = self.q(q)[:, None, :].expand(-1, k, -1)
            dd = self.d(d)
            cc = self.c(torch.cat((c, prior[:, :, None]), dim=-1))
            tok = self.token(torch.cat((qq, dd, qq * dd, cc), dim=-1))
            delta = self.out(self.enc(tok)).squeeze(-1)
            alpha = .5 * torch.sigmoid(self.raw_alpha)
            return prior + alpha * delta, delta, alpha

    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    tq = torch.from_numpy(np.array(qe, copy=True)).to(dev)
    td = torch.from_numpy(np.array(cent, copy=True)).to(dev)
    tx = torch.from_numpy(np.array(x[:, :k], copy=True)).to(dev)
    ty = torch.from_numpy(np.array(y[:, :k], copy=True)).to(dev)
    prior_tr = torch.from_numpy(np.array(ait.zrows(btr)[:, :k], copy=True)).to(dev)
    prior_he = torch.from_numpy(np.array(ait.zrows(bhe)[:, :k], copy=True)).to(dev)
    trpos = {int(q): i for i, q in enumerate(tr)}
    valid = np.asarray([int(q) for q in tr if y[int(q), :k].sum() > 0], np.int32)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    loss_tail = []

    def forward(qis, prior):
        di = torch.as_tensor(short[qis, :k].astype(np.int64), device=dev)
        return model(tq[qis], td[di], tx[qis], prior)

    for ep in range(args.epochs):
        order = valid.copy()
        rng.shuffle(order)
        model.train()
        losses = []
        for st in range(0, len(order), args.batch_queries):
            qis = order[st:st + args.batch_queries]
            base = prior_tr[torch.as_tensor([trpos[int(q)] for q in qis], device=dev)]
            total, delta, _ = forward(qis, base)
            yy = ty[qis]
            target = yy / yy.sum(1, keepdim=True).clamp_min(1)
            listwise = -(target * F.log_softmax(total, dim=1)).sum(1).mean()
            pairs = []
            for i in range(len(qis)):
                pos = torch.where(yy[i] > 0)[0]
                neg = torch.where(yy[i] == 0)[0]
                if len(pos) and len(neg):
                    hard = neg[torch.topk(base[i, neg], min(6, len(neg))).indices]
                    pairs.append(F.softplus(.4 - total[i, pos, None] + total[i, None, hard]).mean())
            pair = torch.stack(pairs).mean() if pairs else listwise * 0
            bce = F.binary_cross_entropy_with_logits(delta, yy, reduction="none")
            pw = .5 / yy.sum(1, keepdim=True).clamp_min(1)
            nw = .5 / (1 - yy).sum(1, keepdim=True).clamp_min(1)
            bce = (bce * torch.where(yy > 0, pw, nw)).sum(1).mean()
            loss = .55 * listwise + .25 * pair + .2 * bce + .003 * delta.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite gold loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        loss_tail = losses
        print(f"[fold{outer}] ep={ep + 1}/{args.epochs} loss={np.mean(losses):.5f} "
              f"alpha={float((.5 * torch.sigmoid(model.raw_alpha)).detach().cpu()):.4f} "
              f"elapsed={(time.perf_counter() - t0) / 60:.1f}m", flush=True)

    model.eval()
    pred = np.empty((len(held), k), np.float32)
    with torch.inference_mode():
        for st in range(0, len(held), args.batch_queries):
            qis = held[st:st + args.batch_queries]
            total, _, _ = forward(qis, prior_he[st:st + len(qis)])
            pred[st:st + len(qis)] = total.cpu().numpy()
    return pred, {"alpha": float((.5 * torch.sigmoid(model.raw_alpha)).detach().cpu()),
                  "loss_tail": float(np.mean(loss_tail[-50:])),
                  "wall_seconds": time.perf_counter() - t0}


def perq(rank, qids, golds, docs):
    return np.asarray([len(set(golds[q]) & {docs[int(d)] for d in rank[i, :5]}) / len(golds[q])
                       for i, q in enumerate(qids)], np.float64)


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
    qids, golds, folds, stress, docs, short, x45, x, y, qe, cent = ait.make_world()
    base = np.empty((len(qids), ait.FULL), np.float32)
    final = np.empty_like(base)
    fold_meta = {}
    for outer, fn in enumerate(folds):
        tr, held, btr, bhe = sw.inner_crossfit_prior(x45, y, folds, qids, fn)
        pred, meta = fit_predict(outer, tr, held, btr, bhe, short, x, y, qe, cent, args)
        score = ait.zrows(bhe)
        score[:, :ait.DEPTH] = pred
        final[held] = score
        base[held] = bhe
        fold_meta[fn] = meta
    baseline = ait.rank(short, base)
    prior_head = ait.rank(short, np.load(ait.CACHE / "gold_head_oof_scores30.f32.npy"))
    candidate = ait.rank(short, final)
    metrics = {"ce_lr": b1.eval_rank(baseline, qids, golds, docs, folds, stress),
               "pointwise_centroid": b1.eval_rank(prior_head, qids, golds, docs, folds, stress),
               "setwise_centroid": b1.eval_rank(candidate, qids, golds, docs, folds, stress)}
    if abs(metrics["ce_lr"]["overall"]["recall_at_5"] - ait.BASE_R) > 3e-6:
        raise RuntimeError("CE-LR baseline parity failed")
    oldq, newq = perq(prior_head, qids, golds, docs), perq(candidate, qids, golds, docs)
    old, new = metrics["pointwise_centroid"], metrics["setwise_centroid"]
    effect = {"delta_vs_pointwise": new["overall"]["recall_at_5"] - old["overall"]["recall_at_5"],
              "wins": int((newq > oldq).sum()), "losses": int((newq < oldq).sum()),
              "fold_deltas": {f: new["per_fold"][f]["recall_at_5"] - old["per_fold"][f]["recall_at_5"] for f in folds}}
    decision = ("PROMOTE_FOR_PRIVATE" if new["overall"]["recall_at_5"] >= .95
                and effect["wins"] >= effect["losses"]
                and new["overall"]["multi_gold_recall_at_5"] >= old["overall"]["multi_gold_recall_at_5"]
                else "KILL")
    np.save(CACHE / "setwise_centroid_oof_scores30.f32.npy", final)
    report = {"schema": "stage07u.gold_embedding_setwise.v1", "teacher_used": False,
              "distillation_used": False, "supervision": "gold_labels_only",
              "raw_embedding_source": "stage02b4_vi_screen/aiteamvn_v1/region_embeddings",
              "metrics": metrics, "effect": effect, "fold_meta": fold_meta,
              "decision": decision}
    (OUT / "OOF_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DECISION", decision, "R", new["overall"]["recall_at_5"],
          "delta", effect["delta_vs_pointwise"], "W/L", effect["wins"], effect["losses"], flush=True)


if __name__ == "__main__":
    main()
