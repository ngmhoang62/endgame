#!/usr/bin/env python
"""Export tiny W75 score bridge for final Qwen+W75 Colab audit."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import src.stage03_rerank.run_aiteam_reranker_oof as b1
import src.stage07_local.run_gold_centroid_head_oof as ait
import src.stage07_local.run_final_clean_fusion_private as yf

OUT=ROOT/"artifacts/stage07ad_w75_qwen_bridge.npz"
P_PROP=ROOT/"reports/stage07r_gold_doc_propensity/propensity_oof_scores30.f32.npy"
P_AIT=ROOT/"cache/stage07o_gold_centroid_head/gold_head_oof_scores30.f32.npy"

def main():
    qids,_,_,_,_,docs,_=b1.load_world()
    short,_=b1.load_shortlist(len(qids))
    prop=np.load(P_PROP); a=np.load(P_AIT)
    w75=(.75*ait.zrows(prop)+.25*ait.zrows(a)).astype(np.float32)

    print("[private] reconstruct exact W75 score world",flush=True)
    pqids,pdocs,pshort,pps,PX45,PX57,pqe,pcent=yf.private_propensity()
    pas=yf.private_ait_score(pqids,pdocs,pshort,PX45,PX57,pqe,pcent)
    pw75=(.75*ait.zrows(pps)+.25*ait.zrows(pas)).astype(np.float32)

    if pdocs!=docs: raise RuntimeError("doc order drift")
    np.savez_compressed(
        OUT,
        public_short=np.asarray(short,np.int32),
        public_w75=w75,
        private_short=np.asarray(pshort,np.int32),
        private_w75=pw75,
    )
    print("READY",OUT,flush=True)
    print("bytes",OUT.stat().st_size,flush=True)

if __name__=="__main__":
    main()
