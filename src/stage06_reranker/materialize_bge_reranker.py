#!/usr/bin/env python
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MODEL_ID="BAAI/bge-reranker-v2-m3"
TARGET=ROOT/"models/rerankers/bge-reranker-v2-m3"
OUT=ROOT/"reports/stage06b0_bge_reranker_materialization"

def stable(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def main():
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    OUT.mkdir(parents=True,exist_ok=True)
    info=HfApi().model_info(MODEL_ID)
    revision=str(info.sha)
    print("resolved revision:",revision,flush=True)

    if not (TARGET.exists() and any(TARGET.iterdir())):
        TARGET.mkdir(parents=True,exist_ok=True)
        snapshot_download(repo_id=MODEL_ID,revision=revision,local_dir=str(TARGET))

    cfg=TARGET/"config.json"; tokcfg=TARGET/"tokenizer_config.json"; weights=TARGET/"model.safetensors"
    for p in (cfg,tokcfg,weights):
        if not p.is_file(): raise FileNotFoundError(p)

    total=0
    with safe_open(str(weights),framework="pt",device="cpu") as f:
        for k in f.keys():
            n=1
            for d in f.get_slice(k).get_shape(): n*=int(d)
            total+=n

    config=json.loads(cfg.read_text(encoding="utf-8"))
    tcfg=json.loads(tokcfg.read_text(encoding="utf-8"))
    manifest={
        "schema":"dsc2026.endgame.stage06b0.bge_reranker_materialization.v1",
        "status":"PASS","model_id":MODEL_ID,"resolved_revision_sha":revision,
        "target_path":str(TARGET),"parameter_count":int(total),"parameter_billions":total/1e9,
        "architecture":config.get("architectures"),"hidden_size":config.get("hidden_size"),
        "num_hidden_layers":config.get("num_hidden_layers"),
        "max_position_embeddings":config.get("max_position_embeddings"),
        "tokenizer_model_max_length":tcfg.get("model_max_length"),
        "screen_contract":{"candidate_max_lengths":[512,1024],
                           "official_model_card_recommended_max_length":1024,
                           "dtype":"float16"},
    }
    manifest["fingerprint"]=stable(manifest)
    (OUT/"MODEL_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("="*88); print("PASS",MODEL_ID); print("revision",revision)
    print("params",total,f"({total/1e9:.3f}B)"); print("manifest",OUT/"MODEL_MANIFEST.json"); print("="*88)

if __name__=="__main__":
    main()
