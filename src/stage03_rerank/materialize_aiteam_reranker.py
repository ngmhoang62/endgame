#!/usr/bin/env python
"""Materialize a fresh pinned AITeamVN/Vietnamese_Reranker snapshot for ENDGAME."""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/"models/rerankers/aiteamvn-vietnamese-reranker"
OUT=ROOT/"reports/stage03b0_reranker_materialization"
MODEL_ID="AITeamVN/Vietnamese_Reranker"

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    from huggingface_hub import HfApi, snapshot_download
    info=HfApi().model_info(MODEL_ID)
    revision=str(info.sha)
    TARGET.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    print(f"[download] {MODEL_ID}@{revision} -> {TARGET}",flush=True)
    snapshot_download(
        MODEL_ID,revision=revision,local_dir=str(TARGET),
        allow_patterns=["*.json","*.txt","*.safetensors","*.bin","*.model","*.py",
                        "tokenizer*","vocab*","merges*","special_tokens_map*"],
        ignore_patterns=["onnx/*","openvino/*","*.onnx","*.h5","*.msgpack"],
    )

    import torch, transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(TARGET,local_files_only=True,trust_remote_code=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        TARGET,local_files_only=True,trust_remote_code=True,dtype=torch.float16
    ).eval().to("cuda")
    params=int(sum(p.numel() for p in model.parameters()))
    pairs=[
        ("Người lao động được nghỉ hằng năm bao nhiêu ngày?",
         "Người lao động làm đủ 12 tháng được nghỉ hằng năm theo quy định của pháp luật."),
        ("Mức phạt khi vi phạm quy định là bao nhiêu?",
         "Cơ quan có thẩm quyền thực hiện thủ tục cấp giấy phép."),
    ]
    if hasattr(model,"compute_score"):
        vals=model.compute_score(pairs,batch_size=2,max_length=512)
        if isinstance(vals,float): vals=[vals]
        vals=[float(x) for x in vals]
        scoring="model.compute_score(pairs,max_length=512)"
    else:
        enc=tok([a for a,b in pairs],[b for a,b in pairs],max_length=512,
                truncation=True,padding=True,return_tensors="pt")
        enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.inference_mode(): logits=model(**enc).logits
        if logits.ndim==2 and logits.shape[1]==1: vv=logits[:,0]
        elif logits.ndim==2: vv=logits[:,-1]
        else: vv=logits.reshape(-1)
        vals=[float(x) for x in vv.float().cpu()]
        scoring="AutoModelForSequenceClassification logits"
    if len(vals)!=2 or not all(map(lambda x: x==x,vals)):
        raise RuntimeError(f"smoke scores invalid: {vals}")

    files=[]
    for p in sorted(TARGET.rglob("*")):
        if p.is_file():
            files.append({"path":str(p.relative_to(TARGET)).replace("\\","/"),
                          "size_bytes":p.stat().st_size,"sha256":sha256(p)})
    manifest={"schema_version":"dsc2026.endgame.stage03b0.reranker_materialization.v1",
              "status":"PASS","model_id":MODEL_ID,"resolved_revision_sha":revision,
              "parameter_count":params,"parameter_billions":params/1e9,
              "competition_budget_fraction":params/4e9,
              "runtime":{"python":sys.version,"torch":torch.__version__,
                         "transformers":transformers.__version__,
                         "gpu":torch.cuda.get_device_name(0)},
              "contract":{"max_length":512,"historical_pair_packaging":"(question, selected_passage)",
                          "historical_passages_per_document":2,"scoring":scoring},
              "smoke_scores":vals,"files":files}
    (OUT/"MODEL_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","revision":revision,"params":params,
                      "parameter_billions":params/1e9,"smoke_scores":vals,
                      "out":str(OUT)},indent=2))

if __name__=="__main__":
    main()
