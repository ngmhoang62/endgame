"""Build the standalone A100 gold-fulltrain + gated-private Colab notebook."""
from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "notebooks/stage07n_a100_fulltrain_and_private.ipynb"
sources = {
    "trainer": (root / "artifacts/stage07b/run_qwen06b_gold_supervised_l4.py").read_text(encoding="utf-8"),
    "fulltrain": (root / "src/stage07_local/run_gold_qwen_fulltrain_a100.py").read_text(encoding="utf-8"),
    "private": (root / "src/stage07_local/run_gold_qwen_private_colab.py").read_text(encoding="utf-8"),
}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


cells = [
    md("# Stage07N — A100 gold fulltrain + private inference\n"
       "Chọn **Runtime → Change runtime type → A100 GPU**. Train trên cả 6991 gold query bằng model gốc "
       "`Qwen/Qwen3-Reranker-0.6B`, không có distillation. Session L4 tiếp tục DEV/CERT độc lập. "
       "Checkpoint A100 chỉ được dùng để nộp sau khi L4 báo `PROMOTE_TO_FULLTRAIN`.\n"),
    code("import torch\n"
         "assert torch.cuda.is_available(), 'Choose a GPU runtime'\n"
         "gpu = torch.cuda.get_device_name(0)\n"
         "print('GPU:', gpu)\n"
         "assert 'A100' in gpu, f'Expected A100, got {gpu}'\n"),
    code("from google.colab import drive\n"
         "drive.mount('/content/drive')\n"),
    md("## Stage07B gold payload từ Drive\n"
       "Cell này copy archive 183 MB đã dùng cho L4 từ `MyDrive/DSC2026/stage07b/`, xác minh SHA256, rồi giải nén vào `/content/stage07b/payload/`.\n"),
    code("from pathlib import Path\n"
         "import hashlib, shutil, json, subprocess, os, zipfile\n"
         "drive_root = Path('/content/drive/MyDrive/DSC2026')\n"
         "stage = drive_root / 'stage07b'\n"
         "def sha256(path):\n"
         "    h = hashlib.sha256()\n"
         "    with Path(path).open('rb') as f:\n"
         "        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)\n"
         "    return h.hexdigest()\n"
         "source = stage / 'stage07b_colab_payload.tar.gz'\n"
         "assert source.is_file(), f'Missing gold payload: {source}'\n"
         "assert sha256(source) == '6649f81738e1f437367bde040b1f8ac55a6d2d1fbd777998df3bd06148084eb3'\n"
         "local_gold = Path('/content/stage07b_colab_payload.tar.gz')\n"
         "shutil.copy2(source, local_gold)\n"
         "assert sha256(local_gold) == sha256(source)\n"
         "print('Gold payload copied:', local_gold, local_gold.stat().st_size)\n"),
    code("%cd /content\n"
         "!tar -xzf stage07b_colab_payload.tar.gz\n"
         "!ls -lh /content/stage07b/payload/MANIFEST.json /content/stage07b/payload/evidence_top20.pkl\n"),
    md("## Copy và giải nén script bundle từ thư mục con trên Drive\n"
       "Bundle nằm ở `MyDrive/DSC2026/stage07b/stage07n_a100_fulltrain/`; notebook và ba script đi cùng nhau. Fulltrain chỉ đổi training folds thành 0–4 và output folder, giữ nguyên gold loss và evidence.\n"),
    code("bundle_source = stage / 'stage07n_a100_fulltrain/stage07n_a100_fulltrain_bundle.zip'\n"
         "assert bundle_source.is_file(), f'Missing script bundle: {bundle_source}'\n"
         "bundle_local = Path('/content/stage07n_a100_fulltrain_bundle.zip')\n"
         "shutil.copy2(bundle_source, bundle_local)\n"
         "print('Script bundle copied:', bundle_local, bundle_local.stat().st_size)\n"),
    code("%cd /content\n"
         "!unzip -oq stage07n_a100_fulltrain_bundle.zip -d /content\n"
         "!ls -lh /content/stage07b/run_qwen06b_gold_supervised_l4.py /content/run_gold_qwen_fulltrain_a100.py /content/run_gold_qwen_private_colab.py\n"),
    code("expected_scripts = {\n"
         + "\n".join(f"    {path!r}: {hashlib.sha256(source.encode('utf-8')).hexdigest()!r}," for path, source in {
             '/content/stage07b/run_qwen06b_gold_supervised_l4.py': sources['trainer'],
             '/content/run_gold_qwen_fulltrain_a100.py': sources['fulltrain'],
             '/content/run_gold_qwen_private_colab.py': sources['private'],
         }.items()) + "\n}\n"
         "for path, expected_hash in expected_scripts.items():\n"
         "    assert sha256(Path(path)) == expected_hash, f'Script hash mismatch: {path}'\n"
         "print('All script hashes verified')\n"),
    code("!pip -q install 'transformers>=4.51.0,<5' accelerate safetensors scikit-learn\n"),
    md("## Start A100 fulltrain now\n"
       "L4 có thể vẫn đang train/evaluate. A100 lưu checkpoint ngay vào `MyDrive/DSC2026/stage07n_full_gold_qwen_a100/`. "
       "Cell này có thể mất khoảng 30–90 phút; thời gian thực phụ thuộc A100.\n"),
    code("train_log = drive_root / 'stage07n_full_gold_qwen_a100/FULLTRAIN_CONSOLE.log'\n"
         "train_log.parent.mkdir(parents=True, exist_ok=True)\n"
         "with train_log.open('a', encoding='utf-8') as log:\n"
         "    process = subprocess.Popen(['python', '-u', '/content/run_gold_qwen_fulltrain_a100.py'],\n"
         "                               cwd='/content', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
         "                               text=True, bufsize=1)\n"
         "    for line in process.stdout:\n"
         "        print(line, end='')\n"
         "        log.write(line); log.flush()\n"
         "    exit_code = process.wait()\n"
         "assert exit_code == 0, f'Fulltrain failed, exit {exit_code}; inspect {train_log}'\n"
         "print('A100 fulltrain checkpoint:', drive_root / 'stage07n_full_gold_qwen_a100/train012_model')\n"),
    md("## Sau khi L4 DEV/CERT PASS: private inference\n"
       "Chỉ chạy các cell dưới khi `MyDrive/DSC2026/stage07n_t4_gold_qwen/REPORT.json` có `decision=PROMOTE_TO_FULLTRAIN`. "
       "Private inference dùng checkpoint A100 fulltrain, output ZIP lưu về Drive.\n"),
    code("gate_path = drive_root / 'stage07n_t4_gold_qwen/REPORT.json'\n"
         "assert gate_path.is_file(), 'L4 chưa xong DEV/CERT'\n"
         "gate = json.loads(gate_path.read_text(encoding='utf-8'))\n"
         "print('L4 decision:', gate.get('decision', gate.get('status')))\n"
         "print('DEV:', gate.get('dev', {}).get('delta_recall'))\n"
         "print('CERT:', (gate.get('cert') or {}).get('delta_recall'))\n"
         "assert gate.get('decision') == 'PROMOTE_TO_FULLTRAIN', 'Gate failed; do not submit A100 fulltrain'\n"
         "private_src = stage / 'stage07n_private_payload.tar.gz'\n"
         "assert private_src.is_file(), f'Missing private evidence: {private_src}'\n"
         "assert sha256(private_src) == 'c2743b544fc885f1c1287375ba45a6a9cd8ecf9e33cd2e135f007f45d8ad77b9'\n"
         "private_local = Path('/content/stage07n_private_payload.tar.gz')\n"
         "shutil.copy2(private_src, private_local)\n"
         "assert sha256(private_local) == sha256(private_src)\n"),
    code("%cd /content\n"
         "!tar -xzf stage07n_private_payload.tar.gz\n"
         "!ls -lh /content/stage07n_private_payload/MANIFEST.json\n"),
    code("environment = os.environ.copy()\n"
         "environment['STAGE07N_CHECKPOINT_ROOT'] = str(drive_root / 'stage07n_full_gold_qwen_a100')\n"
         "infer_log = drive_root / 'stage07n_full_gold_qwen_a100/PRIVATE_CONSOLE.log'\n"
         "with infer_log.open('a', encoding='utf-8') as log:\n"
         "    process = subprocess.Popen(['python', '-u', '/content/run_gold_qwen_private_colab.py'],\n"
         "                               cwd='/content', env=environment, stdout=subprocess.PIPE,\n"
         "                               stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
         "    for line in process.stdout:\n"
         "        print(line, end='')\n"
         "        log.write(line); log.flush()\n"
         "    exit_code = process.wait()\n"
         "assert exit_code == 0, f'Private inference failed, exit {exit_code}; inspect {infer_log}'\n"),
    code("out = drive_root / 'stage07n_full_gold_qwen_a100/private_inference'\n"
         "zp = out / 'GOLD_QWEN06B_FULLTRAIN_PRIVATE_K5.zip'\n"
         "report = json.loads((out / 'PRIVATE_SUBMISSION_REPORT.json').read_text(encoding='utf-8'))\n"
         "with zipfile.ZipFile(zp) as z:\n"
         "    assert z.namelist() == ['submission.json']\n"
         "    rows = json.loads(z.read('submission.json'))\n"
         "assert len(rows) == 2080 and all(len(v['answer']) == 5 and len(set(v['answer'])) == 5 for v in rows.values())\n"
         "assert report['teacher_used'] is False and report['distillation_used'] is False\n"
         "print('READY:', zp)\n"
         "print('SHA256:', sha256(zp), 'queries:', len(rows))\n"),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"name": target.name, "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
bundle = root / "artifacts/stage07n_a100_fulltrain_bundle.zip"
entries = {
    "notebooks/stage07n_a100_fulltrain_and_private.ipynb": target.read_bytes(),
    "stage07b/run_qwen06b_gold_supervised_l4.py": sources["trainer"].encode("utf-8"),
    "run_gold_qwen_fulltrain_a100.py": sources["fulltrain"].encode("utf-8"),
    "run_gold_qwen_private_colab.py": sources["private"].encode("utf-8"),
}
with zipfile.ZipFile(bundle, "w") as z:
    for name, data in entries.items():
        info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, data)
print(target, target.stat().st_size, "bytes", len(cells), "cells")
print(bundle, bundle.stat().st_size, "bytes")
