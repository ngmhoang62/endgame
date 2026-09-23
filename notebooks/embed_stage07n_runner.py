"""Embed the current standalone Stage07N inference runner in its A100 notebook."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb_path = root / "notebooks/stage07n_a100_private_inference.ipynb"
runner = root / "src/stage07_local/run_gold_qwen_private_colab.py"
notebook = json.loads(nb_path.read_text(encoding="utf-8"))
source = runner.read_text(encoding="utf-8")
copy_cell = notebook["cells"][4]
copy_cell["source"] = [line.replace(
    "'run_gold_qwen_private_colab.py': '83a3b6e0d498091bc05c1eb55943d684a310f0a7d5685140b0dc1d2cc5da559e',",
    "'run_gold_qwen_private_colab.py': None,").replace(
    "assert sha256(source) == expected_hash, f'Wrong or incomplete upload: {source}'",
    "assert expected_hash is None or sha256(source) == expected_hash, f'Wrong or incomplete upload: {source}'").replace(
    "assert sha256(destination) == expected_hash",
    "assert expected_hash is None or sha256(destination) == expected_hash")
    for line in copy_cell["source"]]
assert any("'run_gold_qwen_private_colab.py': None" in line for line in copy_cell["source"])
embed = {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "# Embed the updated standalone runner; the earlier Drive upload is accepted.\n",
        "runner_source = " + repr(source) + "\n",
        "runner_path = Path('/content/run_gold_qwen_private_colab.py')\n",
        "runner_path.write_text(runner_source, encoding='utf-8')\n",
        "print('Updated standalone runner:', runner_path, sha256(runner_path))\n",
    ],
}
if any("runner_source = " in line for line in notebook["cells"][5]["source"]):
    notebook["cells"][5] = embed
else:
    notebook["cells"].insert(5, embed)
intro = "Notebook đã nhúng runner mới độc lập; không cần upload lại file Python trước đó.\n"
if intro not in notebook["cells"][0]["source"]:
    notebook["cells"][0]["source"].append(intro)
nb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("runner_sha256", hashlib.sha256(source.encode("utf-8")).hexdigest(), "cells", len(notebook["cells"]))
