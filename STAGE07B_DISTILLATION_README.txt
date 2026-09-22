Stage07B v2 — Teacher→Student Distillation only

RULE:
- Qwen3-Reranker-4B is training-time teacher only.
- Never evaluate or submit teacher predictions.
- Final active component is Qwen3-Reranker-0.6B student.
- Existing active params ~1.73156B + student ~0.6B = ~2.33156B (<4B).

Workflow:
1) Open notebooks/stage07b_colab_main.ipynb in VS Code.
2) Connect notebook to Colab A100 runtime.
3) Mount/open /content as your remote workspace and ensure repo is /content/endgame.
4) Use Colab terminal:
   python src/stage07_breakthrough/run_qwen4b_teacher_targets_a100.py
5) Then:
   python src/stage07_breakthrough/run_qwen06b_distilled_oof_a100.py

Teacher target generation emits no recall/precision metrics by design.
Student script performs strict 5-fold OOF and reports target 0.96.
