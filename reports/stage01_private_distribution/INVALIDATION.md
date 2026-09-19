# Stage 01B v2 semantic-component invalidation

Status: **PARTIALLY INVALIDATED**

The Stage 01B v2 run loaded its embedding model from the historical
`D:/Study/DSC2026/sota/.../vietlegal-e5` bundle. During tokenizer loading,
Transformers emitted the `incorrect regex pattern` warning and recommended
`fix_mistral_regex=True`.

Scientific disposition:

- **KEEP**: query preflight, input fingerprints, schema checks, exact-text
  overlap, punctuation/accent-fold duplicate audit, and TF-IDF lexical
  diagnostics.
- **INVALIDATE**: v2 embedding nearest-neighbor similarities.
- **INVALIDATE**: v2 embedding-based semantic clusters.
- **DO NOT USE**: v2 embedding values for routing, fold design, promotion,
  model choice, or private-distribution conclusions.

The authoritative semantic diagnostic is Stage 01B v3, which requires a fresh,
revision-pinned Hugging Face snapshot under `endgame/models/`, uses
SentenceTransformer's native model architecture, explicitly initializes the
processor/tokenizer with `fix_mistral_regex=True`, and writes to a new cache
namespace.
