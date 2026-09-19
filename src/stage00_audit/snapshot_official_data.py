#!/usr/bin/env python
"""Snapshot official DSC2026 inputs into ENDGAME.

Purpose:
- remove live runtime dependence on sota/ or LegalIR/;
- preserve source hashes;
- copy official TRAIN/PUBLIC, ENDGAME PRIVATE, and selected-context corpus;
- refuse accidental overwrite unless --force.

Run:
    python src/stage00_audit/snapshot_official_data.py \
      --sota-root D:/Study/DSC2026/sota
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOTA = Path("D:/Study/DSC2026/sota")
TARGET = ROOT / "data" / "official_v1"
REPORT_DIR = ROOT / "reports" / "stage00_data_snapshot"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def file_row(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sota-root", type=Path, default=DEFAULT_SOTA)
    p.add_argument("--private", type=Path, default=ROOT / "private-official.json")
    p.add_argument("--target", type=Path, default=TARGET)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    sota = args.sota_root.resolve()
    source_data = sota / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset"
    train = source_data / "train.json"
    public = source_data / "public-official.json"
    contexts = source_data / "selected-contexts"
    private = args.private.resolve()
    target = args.target.resolve()

    required = [train, public, private, contexts]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing official source(s):\n" + "\n".join(missing))

    context_files = sorted(contexts.glob("context_*.json"))
    if not context_files:
        raise RuntimeError(f"No context_*.json files in {contexts}")

    if target.exists():
        if not args.force:
            raise FileExistsError(
                f"Target exists: {target}\n"
                "Refusing to overwrite official snapshot. Use --force only deliberately."
            )
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".official_v1-", dir=str(target.parent)))
    try:
        (temp / "selected-contexts").mkdir(parents=True)
        shutil.copy2(train, temp / "train.json")
        shutil.copy2(public, temp / "public-official.json")
        shutil.copy2(private, temp / "private-official.json")

        print(f"[snapshot] copying {len(context_files)} context files...", flush=True)
        for i, src in enumerate(context_files, 1):
            shutil.copy2(src, temp / "selected-contexts" / src.name)
            if i % 1000 == 0 or i == len(context_files):
                print(f"[snapshot] contexts {i}/{len(context_files)}", flush=True)

        files = sorted(p for p in temp.rglob("*") if p.is_file())
        inventory = [file_row(p, temp) for p in files]
        manifest = {
            "schema_version": "dsc2026.endgame.official_data_snapshot.v1",
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "sota_root": str(sota),
                "source_data_dir": str(source_data),
                "private_source": str(private),
            },
            "target": str(target),
            "runtime_dependency_on_historical_repo_after_snapshot": False,
            "files": inventory,
            "file_count": len(inventory),
            "context_file_count": len(context_files),
            "top_level_sha256": {
                "train.json": sha256(temp / "train.json"),
                "public-official.json": sha256(temp / "public-official.json"),
                "private-official.json": sha256(temp / "private-official.json"),
            },
        }
        payload = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest["content_fingerprint"] = hashlib.sha256(payload).hexdigest()
        (temp / "DATA_SNAPSHOT_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(target)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "DATA_SNAPSHOT.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (REPORT_DIR / "REPORT.md").write_text(
            "\n".join([
                "# ENDGAME official data snapshot",
                "",
                "- Status: **PASS**",
                f"- Context files: **{len(context_files)}**",
                f"- Target: `{target}`",
                f"- Fingerprint: `{manifest['content_fingerprint']}`",
                "- Historical repo runtime dependency after snapshot: **false**",
                "",
            ]),
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "PASS",
            "target": str(target),
            "context_file_count": len(context_files),
            "content_fingerprint": manifest["content_fingerprint"],
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
