from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ENCODER_FILES = ("backbone/config.json", "heads.pt", "label_map.json", "thresholds.json", "taxonomy.json")
DECODER_FILES = ("adapter_config.json",)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _revision(name: str) -> str:
    value = _required(name)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise RuntimeError(f"{name} must be an immutable 40-character Hugging Face commit")
    return value.lower()


def _validate(root: Path, required: tuple[str, ...], kind: str) -> None:
    missing = [name for name in required if not (root / name).is_file()]
    if kind == "decoder" and not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise RuntimeError(f"Downloaded {kind} artifact is incomplete: {', '.join(missing)}")
    if kind == "decoder":
        adapter = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
        base_revision = str(adapter.get("revision") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", base_revision):
            raise RuntimeError(
                "Decoder adapter_config.json must pin its base model with a 40-character revision"
            )


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

    model_root = Path(os.environ.get("PALDA_MODEL_ROOT", "/models"))
    encoder_repo = _required("PALDA_ENCODER_REPO")
    decoder_repo = _required("PALDA_DECODER_REPO")
    encoder_revision = _revision("PALDA_ENCODER_REVISION")
    decoder_revision = _revision("PALDA_DECODER_REVISION")
    decoder_base_repo = _required("PALDA_DECODER_BASE_REPO")
    decoder_base_revision = _revision("PALDA_DECODER_BASE_REVISION")
    encoder_dir = model_root / f"encoder-{encoder_revision[:12]}"
    decoder_dir = model_root / f"decoder-{decoder_revision[:12]}"
    snapshot_download(encoder_repo, revision=encoder_revision, local_dir=encoder_dir)
    snapshot_download(decoder_repo, revision=decoder_revision, local_dir=decoder_dir)
    snapshot_download(
        decoder_base_repo,
        revision=decoder_base_revision,
        cache_dir=model_root / "hf-cache",
    )
    _validate(encoder_dir, ENCODER_FILES, "encoder")
    _validate(decoder_dir, DECODER_FILES, "decoder")
    adapter = json.loads((decoder_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter.get("base_model_name_or_path") != decoder_base_repo:
        raise RuntimeError("Decoder adapter base_model_name_or_path does not match deployment config")
    if str(adapter.get("revision") or "").lower() != decoder_base_revision:
        raise RuntimeError("Decoder adapter revision does not match deployment config")
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "encoder": {"repo": encoder_repo, "revision": encoder_revision, "path": str(encoder_dir)},
        "decoder": {"repo": decoder_repo, "revision": decoder_revision, "path": str(decoder_dir)},
        "decoderBase": {
            "repo": decoder_base_repo,
            "revision": decoder_base_revision,
            "cache": str(model_root / "hf-cache"),
        },
    }
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "deployment-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
