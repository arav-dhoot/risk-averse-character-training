"""Rewrite a PEFT adapter's tensor names for vLLM's multimodal model classes.

Some checkpoints (Qwen3.5 and Qwen3.8 among them) load in transformers as a
text-only causal LM, so PEFT saves adapter weights under
`base_model.model.model.layers.*`, while vLLM serves the same checkpoint as a
conditional-generation model and expects
`base_model.model.model.language_model.layers.*`. With the original names vLLM
matches no tensors and silently serves the base model. This writes a renamed
copy (the source adapter is left untouched) and prints its path; adapters that
already use the `language_model` prefix are returned unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

OLD = "base_model.model.model.layers."
NEW = "base_model.model.model.language_model.layers."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(src: Path, out_root: Path) -> Path:
    from safetensors.torch import load_file, save_file

    weights = src / "adapter_model.safetensors"
    tensors = load_file(str(weights), device="cpu")
    names = list(tensors)
    if any(".language_model." in k for k in names):
        return src
    bad = [k for k in names if not k.startswith(OLD)]
    if bad:
        raise SystemExit(f"{src}: unexpected tensor names: {bad[:3]}")
    dst = out_root / (src.resolve().parent.name + "__" + src.name)
    stamp = dst / "SOURCE_SHA256"
    src_sha = sha256(weights)
    if stamp.is_file() and stamp.read_text().strip() == src_sha \
            and (dst / "adapter_model.safetensors").is_file():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    save_file({NEW + k[len(OLD):]: v for k, v in tensors.items()},
              str(dst / "adapter_model.safetensors"),
              metadata={"format": "pt", "converted_from_sha256": src_sha,
                        "key_remap": f"{OLD} -> {NEW}"})
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    stamp.write_text(src_sha + "\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir", help="PEFT adapter directory")
    ap.add_argument("--out-root", default="runs/vllm-adapters",
                    help="where converted copies are written")
    args = ap.parse_args()
    print(convert(Path(args.adapter_dir), Path(args.out_root)))


if __name__ == "__main__":
    main()
