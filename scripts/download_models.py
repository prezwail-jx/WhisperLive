#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path


ASR_MODELS = {
    "tiny": {
        "local_dir": "asr/tiny",
        "modelscope": "Systran/faster-whisper-tiny",
        "huggingface": "Systran/faster-whisper-tiny",
    },
    "tiny.en": {
        "local_dir": "asr/tiny.en",
        "modelscope": "Systran/faster-whisper-tiny.en",
        "huggingface": "Systran/faster-whisper-tiny.en",
    },
    "base": {
        "local_dir": "asr/base",
        "modelscope": "Systran/faster-whisper-base",
        "huggingface": "Systran/faster-whisper-base",
    },
    "base.en": {
        "local_dir": "asr/base.en",
        "modelscope": "Systran/faster-whisper-base.en",
        "huggingface": "Systran/faster-whisper-base.en",
    },
    "small": {
        "local_dir": "asr/small",
        "modelscope": "Systran/faster-whisper-small",
        "huggingface": "Systran/faster-whisper-small",
    },
    "small.en": {
        "local_dir": "asr/small.en",
        "modelscope": "Systran/faster-whisper-small.en",
        "huggingface": "Systran/faster-whisper-small.en",
    },
    "medium": {
        "local_dir": "asr/medium",
        "modelscope": "Systran/faster-whisper-medium",
        "huggingface": "Systran/faster-whisper-medium",
    },
    "medium.en": {
        "local_dir": "asr/medium.en",
        "modelscope": "Systran/faster-whisper-medium.en",
        "huggingface": "Systran/faster-whisper-medium.en",
    },
    "large-v3-turbo": {
        "local_dir": "asr/large-v3-turbo",
        # ModelScope 上我建议用这个，能搜到明确页面
        "modelscope": "pengzhendong/faster-whisper-large-v3-turbo",
        # 你原来这个 HuggingFace 源保持不变
        "huggingface": "Systran/faster-whisper-large-v3-turbo",
    },
    "large-v3": {
        "local_dir": "asr/large-v3",
        "modelscope": "Systran/faster-whisper-large-v3",
        "huggingface": "Systran/faster-whisper-large-v3",
    },
}

TRANSLATION_MODELS = {
    "opus-mt-zh-en": {
        "local_dir": "opus-mt-zh-en",
        "modelscope": None,
        "huggingface": "Helsinki-NLP/opus-mt-zh-en",
    },
    "opus-mt-en-zh": {
        "local_dir": "opus-mt-en-zh",
        "modelscope": None,
        "huggingface": "Helsinki-NLP/opus-mt-en-zh",
    },
}

ASR_REQUIRED = ("config.json",)
ASR_REQUIRED_ANY = ("model.bin", "pytorch_model.bin")
TRANSLATION_REQUIRED = ("config.json", "source.spm", "target.spm", "vocab.json")


def load_manifest_override(path):
    if path is None:
        return
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for group_name, target in (("asr", ASR_MODELS), ("translation", TRANSLATION_MODELS)):
        for name, values in data.get(group_name, {}).items():
            if name not in target:
                target[name] = {}
            target[name].update(values)


def validate_model(path, model_type):
    path = Path(path)
    if not path.is_dir():
        return False
    if model_type == "asr":
        return all((path / item).exists() for item in ASR_REQUIRED) and any(
            (path / item).exists() for item in ASR_REQUIRED_ANY
        )
    return all((path / item).exists() for item in TRANSLATION_REQUIRED)


def ensure_clean_target(path, force):
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_from_modelscope(model_id, target_dir):
    if not model_id:
        raise ValueError("ModelScope model id is not configured")
    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "modelscope is not installed. Run: pip install -r requirements/model_download.txt"
        ) from error
    snapshot_download(model_id=model_id, local_dir=str(target_dir))


def download_from_huggingface(repo_id, target_dir):
    if not repo_id:
        raise ValueError("Hugging Face repo id is not configured")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install -r requirements/model_download.txt"
        ) from error
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(target_dir))


def download_one(name, spec, model_root, model_type, source, force):
    target_dir = model_root / spec["local_dir"]
    if validate_model(target_dir, model_type) and not force:
        print(f"[skip] {name}: already exists at {target_dir}")
        return

    ensure_clean_target(target_dir, force)
    errors = []
    sources = ("modelscope", "huggingface") if source == "auto" else (source,)

    for selected_source in sources:
        try:
            if selected_source == "modelscope":
                print(f"[download] {name}: ModelScope -> {target_dir}")
                download_from_modelscope(spec.get("modelscope"), target_dir)
            elif selected_source == "huggingface":
                print(f"[download] {name}: Hugging Face -> {target_dir}")
                download_from_huggingface(spec.get("huggingface"), target_dir)
            else:
                raise ValueError(f"Unsupported source: {selected_source}")

            if validate_model(target_dir, model_type):
                print(f"[ready] {name}: {target_dir}")
                return
            raise RuntimeError(f"Downloaded files do not look like a valid {model_type} model")
        except Exception as error:
            errors.append(f"{selected_source}: {error}")
            print(f"[warn] {name}: {errors[-1]}")
            if source != "auto":
                break

    raise RuntimeError(f"Failed to download {name}. " + " | ".join(errors))


def selected_models(args):
    items = []
    if args.all or args.asr:
        items.extend(("asr", name, spec) for name, spec in ASR_MODELS.items())
    if args.all or args.translation:
        items.extend(("translation", name, spec) for name, spec in TRANSLATION_MODELS.items())
    for name in args.model:
        if name in ASR_MODELS:
            items.append(("asr", name, ASR_MODELS[name]))
        elif name in TRANSLATION_MODELS:
            items.append(("translation", name, TRANSLATION_MODELS[name]))
        else:
            raise ValueError(f"Unknown model: {name}")
    if not items:
        items.extend(("translation", name, spec) for name, spec in TRANSLATION_MODELS.items())
    return items


def main():
    parser = argparse.ArgumentParser(description="Download WhisperLive ASR and translation models.")
    parser.add_argument("--model-dir", default="model", help="Root directory for downloaded models.")
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="Download source. auto tries ModelScope first, then Hugging Face.",
    )
    parser.add_argument("--all", action="store_true", help="Download ASR and translation models.")
    parser.add_argument("--asr", action="store_true", help="Download all ASR models.")
    parser.add_argument("--translation", action="store_true", help="Download translation models.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Download one model by name. Can be used multiple times.",
    )
    parser.add_argument("--manifest", help="Optional JSON file overriding model ids and local dirs.")
    parser.add_argument("--force", action="store_true", help="Remove target directory and redownload.")
    args = parser.parse_args()

    load_manifest_override(args.manifest)
    model_root = Path(args.model_dir)

    try:
        items = selected_models(args)
        for model_type, name, spec in items:
            download_one(name, spec, model_root, model_type, args.source, args.force)
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
