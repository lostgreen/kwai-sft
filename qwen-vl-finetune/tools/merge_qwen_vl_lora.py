#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a full Qwen-VL Hugging Face model dir."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def has_adapter_files(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / ADAPTER_CONFIG).is_file()
        and any((path / name).is_file() for name in ADAPTER_WEIGHT_NAMES)
    )


def checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    step = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return step, mtime


def resolve_adapter_path(adapter_or_run_dir: str | Path) -> Path:
    """Return a PEFT adapter dir, accepting either final dir or run root."""

    root = Path(adapter_or_run_dir).expanduser()
    if has_adapter_files(root):
        return root

    if not root.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {root}")

    candidates = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and has_adapter_files(path)
    ]
    if candidates:
        return sorted(candidates, key=checkpoint_sort_key)[-1]

    raise FileNotFoundError(
        "No PEFT LoRA adapter found. Expected adapter_config.json plus one of "
        f"{', '.join(ADAPTER_WEIGHT_NAMES)} in {root} or its checkpoint-* dirs."
    )


def parse_torch_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"

    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        choices = ", ".join(["auto", *sorted(mapping)])
        raise ValueError(f"Unsupported dtype {value!r}. Choose from: {choices}") from exc


def load_base_model(
    base_model: str,
    dtype: Any,
    device_map: str | None,
    attn_implementation: str | None,
    trust_remote_code: bool,
):
    from transformers import AutoModelForImageTextToText

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)


def save_processor(
    base_model: str,
    requested_adapter_path: Path,
    resolved_adapter_path: Path,
    output_dir: Path,
    trust_remote_code: bool,
    processor_source: str | None,
) -> str:
    from transformers import AutoProcessor

    if processor_source:
        sources = [Path(processor_source).expanduser()]
    else:
        sources = [requested_adapter_path, resolved_adapter_path, Path(base_model)]

    errors: list[str] = []
    for source in dict.fromkeys(str(path) for path in sources):
        try:
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - depends on local HF files.
            errors.append(f"{source}: {exc}")
            continue
        processor.save_pretrained(output_dir)
        return source

    raise RuntimeError(
        "Failed to load a processor from adapter or base model.\n"
        + "\n".join(errors)
    )


def merge_lora(args: argparse.Namespace) -> int:
    from peft import PeftModel

    requested_adapter_path = Path(args.adapter).expanduser()
    adapter_path = resolve_adapter_path(requested_adapter_path)
    output_dir = Path(args.output_dir).expanduser()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_torch_dtype(args.dtype)
    device_map = None if args.device_map.lower() == "none" else args.device_map
    trust_remote_code = not args.no_trust_remote_code

    print(f"[merge] base model: {args.base_model}", file=sys.stderr)
    print(f"[merge] adapter:    {adapter_path}", file=sys.stderr)
    print(f"[merge] output:     {output_dir}", file=sys.stderr)

    base = load_base_model(
        base_model=args.base_model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    if hasattr(merged, "config"):
        merged.config.use_cache = True
    merged.save_pretrained(
        output_dir,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    processor_source = save_processor(
        base_model=args.base_model,
        requested_adapter_path=requested_adapter_path,
        resolved_adapter_path=adapter_path,
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        processor_source=args.processor_source,
    )
    print(f"[merge] processor source: {processor_source}", file=sys.stderr)
    print("[merge] done", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge qwen-vl-finetune PEFT LoRA output into a full HF model "
            "directory that eval/vLLM can load directly."
        )
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter",
        required=True,
        help=(
            "LoRA adapter dir, or the training output root containing "
            "checkpoint-* adapter dirs."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--processor-source", default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-safe-serialization", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return merge_lora(args)


if __name__ == "__main__":
    raise SystemExit(main())
