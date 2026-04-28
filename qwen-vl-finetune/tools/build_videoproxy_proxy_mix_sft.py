#!/usr/bin/env python3
"""Convert VideoProxy mixed EasyR1/verl JSONL into Qwen-VL SFT JSONL.

Source records are the final mixed Proxy-training rows from ``train``:

    {"prompt": "...<video>...", "answer": "...", "videos": [...], ...}

The output follows qwen-vl-finetune's ShareGPT-style schema:

    {
      "video": "... or [...]",
      "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
      ],
      "metadata": {...}
    }

Answer normalization is intentionally conservative:
- segmentation answers with ``<events>`` are preserved;
- answers already wrapped in ``<answer>`` are preserved;
- if the prompt asks for ``<answer>`` but the ground truth is a bare choice or
  sort string, the answer is wrapped as ``<answer>...</answer>``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ANSWER_TAG_RE = re.compile(r"<answer>\s*[\s\S]*?\s*</answer>", re.IGNORECASE)
EVENTS_TAG_RE = re.compile(r"<events>\s*[\s\S]*?\s*</events>", re.IGNORECASE)
VIDEO_TOKEN = "<video>"

ANSWER_TAG_PROBLEM_PREFIXES = (
    "llava_mcq",
    "event_logic_",
    "seg_aot_",
    "sort",
)

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(clean_text(item.get("text")))
                elif item_type == "video" or "video" in item:
                    parts.append(VIDEO_TOKEN)
                elif item_type == "image" or "image" in item:
                    parts.append("<image>")
        return "\n".join(part for part in parts if part)
    return clean_text(content)


def extract_prompt(row: dict[str, Any]) -> str:
    prompt = clean_text(row.get("prompt"))
    if prompt:
        return prompt

    messages = row.get("messages") or []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = clean_text(message.get("role") or message.get("from")).lower()
            if role in {"user", "human"}:
                prompt = content_to_text(message.get("content") or message.get("value"))
                if prompt:
                    return prompt
    return ""


def extract_answer(row: dict[str, Any]) -> str:
    answer = clean_text(row.get("answer"))
    if answer:
        return answer
    return clean_text(row.get("ground_truth"))


def looks_like_frame_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return Path(value).suffix.lower() in FRAME_EXTENSIONS


def extract_videos(row: dict[str, Any], prompt: str) -> list[Any]:
    raw = row.get("videos")
    if raw is None:
        raw = row.get("video")
    if raw is None:
        raw = row.get("video_path")

    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        return []
    if not raw:
        return []

    placeholder_count = prompt.count(VIDEO_TOKEN)
    if placeholder_count == 1 and all(looks_like_frame_path(item) for item in raw):
        return [raw]
    return list(raw)


def prompt_requires_answer_tag(prompt: str, problem_type: str) -> bool:
    if "<answer" in prompt.lower() or "</answer>" in prompt.lower():
        return True
    return any(problem_type.startswith(prefix) for prefix in ANSWER_TAG_PROBLEM_PREFIXES)


def normalize_answer(answer: str, prompt: str, problem_type: str) -> tuple[str, str]:
    answer = clean_text(answer)
    if not answer:
        return "", "empty"
    if EVENTS_TAG_RE.search(answer) or ANSWER_TAG_RE.search(answer):
        return answer, "preserved"
    if prompt_requires_answer_tag(prompt, problem_type):
        return f"<answer>{answer}</answer>", "wrapped_answer_tag"
    return answer, "preserved"


def ensure_video_placeholders(prompt: str, videos: list[Any]) -> tuple[str, list[Any]]:
    count = prompt.count(VIDEO_TOKEN)
    if not videos:
        return prompt, videos
    if count == len(videos):
        return prompt, videos
    if count == 0:
        if len(videos) == 1:
            return f"{VIDEO_TOKEN}\n{prompt}", videos
        prefix = "\n".join(f"Video {idx + 1}:\n{VIDEO_TOKEN}" for idx in range(len(videos)))
        return f"{prefix}\n\n{prompt}", videos
    if count < len(videos):
        return prompt, videos[:count]
    return prompt, videos


def qwen_video_value(videos: list[Any]) -> Any:
    if len(videos) == 1 and isinstance(videos[0], str):
        return videos[0]
    return videos


def convert_record(
    row: dict[str, Any],
    source_path: str = "",
    line_no: int = 0,
) -> dict[str, Any] | None:
    prompt = extract_prompt(row)
    answer = extract_answer(row)
    problem_type = clean_text(row.get("problem_type") or row.get("task_type"))
    if not prompt or not answer:
        return None

    videos = extract_videos(row, prompt)
    prompt, videos = ensure_video_placeholders(prompt, videos)
    if prompt.count(VIDEO_TOKEN) > len(videos):
        return None

    answer, normalization = normalize_answer(answer, prompt, problem_type)
    if not answer:
        return None

    metadata = {
        "source": "videoproxy_proxy_mix_easyr1",
        "source_path": source_path,
        "source_line": line_no,
        "source_uid": clean_text(row.get("uid") or row.get("id")),
        "source_problem_type": problem_type,
        "answer_normalization": normalization,
    }
    source_meta = row.get("metadata")
    if isinstance(source_meta, dict):
        for key in (
            "domain_l1",
            "domain_l2",
            "clip_key",
            "task",
            "data_source",
            "experiment_frame_sampling",
            "video_fps_override",
            "shared_source_frames",
            "offline_frame_extraction",
            "level",
            "l1_fps",
        ):
            if key in source_meta:
                metadata[key] = source_meta[key]

    return {
        "video": qwen_video_value(videos),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "metadata": metadata,
    }


def convert_rows(rows: list[dict[str, Any]], source_path: str) -> tuple[list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for idx, row in enumerate(rows, 1):
        record = convert_record(row, source_path=source_path, line_no=idx)
        if record is None:
            stats["skipped"] += 1
            continue
        records.append(record)
        stats["kept"] += 1
        problem_type = record.get("metadata", {}).get("source_problem_type") or "unknown"
        stats[f"problem_type:{problem_type}"] += 1
        if record.get("metadata", {}).get("answer_normalization") == "wrapped_answer_tag":
            stats["answer_tag_wrapped"] += 1
    return records, stats


def update_stats(stats: Counter, record: dict[str, Any] | None) -> None:
    if record is None:
        stats["skipped"] += 1
        return
    stats["kept"] += 1
    problem_type = record.get("metadata", {}).get("source_problem_type") or "unknown"
    stats[f"problem_type:{problem_type}"] += 1
    if record.get("metadata", {}).get("answer_normalization") == "wrapped_answer_tag":
        stats["answer_tag_wrapped"] += 1


def stream_convert_jsonl(
    input_path: Path,
    output_path: Path,
    max_records: int = 0,
    progress_interval: int = 1000,
) -> tuple[int, int, Counter]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    read_rows = 0
    written = 0

    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            read_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {input_path}:{line_no}: {exc}") from exc

            record = convert_record(row, source_path=str(input_path), line_no=line_no) if isinstance(row, dict) else None
            update_stats(stats, record)
            if record is not None:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if max_records > 0 and written >= max_records:
                    break

            if progress_interval > 0 and read_rows % progress_interval == 0:
                print(
                    f"[proxy-mix-sft] processed={read_rows} written={written} skipped={stats['skipped']}",
                    file=sys.stderr,
                    flush=True,
                )

    if progress_interval > 0:
        print(
            f"[proxy-mix-sft] processed={read_rows} written={written} skipped={stats['skipped']}",
            file=sys.stderr,
            flush=True,
        )
    return written, read_rows, stats


def selected_stats(records: Iterable[dict[str, Any]], skipped: int = 0) -> Counter:
    stats: Counter = Counter({"skipped": skipped})
    for record in records:
        update_stats(stats, record)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert final VideoProxy mixed Proxy JSONL to Qwen-VL SFT JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True, help="Input mixed EasyR1/verl JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output Qwen-VL ShareGPT JSONL")
    parser.add_argument("--max-records", type=int, default=0, help="Maximum rows to write; 0 means all")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying max-records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print streaming conversion progress every N source rows; <=0 disables progress logs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"input-jsonl not found: {input_jsonl}")

    output_jsonl = Path(args.output_jsonl)
    if not args.shuffle:
        written, read_rows, stats = stream_convert_jsonl(
            input_path=input_jsonl,
            output_path=output_jsonl,
            max_records=args.max_records,
            progress_interval=args.progress_interval,
        )
    else:
        print("[proxy-mix-sft] --shuffle enabled; loading all records before writing.", file=sys.stderr)
        rows = load_jsonl(input_jsonl)
        records, stats = convert_rows(rows, source_path=str(input_jsonl))
        random.Random(args.seed).shuffle(records)
        if args.max_records > 0:
            records = records[: args.max_records]
        written = write_jsonl(records, output_jsonl)
        read_rows = len(rows)
        stats = selected_stats(records, skipped=stats["skipped"])

    task_counts = {
        key.removeprefix("problem_type:"): value
        for key, value in sorted(stats.items())
        if key.startswith("problem_type:")
    }
    task_summary = ", ".join(f"{key}={value}" for key, value in task_counts.items())
    print(f"Wrote {written} records from {read_rows} processed rows to {output_jsonl}")
    print(f"kept={stats['kept']} skipped={stats['skipped']} answer_tag_wrapped={stats['answer_tag_wrapped']}")
    if task_summary:
        print(f"Problem types: {task_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
