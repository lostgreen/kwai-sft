#!/usr/bin/env python3
"""Build Qwen-VL SFT JSONL from VideoProxy hierarchy annotations.

The output follows the qwen-vl-finetune ShareGPT-style schema:

    {
      "video": "/path/to/video_or_clip.mp4",
      "conversations": [
        {"from": "human", "value": "<video>\\n..."},
        {"from": "gpt", "value": "..."}
      ],
      "metadata": {...}
    }

By default this script builds one whole-video description record per annotation,
up to 10K videos, and records the intended training cap of 256 video frames in
metadata. The frame cap is enforced by the training launcher via
`--video_max_frames 256`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_TASKS = {"video_description", "dense_caption", "dense_video_caption"}

VISIBLE_GROUNDING_RULES = (
    "Every statement must be grounded in what is visible in the frames.\n"
    "Do not infer dialogue content, narration topics, brand purposes, or "
    "intent beyond visible evidence.\n"
    "If someone is talking to the camera, describe body language, gestures, "
    "and setting rather than speech content."
)

VIDEO_DESCRIPTION_PROMPT = (
    "<video>\n"
    "Write a detailed description of the entire video (3-5 sentences).\n"
    "Cover: setting/environment, main subjects, key objects, overall "
    "progression, and outcome.\n"
    f"{VISIBLE_GROUNDING_RULES}"
)

DENSE_CAPTION_PROMPT = (
    "<video>\n"
    "Write a dense visual caption for this continuous video event (2-4 sentences).\n"
    "Describe visible actions, objects, spatial relations, and visible state "
    "changes in chronological order.\n"
    f"{VISIBLE_GROUNDING_RULES}"
)

DENSE_VIDEO_CAPTION_PROMPT = (
    "<video>\n"
    "Write a dense video caption as a chronological list of timestamped mid-level visual events.\n"
    "Use visible scene or shot boundaries as anchors, then decide whether to keep, "
    "merge, or split them into events.\n"
    "Merge adjacent shots only when they show the same unbroken event in the same "
    "continuous space/time, such as a framing change, shot/reverse-shot, or continuous "
    "camera movement.\n"
    "Keep events separate when the location/time, main subject, activity step, object "
    "interaction, or resulting state clearly changes; keep title cards, intros/outros, "
    "static text, and cut-away/B-roll as separate visible events if present.\n"
    "Split a long continuous shot when it contains multiple distinct activities or "
    "sub-tasks; avoid sub-events shorter than about 5 seconds unless there is a clear "
    "scene/activity change.\n"
    "Use one line per event in the format [start_second-end_second] with plain "
    "integer seconds, followed by 8-20 words describing "
    "WHAT happens, WITH WHICH visible objects, and the visible outcome or state change.\n"
    "Keep each event concise, objective, and grounded in visible evidence.\n"
    "Do not add events, speech content, or intent that cannot be seen.\n"
    f"{VISIBLE_GROUNDING_RULES}"
)


@dataclass(frozen=True)
class BuildConfig:
    tasks: tuple[str, ...] = ("video_description",)
    max_videos: int = 10_000
    max_records: int = 0
    max_frames: int = 256
    l2_clip_dir: str = ""
    require_video_exists: bool = False
    shuffle: bool = False
    seed: int = 42


def parse_task_list(raw: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(tasks) - SUPPORTED_TASKS)
    if unknown:
        raise ValueError(
            f"Unsupported task(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_TASKS))}"
        )
    if not tasks:
        raise ValueError("At least one task must be selected")
    return tasks


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_annotation_paths(annotation_dir: Path) -> Iterable[Path]:
    yield from sorted(annotation_dir.glob("*.json"))


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def get_source_video_path(annotation: dict[str, Any]) -> str:
    return clean_text(
        annotation.get("source_video_path")
        or annotation.get("video_path")
        or annotation.get("video")
    )


def numeric_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    return str(total_seconds)


def path_time(value: Any) -> str:
    parsed = numeric_or_none(value)
    if parsed is None:
        return clean_text(value).replace("/", "_")
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def l2_event_clip_path(clip_dir: str, clip_key: str, event: dict[str, Any]) -> str:
    event_id = clean_text(event.get("event_id") or event.get("id"))
    start = path_time(event.get("start_time"))
    end = path_time(event.get("end_time"))
    return str(Path(clip_dir) / "L2" / f"{clip_key}_L2_ev{event_id}_{start}_{end}.mp4")


def iter_l2_events(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    events = annotation.get("level2", {}).get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def first_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    for idx, char in enumerate(text):
        if char in ".!?":
            return text[: idx + 1].strip()
    return text


def event_short_description(event: dict[str, Any]) -> tuple[str, str]:
    instruction = clean_text(event.get("instruction"))
    if instruction:
        return instruction, "instruction"

    dense_caption = first_sentence(event.get("dense_caption"))
    if dense_caption:
        return dense_caption, "dense_caption"

    return "", ""


def base_metadata(
    annotation: dict[str, Any],
    config: BuildConfig,
    task: str,
) -> dict[str, Any]:
    return {
        "source": "videoproxy_hier_seg_annotation",
        "task": task,
        "clip_key": clean_text(annotation.get("clip_key")),
        "clip_duration_sec": numeric_or_none(annotation.get("clip_duration_sec")),
        "domain_l1": clean_text(annotation.get("domain_l1") or "other"),
        "domain_l2": clean_text(annotation.get("domain_l2") or "other"),
        "topology": clean_text(annotation.get("topology_type")),
        "max_frames": config.max_frames,
    }


def make_record(
    video_path: str,
    prompt: str,
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video": video_path,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "metadata": metadata,
    }


def video_exists_if_required(video_path: str, config: BuildConfig) -> bool:
    if not config.require_video_exists:
        return True
    return bool(video_path) and Path(video_path).is_file()


def build_video_description_record(
    annotation: dict[str, Any],
    config: BuildConfig,
) -> dict[str, Any] | None:
    answer = clean_text(annotation.get("video_caption"))
    video_path = get_source_video_path(annotation)
    if not answer or not video_path or not video_exists_if_required(video_path, config):
        return None

    metadata = base_metadata(annotation, config, "video_description")
    return make_record(
        video_path=video_path,
        prompt=VIDEO_DESCRIPTION_PROMPT,
        answer=answer,
        metadata=metadata,
    )


def dense_caption_prompt_for_event(event: dict[str, Any], using_event_clip: bool) -> str:
    if using_event_clip:
        return DENSE_CAPTION_PROMPT

    start = numeric_or_none(event.get("start_time"))
    end = numeric_or_none(event.get("end_time"))
    if start is None or end is None:
        return DENSE_CAPTION_PROMPT
    return (
        f"{DENSE_CAPTION_PROMPT}\n"
        f"The target event spans approximately {start:g}s to {end:g}s in the "
        "source video. Focus on that interval."
    )


def build_dense_caption_records(
    annotation: dict[str, Any],
    config: BuildConfig,
) -> list[dict[str, Any]]:
    clip_key = clean_text(annotation.get("clip_key"))
    source_video = get_source_video_path(annotation)
    events = iter_l2_events(annotation)

    records: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: numeric_or_none(item.get("start_time")) or 0.0):
        if not isinstance(event, dict):
            continue

        answer = clean_text(event.get("dense_caption"))
        if not answer:
            continue

        using_event_clip = bool(config.l2_clip_dir)
        video_path = (
            l2_event_clip_path(config.l2_clip_dir, clip_key, event)
            if using_event_clip
            else source_video
        )
        if not video_path:
            continue
        if not video_exists_if_required(video_path, config):
            continue

        metadata = base_metadata(annotation, config, "dense_caption")
        metadata.update(
            {
                "event_id": event.get("event_id") or event.get("id"),
                "event_instruction": clean_text(event.get("instruction")),
                "segment_start_sec": numeric_or_none(event.get("start_time")),
                "segment_end_sec": numeric_or_none(event.get("end_time")),
                "video_source": "l2_event_clip" if using_event_clip else "source_video",
            }
        )

        records.append(
            make_record(
                video_path=video_path,
                prompt=dense_caption_prompt_for_event(event, using_event_clip),
                answer=answer,
                metadata=metadata,
            )
        )

    return records


def build_dense_video_caption_record(
    annotation: dict[str, Any],
    config: BuildConfig,
) -> dict[str, Any] | None:
    video_path = get_source_video_path(annotation)
    if not video_path or not video_exists_if_required(video_path, config):
        return None

    duration = numeric_or_none(annotation.get("clip_duration_sec"))
    lines: list[str] = []
    event_ids: list[Any] = []
    text_sources: set[str] = set()

    for event in sorted(iter_l2_events(annotation), key=lambda item: numeric_or_none(item.get("start_time")) or 0.0):
        start = numeric_or_none(event.get("start_time"))
        end = numeric_or_none(event.get("end_time"))
        if start is None or end is None:
            continue
        if duration is not None:
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
        if start >= end:
            continue

        description, source = event_short_description(event)
        if not description:
            continue

        lines.append(
            f"[{format_seconds(start)}-{format_seconds(end)}] {description}"
        )
        event_ids.append(event.get("event_id") or event.get("id"))
        text_sources.add(source)

    if not lines:
        return None

    if text_sources == {"instruction"}:
        event_text_source = "instruction"
    elif text_sources == {"dense_caption"}:
        event_text_source = "dense_caption"
    else:
        event_text_source = "instruction_fallback_dense_caption"

    metadata = base_metadata(annotation, config, "dense_video_caption")
    metadata.update(
        {
            "num_events": len(lines),
            "event_ids": event_ids,
            "timestamp_format": "seconds",
            "caption_style": "timestamped_l2_short_events",
            "event_text_source": event_text_source,
            "video_source": "source_video",
        }
    )

    return make_record(
        video_path=video_path,
        prompt=DENSE_VIDEO_CAPTION_PROMPT,
        answer="\n".join(lines),
        metadata=metadata,
    )


def build_records_for_annotation(
    annotation: dict[str, Any],
    config: BuildConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if "video_description" in config.tasks:
        record = build_video_description_record(annotation, config)
        if record is not None:
            records.append(record)
    if "dense_caption" in config.tasks:
        records.extend(build_dense_caption_records(annotation, config))
    if "dense_video_caption" in config.tasks:
        record = build_dense_video_caption_record(annotation, config)
        if record is not None:
            records.append(record)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VideoProxy hierarchy annotations to Qwen-VL description SFT JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-dir", required=True, help="Directory containing hierarchy annotation *.json files")
    parser.add_argument("--output-jsonl", required=True, help="Output Qwen-VL SFT JSONL")
    parser.add_argument(
        "--tasks",
        default="video_description",
        help="Comma-separated tasks: video_description,dense_caption,dense_video_caption",
    )
    parser.add_argument("--max-videos", type=int, default=10_000, help="Maximum annotation videos to convert; 0 means unlimited")
    parser.add_argument("--max-records", type=int, default=0, help="Maximum output records; 0 means unlimited")
    parser.add_argument("--max-frames", type=int, default=256, help="Training-side video frame cap recorded in metadata")
    parser.add_argument("--l2-clip-dir", default="", help="Optional atomic L2 clip root; expects L2/{clip_key}_L2_ev{id}_{start}_{end}.mp4")
    parser.add_argument("--require-video-exists", action="store_true", help="Skip records whose selected video path does not exist")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output records before applying max-records")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    annotation_dir = Path(args.annotation_dir)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"annotation-dir not found: {annotation_dir}")

    config = BuildConfig(
        tasks=parse_task_list(args.tasks),
        max_videos=args.max_videos,
        max_records=args.max_records,
        max_frames=args.max_frames,
        l2_clip_dir=args.l2_clip_dir,
        require_video_exists=args.require_video_exists,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    annotation_paths = list(iter_annotation_paths(annotation_dir))
    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(annotation_paths)
    if config.max_videos > 0:
        annotation_paths = annotation_paths[: config.max_videos]

    records: list[dict[str, Any]] = []
    for path in annotation_paths:
        annotation = load_annotation(path)
        records.extend(build_records_for_annotation(annotation, config))

    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(records)

    if config.max_records > 0:
        records = records[: config.max_records]

    output_jsonl = Path(args.output_jsonl)
    write_jsonl(records, output_jsonl)

    task_counts: dict[str, int] = {}
    for record in records:
        task = record.get("metadata", {}).get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    counts = ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
    print(f"Wrote {len(records)} records from {len(annotation_paths)} videos to {output_jsonl}")
    if counts:
        print(f"Task counts: {counts}")
    print(f"Use --video_max_frames {config.max_frames} in qwen-vl-finetune training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
