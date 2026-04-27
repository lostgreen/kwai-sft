import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_videoproxy_description_sft.py"
)


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_videoproxy_description_sft",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildVideoProxyDescriptionSftTest(unittest.TestCase):
    def test_builds_whole_video_description_record_with_annotation_prompt_style(self):
        builder = load_builder_module()
        annotation = {
            "clip_key": "clip_0001",
            "source_video_path": "/data/videos/clip_0001.mp4",
            "clip_duration_sec": 42.5,
            "domain_l1": "cooking",
            "domain_l2": "food preparation",
            "topology_type": "procedural",
            "video_caption": (
                "A person stands at a kitchen counter and arranges several bowls "
                "and ingredients. The person mixes items in order while the camera "
                "stays focused on the workspace."
            ),
            "level2": {"events": []},
        }

        config = builder.BuildConfig(tasks=("video_description",), max_frames=256)
        records = builder.build_records_for_annotation(annotation, config)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["video"], "/data/videos/clip_0001.mp4")
        self.assertEqual(record["conversations"][0]["from"], "human")
        self.assertTrue(record["conversations"][0]["value"].startswith("<video>\n"))
        self.assertIn(
            "Write a detailed description of the entire video (3-5 sentences).",
            record["conversations"][0]["value"],
        )
        self.assertIn(
            "Every statement must be grounded in what is visible in the frames.",
            record["conversations"][0]["value"],
        )
        self.assertEqual(record["conversations"][1]["from"], "gpt")
        self.assertEqual(record["conversations"][1]["value"], annotation["video_caption"])
        self.assertEqual(record["metadata"]["task"], "video_description")
        self.assertEqual(record["metadata"]["max_frames"], 256)

    def test_builds_l2_dense_caption_records_with_event_clip_metadata(self):
        builder = load_builder_module()
        annotation = {
            "clip_key": "clip_0002",
            "source_video_path": "/data/videos/clip_0002.mp4",
            "clip_duration_sec": 60,
            "domain_l1": "assembly",
            "domain_l2": "tool use",
            "topology_type": "procedural",
            "level2": {
                "events": [
                    {
                        "event_id": 3,
                        "start_time": 10,
                        "end_time": 18,
                        "instruction": "Tighten a screw with a handheld tool",
                        "dense_caption": (
                            "A person positions a small tool over the screw head "
                            "and turns it while holding the part steady. The tool "
                            "stays aligned with the fastener as the component is secured."
                        ),
                    },
                    {
                        "event_id": 4,
                        "start_time": 20,
                        "end_time": 23,
                        "instruction": "Empty caption should be skipped",
                        "dense_caption": "",
                    },
                ]
            },
        }
        config = builder.BuildConfig(
            tasks=("dense_caption",),
            max_frames=256,
            l2_clip_dir="/clips",
        )

        records = builder.build_records_for_annotation(annotation, config)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["video"], "/clips/L2/clip_0002_L2_ev3_10_18.mp4")
        self.assertIn(
            "Write a dense visual caption for this continuous video event (2-4 sentences).",
            record["conversations"][0]["value"],
        )
        self.assertEqual(record["conversations"][1]["value"], annotation["level2"]["events"][0]["dense_caption"])
        self.assertEqual(record["metadata"]["task"], "dense_caption")
        self.assertEqual(record["metadata"]["event_id"], 3)
        self.assertEqual(record["metadata"]["segment_start_sec"], 10)
        self.assertEqual(record["metadata"]["segment_end_sec"], 18)

    def test_cli_writes_jsonl_and_honors_max_videos(self):
        builder = load_builder_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            for idx in range(2):
                (ann_dir / f"clip_{idx}.json").write_text(
                    json.dumps(
                        {
                            "clip_key": f"clip_{idx}",
                            "source_video_path": f"/videos/clip_{idx}.mp4",
                            "clip_duration_sec": 12,
                            "video_caption": f"Caption for clip {idx}.",
                            "level2": {"events": []},
                        }
                    ),
                    encoding="utf-8",
                )
            out_path = root / "out.jsonl"

            with redirect_stdout(io.StringIO()):
                exit_code = builder.main(
                    [
                        "--annotation-dir",
                        str(ann_dir),
                        "--output-jsonl",
                        str(out_path),
                        "--tasks",
                        "video_description",
                        "--max-videos",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["metadata"]["clip_key"], "clip_0")


if __name__ == "__main__":
    unittest.main()
