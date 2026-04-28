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
    / "build_videoproxy_proxy_mix_sft.py"
)


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_videoproxy_proxy_mix_sft",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildVideoProxyProxyMixSftTest(unittest.TestCase):
    def test_wraps_plain_choice_answer_when_prompt_requires_answer_tag(self):
        builder = load_builder_module()
        row = {
            "uid": "mcq-1",
            "prompt": (
                "<video>\nQuestion: What happens next?\n"
                "A. chop vegetables\nB. stir sauce\n"
                "Provide your answer (a single letter) inside <answer></answer> tags."
            ),
            "answer": "B",
            "videos": ["/data/video.mp4"],
            "problem_type": "llava_mcq",
        }

        record = builder.convert_record(row, source_path="train.jsonl", line_no=1)

        self.assertIsNotNone(record)
        self.assertEqual(record["video"], "/data/video.mp4")
        self.assertEqual(record["conversations"][0]["value"], row["prompt"])
        self.assertEqual(record["conversations"][1]["value"], "<answer>B</answer>")
        self.assertEqual(record["metadata"]["answer_normalization"], "wrapped_answer_tag")

    def test_preserves_events_answer_and_nested_frame_list_video(self):
        builder = load_builder_module()
        frames = ["/frames/000001.jpg", "/frames/000003.jpg"]
        row = {
            "uid": "seg-1",
            "messages": [{"role": "user", "content": "<video>\nReturn <events> timestamps."}],
            "answer": "<events>[[0, 5], [8, 12]]</events>",
            "videos": [frames],
            "problem_type": "temporal_seg_hier_L2",
            "metadata": {
                "domain_l1": "culinary_food",
                "video_fps_override": 2.0,
                "experiment_frame_sampling": {
                    "videos": [{"duration_sec": 12.0, "output_frames": 6}]
                },
            },
        }

        record = builder.convert_record(row, source_path="train.jsonl", line_no=2)

        self.assertIsNotNone(record)
        self.assertEqual(record["video"], [frames])
        self.assertEqual(record["conversations"][1]["value"], row["answer"])
        self.assertEqual(record["metadata"]["answer_normalization"], "preserved")
        self.assertEqual(record["metadata"]["source_problem_type"], "temporal_seg_hier_L2")
        self.assertEqual(record["metadata"]["video_fps_override"], 2.0)
        self.assertEqual(
            record["metadata"]["experiment_frame_sampling"]["videos"][0]["output_frames"],
            6,
        )

    def test_preserves_existing_answer_tag_without_double_wrapping(self):
        builder = load_builder_module()
        row = {
            "prompt": "<video>\nChoose the valid option inside <answer></answer> tags.",
            "answer": "<answer>C</answer>",
            "videos": ["/data/video.mp4"],
            "problem_type": "event_logic_predict_next",
        }

        record = builder.convert_record(row, source_path="train.jsonl", line_no=3)

        self.assertIsNotNone(record)
        self.assertEqual(record["conversations"][1]["value"], "<answer>C</answer>")
        self.assertEqual(record["metadata"]["answer_normalization"], "preserved")

    def test_cli_writes_qwen_sharegpt_jsonl_and_summary(self):
        builder = load_builder_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            in_path = root / "mixed.jsonl"
            out_path = root / "sft.jsonl"
            rows = [
                {
                    "prompt": "<video>\nSort clips. Put digits inside <answer></answer> tags.",
                    "answer": "132",
                    "videos": ["/clips/a.mp4"],
                    "problem_type": "event_logic_sort",
                },
                {
                    "prompt": "<video>\nFind the interval.",
                    "answer": "The event happens in the 3.00 - 9.00 seconds.",
                    "videos": ["/clips/b.mp4"],
                    "problem_type": "temporal_grounding",
                },
            ]
            in_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = builder.main(
                    [
                        "--input-jsonl",
                        str(in_path),
                        "--output-jsonl",
                        str(out_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["conversations"][1]["value"], "<answer>132</answer>")
            self.assertEqual(records[1]["conversations"][1]["value"], rows[1]["answer"])
            self.assertIn("answer_tag_wrapped=1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
