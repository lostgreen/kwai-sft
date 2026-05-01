import importlib.util
import sys
import types
import unittest
from pathlib import Path


DATA_PROCESSOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "qwenvl"
    / "data"
    / "data_processor.py"
)


def load_data_processor_module():
    sys.path.insert(0, str(DATA_PROCESSOR_PATH.parents[2]))
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        utils_mod = types.ModuleType("torch.utils")
        data_mod = types.ModuleType("torch.utils.data")
        data_mod.Dataset = object
        torch_mod.Tensor = object
        torch_mod.LongTensor = object
        utils_mod.data = data_mod
        torch_mod.utils = utils_mod
        sys.modules["torch"] = torch_mod
        sys.modules["torch.utils"] = utils_mod
        sys.modules["torch.utils.data"] = data_mod
    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.PreTrainedTokenizer = object
        sys.modules["transformers"] = transformers_mod
    spec = importlib.util.spec_from_file_location("qwenvl.data.data_processor", DATA_PROCESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "qwenvl.data"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DataProcessorFrameListVideoTest(unittest.TestCase):
    def test_make_abs_paths_preserves_nested_frame_list_video(self):
        data_processor = load_data_processor_module()
        base = Path("/dataset")

        result = data_processor._make_abs_paths(base, ["frames/000001.jpg", "/abs/000002.jpg"])

        self.assertEqual(result, ["/dataset/frames/000001.jpg", "/abs/000002.jpg"])

    def test_resolve_video_fps_prefers_experiment_frame_sampling(self):
        data_processor = load_data_processor_module()
        metadata = {
            "video_fps_override": 2.0,
            "experiment_frame_sampling": {
                "videos": [
                    {
                        "duration_sec": 48.38,
                        "target_fps": 1.0,
                        "output_frames": 49,
                    }
                ]
            },
        }

        fps = data_processor.resolve_video_fps_list(metadata, default_fps=2.0, n_videos=1)

        self.assertAlmostEqual(fps[0], 49 / 48.38)

    def test_frame_list_processor_kwargs_disable_sampling_and_set_metadata(self):
        data_processor = load_data_processor_module()
        processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(fps=2.0))
        item = {
            "video": [["/frames/000001.jpg", "/frames/000003.jpg", "/frames/000005.jpg"]],
            "metadata": {
                "experiment_frame_sampling": {
                    "videos": [{"duration_sec": 1.5, "output_frames": 3}]
                }
            },
        }

        kwargs = data_processor._frame_list_processor_kwargs(item, processor)

        self.assertEqual(kwargs["do_sample_frames"], False)
        self.assertEqual(kwargs["video_metadata"][0]["fps"], 2.0)
        self.assertEqual(kwargs["video_metadata"][0]["frames_indices"], [0, 1, 2])
        self.assertEqual(kwargs["video_metadata"][0]["total_num_frames"], 3)

    def test_mp4_processor_kwargs_do_not_disable_sampling(self):
        data_processor = load_data_processor_module()
        processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(fps=2.0))
        item = {"video": "/videos/a.mp4", "metadata": {"video_fps_override": 1.0}}

        kwargs = data_processor._frame_list_processor_kwargs(item, processor)

        self.assertEqual(kwargs, {})

    def test_assistant_label_spans_ignore_plain_assistant_token_in_user_prompt(self):
        data_processor = load_data_processor_module()

        class FakeTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, token):
                return {
                    "<|im_start|>": 151644,
                    "<|im_end|>": 151645,
                }.get(token, self.unk_token_id)

            def encode(self, text, add_special_tokens=False):
                assert add_special_tokens is False
                return {
                    "assistant\n": [77091, 198],
                    "\n": [198],
                }[text]

        ids = [
            151644,
            872,
            198,
            100,
            77091,
            200,
            151645,
            198,
            151644,
            77091,
            198,
            300,
            301,
            151645,
            198,
        ]

        spans = data_processor._assistant_label_spans(ids, FakeTokenizer())

        self.assertEqual(spans, [(11, 15)])


if __name__ == "__main__":
    unittest.main()
