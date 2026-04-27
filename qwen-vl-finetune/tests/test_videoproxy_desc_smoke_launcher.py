import importlib.util
import sys
import unittest
from pathlib import Path


FINETUNE_ROOT = Path(__file__).resolve().parents[1]
DATA_INIT = FINETUNE_ROOT / "qwenvl" / "data" / "__init__.py"
SCRIPT_PATH = FINETUNE_ROOT / "scripts" / "run_videoproxy_desc_smoke_lora_2gpu.sh"
FORMAL_SCRIPT_PATH = FINETUNE_ROOT / "scripts" / "run_videoproxy_desc_10k_lora_1gpu.sh"


def load_data_module():
    spec = importlib.util.spec_from_file_location("qwenvl_data_config", DATA_INIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VideoProxyDescSmokeLauncherTest(unittest.TestCase):
    def test_smoke_dataset_is_registered_with_absolute_jsonl_and_empty_data_path(self):
        data_module = load_data_module()

        config = data_module.data_list(["videoproxy_desc_smoke"])[0]

        self.assertEqual(
            config["annotation_path"],
            "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/qwen_sft_data/videoproxy_description_smoke.jsonl",
        )
        self.assertEqual(config["data_path"], "")
        self.assertEqual(config["sampling_rate"], 1.0)

    def test_10k_dataset_is_registered_with_absolute_jsonl_and_empty_data_path(self):
        data_module = load_data_module()

        config = data_module.data_list(["videoproxy_desc_10k"])[0]

        self.assertEqual(
            config["annotation_path"],
            "/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/qwen_sft_data/videoproxy_description_10k.jsonl",
        )
        self.assertEqual(config["data_path"], "")
        self.assertEqual(config["sampling_rate"], 1.0)

    def test_two_gpu_smoke_launcher_uses_registered_dataset_and_local_4b_model(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("videoproxy_desc_smoke", text)
        self.assertIn("/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct", text)
        self.assertIn("--nproc_per_node=${NPROC_PER_NODE}", text)
        self.assertIn('--video_max_frames "${VIDEO_MAX_FRAMES}"', text)
        self.assertIn('--video_max_pixels "${VIDEO_MAX_PIXELS}"', text)

    def test_single_gpu_10k_launcher_uses_tensorboard_and_sft_model_root(self):
        text = FORMAL_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("videoproxy_desc_10k", text)
        self.assertIn("NPROC_PER_NODE=\"${NPROC_PER_NODE:-1}\"", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\"", text)
        self.assertIn("/m2v_intern/xuboshen/zgw/SFT-Models/VideoProxyMixed", text)
        self.assertIn("--num_train_epochs \"${NUM_TRAIN_EPOCHS}\"", text)
        self.assertIn("NUM_TRAIN_EPOCHS=\"${NUM_TRAIN_EPOCHS:-2}\"", text)
        self.assertIn("--save_total_limit 2", text)
        self.assertIn("--report_to tensorboard", text)
        self.assertIn("--logging_dir \"${TENSORBOARD_DIR}\"", text)
        self.assertIn("EFFECTIVE_BATCH=$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM_STEPS))", text)


if __name__ == "__main__":
    unittest.main()
