import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "merge_qwen_vl_lora.py"
)


def load_merge_module():
    spec = importlib.util.spec_from_file_location(
        "merge_qwen_vl_lora",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_adapter_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"fake")


class MergeQwenVLLoraTest(unittest.TestCase):
    def test_resolve_adapter_path_accepts_final_adapter_dir(self):
        merger = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_adapter_files(root)

            self.assertEqual(merger.resolve_adapter_path(root), root)

    def test_resolve_adapter_path_picks_latest_checkpoint_with_adapter_files(self):
        merger = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_adapter_files(root / "checkpoint-20")
            write_adapter_files(root / "checkpoint-105")
            (root / "checkpoint-200").mkdir()

            self.assertEqual(
                merger.resolve_adapter_path(root),
                root / "checkpoint-105",
            )

    def test_resolve_adapter_path_requires_peft_adapter_files(self):
        merger = load_merge_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "checkpoint-1").mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "No PEFT LoRA adapter"):
                merger.resolve_adapter_path(root)

    def test_parse_torch_dtype_keeps_auto_as_string(self):
        merger = load_merge_module()

        self.assertEqual(merger.parse_torch_dtype("auto"), "auto")


if __name__ == "__main__":
    unittest.main()
