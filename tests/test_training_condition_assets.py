"""Checkpoint-local tokenizers must work for normal LoRA, not just MeanFlow."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import train
from irodori_tts.config import ModelConfig, TrainConfig


class TrainingConditionAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "model.safetensors"
        self.checkpoint.touch()
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.model = ModelConfig(
            text_encoder_type="pretrained", text_tokenizer_repo="google/t5gemma-2-1b-1b",
            text_encoder_revision="pinned-revision", use_caption_condition=True,
            caption_tokenizer_repo="google/t5gemma-2-1b-1b",
        )
        self.training = TrainConfig(train_mode="rf", lora_enabled=True)
        self.embedded = {"model_type": "t5gemma2"}

    def tearDown(self):
        self.temporary.cleanup()

    def test_regular_lora_uses_local_assets_without_changing_model_identity(self):
        with patch.object(train, "_load_embedded_text_encoder_config", return_value=self.embedded):
            assets, embedded, local = train._resolve_training_condition_assets(
                self.model, self.training, init_checkpoint=self.checkpoint)
        self.assertTrue(local)
        self.assertEqual(assets.text_tokenizer_repo, str(self.tokenizer))
        self.assertEqual(assets.caption_tokenizer_repo, str(self.tokenizer))
        self.assertIsNone(assets.text_encoder_revision)
        self.assertIs(embedded, self.embedded)
        self.assertEqual(self.model.text_tokenizer_repo, "google/t5gemma-2-1b-1b")
        self.assertEqual(self.model.text_encoder_revision, "pinned-revision")

    def test_lora_resume_uses_recorded_base_and_existing_encoder_config(self):
        adapter = self.root / "adapter"
        adapter.mkdir()
        with patch.object(train, "_load_embedded_text_encoder_config") as load:
            assets, embedded, local = train._resolve_training_condition_assets(
                self.model, self.training, resume_checkpoint=adapter,
                resume_base_init={"mode": "checkpoint", "checkpoint_path": str(self.checkpoint)},
                resume_text_encoder_config=self.embedded)
        load.assert_not_called()
        self.assertTrue(local)
        self.assertEqual(assets.text_tokenizer_repo, str(self.tokenizer))
        self.assertIs(embedded, self.embedded)

    def test_full_resume_does_not_reload_checkpoint_weights_to_get_config(self):
        resumed = self.root / "checkpoint.pt"
        resumed.touch()
        with patch.object(train, "_load_embedded_text_encoder_config") as load:
            _, embedded, local = train._resolve_training_condition_assets(
                self.model, self.training, resume_checkpoint=resumed,
                resume_text_encoder_config=self.embedded)
        load.assert_not_called()
        self.assertTrue(local)
        self.assertIs(embedded, self.embedded)

    def test_meanflow_keeps_teacher_asset_priority(self):
        config = TrainConfig(train_mode="meanflow_distill", teacher_checkpoint=str(self.checkpoint))
        with patch.object(train, "_load_embedded_text_encoder_config", return_value=self.embedded) as load:
            _, _, local = train._resolve_training_condition_assets(
                self.model, config, init_checkpoint=self.root / "different.safetensors")
        self.assertTrue(local)
        load.assert_called_once_with(self.checkpoint)

    def test_absent_bundle_preserves_repository_fallback(self):
        (self.tokenizer / "tokenizer_config.json").unlink()
        with patch.object(train, "_load_embedded_text_encoder_config", return_value=self.embedded):
            assets, _, local = train._resolve_training_condition_assets(
                self.model, self.training, init_checkpoint=self.checkpoint)
        self.assertIs(assets, self.model)
        self.assertFalse(local)

    def test_incomplete_bundle_reports_missing_embedded_config(self):
        with patch.object(train, "_load_embedded_text_encoder_config", return_value=None):
            with self.assertRaisesRegex(ValueError, "no embedded text encoder config"):
                train._resolve_training_condition_assets(self.model, self.training, init_checkpoint=self.checkpoint)

    def test_scratch_training_and_legacy_encoder_are_unchanged(self):
        assets, embedded, local = train._resolve_training_condition_assets(self.model, self.training)
        self.assertIs(assets, self.model)
        self.assertIsNone(embedded)
        self.assertFalse(local)
        legacy = ModelConfig()
        with patch.object(train, "_resolve_bundled_tokenizer_source") as resolve:
            assets, embedded, local = train._resolve_training_condition_assets(
                legacy, self.training, init_checkpoint=self.checkpoint)
        resolve.assert_not_called()
        self.assertIs(assets, legacy)
        self.assertFalse(local)


if __name__ == "__main__":
    unittest.main()
