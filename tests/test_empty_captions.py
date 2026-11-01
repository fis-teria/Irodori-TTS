"""Regression checks for missing captions, including gradients and AdamW state."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from irodori_tts.config import ModelConfig
from irodori_tts.dataset import LatentTextDataset, TTSCollator
from irodori_tts.model import TextToLatentRFDiT


class TinyTokenizer:
    def __init__(self):
        self.calls = 0

    def batch_encode(self, texts, max_length):
        self.calls += 1
        ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, text in enumerate(texts):
            values = [1] + [2 + ord(c) % 30 for c in text]
            n = min(len(values), max_length)
            ids[i, :n] = torch.tensor(values[:n])
            mask[i, :n] = True
        return ids, mask


class TinyBackbone(torch.nn.Module):
    """Exercise the shared pretrained-encoder path without network/model downloads."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hidden_size = 16
        self.embedding = torch.nn.Embedding(64, 16)

    def forward(self, input_ids, mask):
        return self.embedding(input_ids) * mask[..., None]

    def set_gradient_checkpointing(self, enabled):
        pass


def samples(captions):
    return [
        dict(text="hello", caption=caption, has_caption=bool(caption),
             latent=torch.ones(4, 4) * (i + 1), ref_latent=torch.ones(4, 4),
             has_speaker=False, num_frames=4)
        for i, caption in enumerate(captions)
    ]


def collate(rows, *, skip=True):
    return TTSCollator(
        tokenizer=TinyTokenizer(), caption_tokenizer=TinyTokenizer(),
        latent_dim=4, latent_patch_size=1, max_text_len=8, max_caption_len=12,
        skip_empty_captions=skip,
    )(rows)


def forward(model, batch):
    return model(
        x_t=batch["latent_patched"], t=torch.full((2,), 0.5, device=batch["latent_patched"].device),
        text_input_ids=batch["text_ids"], text_mask=batch["text_mask"],
        ref_latent=batch["ref_latent_patched"], ref_mask=batch["ref_latent_mask_patched"],
        caption_input_ids=batch["caption_ids"], caption_mask=batch["caption_mask"],
        latent_mask=batch["latent_mask_patched"],
        duration_features=batch["duration_features"],
        duration_has_speaker=batch["has_speaker"], duration_has_caption=batch["has_caption"],
        duration_backprop_to_condition=True,
    )


class EmptyCaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_precheck_handles_missing_null_whitespace_and_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            rows = [dict(text="hi", latent_path="unused.pt", num_frames=4)]
            rows += [dict(rows[0], caption=value) for value in (None, "", " \t", [], [None, " "])]
            rows += [dict(rows[0], caption=[" ", " happy "])]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            dataset = LatentTextDataset(path, latent_dim=4, enable_caption_condition=True)
            self.assertEqual(dataset.caption_labeled_count, 1)
            empty_subset = LatentTextDataset(
                path, latent_dim=4, enable_caption_condition=True,
                subset_indices=list(range(6)), manifest_index=dataset.manifest_index,
            )
            self.assertEqual(empty_subset.caption_labeled_count, 0)

    def test_empty_batch_never_tokenizes_captions(self):
        tokenizer = TinyTokenizer()
        collator = TTSCollator(TinyTokenizer(), tokenizer, 4, 1, max_text_len=8)
        batch = collator(samples(["", ""]))
        self.assertEqual(tokenizer.calls, 0)
        self.assertEqual(tuple(batch["caption_ids"].shape), (2, 0))
        self.assertFalse(batch["has_caption"].any())

    def test_present_and_mixed_batches_are_unchanged(self):
        for captions in (["happy", ""], ["happy", "calm"]):
            with self.subTest(captions=captions):
                before = collate(samples(captions), skip=False)
                after = collate(samples(captions))
                for key in before:
                    torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)

    def test_outputs_gradients_and_optimizer_updates_match(self):
        for pretrained in (False, True):
            for checkpointing in (False, True):
                with self.subTest(pretrained=pretrained, checkpointing=checkpointing):
                    torch.manual_seed(42)
                    cfg = ModelConfig(
                        latent_dim=4, model_dim=32, num_layers=2, num_heads=4,
                        text_vocab_size=64, text_dim=16, text_layers=1, text_heads=2,
                        text_encoder_type="pretrained" if pretrained else "scratch",
                        pretrained_projector_type="residual_mlp",
                        use_caption_condition=True, caption_dim=16,
                        use_speaker_condition=True, speaker_dim=16, speaker_layers=1,
                        speaker_heads=2, timestep_embed_dim=16, adaln_rank=4,
                        use_duration_predictor=True, duration_hidden_dim=32,
                        duration_layers=1, duration_attention_heads=2, duration_dropout=0,
                        duration_architecture="token_sum_dual_adarn_zero_no_aux",
                    )
                    with patch("irodori_tts.model.PretrainedTextBackbone", TinyBackbone):
                        legacy = TextToLatentRFDiT(cfg)
                    # Nonzero DiT gates ensure the attention branch affects outputs.
                    with torch.no_grad():
                        for parameter in legacy.parameters():
                            parameter.uniform_(-0.2, 0.2)
                    optimized = copy.deepcopy(legacy)
                    legacy.set_gradient_checkpointing(checkpointing)
                    optimized.set_gradient_checkpointing(checkpointing)
                    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
                                  for m in (legacy, optimized)]
                    # First populate AdamW momentum with a captioned batch; then
                    # compare empty batches, where grad=None would change updates.
                    for captions in (["happy", "calm"], ["", ""]):
                        batches = [collate(samples(captions), skip=False), collate(samples(captions))]
                        outputs = []
                        for model, optimizer, batch in zip((legacy, optimized), optimizers, batches):
                            optimizer.zero_grad(set_to_none=True)
                            outputs.append(forward(model, batch))
                            sum(t.square().mean() for t in outputs[-1]).backward()
                        for old, new in zip(*outputs):
                            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                        for (name, old), (_, new) in zip(legacy.named_parameters(), optimized.named_parameters()):
                            self.assertEqual(old.grad is None, new.grad is None, name)
                            if old.grad is not None:
                                torch.testing.assert_close(new.grad, old.grad, rtol=2e-4, atol=1e-6, msg=name)
                        for optimizer in optimizers:
                            optimizer.step()
                        for name, old in legacy.state_dict().items():
                            torch.testing.assert_close(optimized.state_dict()[name], old, rtol=2e-4, atol=1e-6, msg=name)
                    # The same instance still accepts captions after an empty batch.
                    with torch.no_grad(), patch.object(optimized.caption_encoder, "forward", side_effect=AssertionError("caption encoder ran")):
                        forward(optimized, collate(samples(["", ""])))
                    optimized.load_state_dict(legacy.state_dict(), strict=True)
                    optimized.eval()
                    with torch.no_grad():
                        for old, new in zip(forward(legacy.eval(), collate(samples(["happy", ""]))),
                                            forward(optimized, collate(samples(["happy", ""])) )):
                            torch.testing.assert_close(new, old, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
