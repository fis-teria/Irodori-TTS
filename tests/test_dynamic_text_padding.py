"""Padding-only changes must preserve tokens, features and masked model semantics."""

import copy
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from irodori_tts.dataset import TTSCollator
from irodori_tts.tokenizer import PretrainedTextTokenizer
from test_empty_captions import samples
from test_frozen_backbone import build_lora


def tokenizer(add_bos=True):
    backend = Tokenizer(WordLevel(
        {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = Whitespace()
    return PretrainedTextTokenizer(PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", unk_token="[UNK]",
    ), add_bos=add_bos)


def collate(rows, dynamic, *, add_bos=True, limit=8):
    return TTSCollator(
        tokenizer=tokenizer(add_bos), caption_tokenizer=tokenizer(), latent_dim=4,
        latent_patch_size=1, max_text_len=limit, max_caption_len=12,
        dynamic_text_padding=dynamic,
    )(rows)


def forward(model, batch, drop):
    return model(
        x_t=batch["latent_patched"], t=torch.full((2,), 0.5),
        text_input_ids=batch["text_ids"], text_mask=batch["text_mask"],
        ref_latent=batch["ref_latent_patched"], ref_mask=batch["ref_latent_mask_patched"],
        caption_input_ids=batch["caption_ids"], caption_mask=batch["caption_mask"],
        latent_mask=batch["latent_mask_patched"], duration_features=batch["duration_features"],
        duration_has_speaker=batch["has_speaker"], duration_has_caption=batch["has_caption"],
        duration_backprop_to_condition=True, text_condition_dropout=drop,
    )


class DynamicTextPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tokens_masks_truncation_and_features_are_preserved(self):
        for add_bos in (False, True):
            for limit in (1, 8):
                for texts in (("", ""), ("a", "a b c"), ("a " * 20, "b")):
                    with self.subTest(add_bos=add_bos, limit=limit, texts=texts):
                        rows = samples(["a b", ""])
                        for row, text in zip(rows, texts):
                            row["text"] = text
                        old = collate(rows, False, add_bos=add_bos, limit=limit)
                        new = collate(rows, True, add_bos=add_bos, limit=limit)
                        n = new["text_ids"].shape[1]
                        self.assertEqual(n, max(1, int(old["text_mask"].sum(1).max())))
                        self.assertTrue(new["text_ids"].is_contiguous())
                        self.assertFalse(old["text_mask"][:, n:].any())
                        for key in old:
                            expected = old[key][:, :n] if key in ("text_ids", "text_mask") else old[key]
                            torch.testing.assert_close(new[key], expected, rtol=0, atol=0, msg=key)
        # Length is chosen anew each batch, rather than once for the dataset.
        rows = samples(["", ""])
        rows[0]["text"], rows[1]["text"] = "a", "b"
        self.assertEqual(collate(rows, True)["text_ids"].shape[1], 2)
        rows[1]["text"] = "a b c"
        self.assertEqual(collate(rows, True)["text_ids"].shape[1], 4)

    def test_outputs_gradients_and_adamw_updates(self):
        # Disable stochastic dropout for a meaningful numerical equivalence
        # check. Production dropout retains its probability, but shape changes
        # can change random masks; exact training trajectories are not promised.
        for target in ("diffusion_attn_mlp", "pretrained_backbone_attn"):
            for checkpointing in (False, True):
                with self.subTest(target=target, checkpointing=checkpointing):
                    torch.manual_seed(25)
                    old = build_lora(target)
                    new = copy.deepcopy(old)
                    for model in (old, new):
                        model.set_gradient_checkpointing(checkpointing)
                    optimizers = [torch.optim.AdamW(
                        (p for p in model.parameters() if p.requires_grad), lr=1e-3,
                    ) for model in (old, new)]
                    for texts, captions, drop in (
                        (("a", "a b c"), ["", ""], torch.tensor([False, False])),
                        (("", "a b"), ["a", ""], torch.tensor([True, True])),
                        (("a " * 20, "b"), ["a", "b"], torch.tensor([False, True])),
                    ):
                        rows = samples(captions)
                        for row, text in zip(rows, texts):
                            row["text"] = text
                        batches = [collate(rows, False), collate(rows, True)]
                        outputs = []
                        for model, optimizer, batch in zip((old, new), optimizers, batches):
                            optimizer.zero_grad(set_to_none=True)
                            outputs.append(forward(model, batch, drop))
                            sum(x.square().mean() for x in outputs[-1]).backward()
                        for a, b in zip(*outputs):
                            torch.testing.assert_close(b, a, rtol=2e-5, atol=2e-6)
                        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
                            self.assertEqual(a.grad is None, b.grad is None, name)
                            if a.grad is not None:
                                torch.testing.assert_close(b.grad, a.grad, rtol=3e-4, atol=2e-6, msg=name)
                        for optimizer in optimizers:
                            optimizer.step()
                        for name, a in old.state_dict().items():
                            torch.testing.assert_close(new.state_dict()[name], a, rtol=3e-4, atol=2e-6, msg=name)
                    new.load_state_dict(old.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
