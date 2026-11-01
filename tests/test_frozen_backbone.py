"""Use real tiny ModernBERT/PEFT models to check frozen-backbone optimization."""

import copy
import types
import unittest

import torch
from transformers import ModernBertConfig

from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.lora import apply_lora
from irodori_tts.model import PretrainedConditionProjector, PretrainedTextBackbone, TextToLatentRFDiT
from test_empty_captions import collate, forward, samples


def backbone_config():
    return ModernBertConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=32, local_attention=16,
        attention_dropout=0, embedding_dropout=0, mlp_dropout=0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
    ).to_dict()


def initialize(module):
    # from_config(load_pretrained_weights=False) intentionally skips HF init.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.2, 0.2)


def legacy_forward(self, input_ids, mask):
    state = self.backbone(input_ids=input_ids, attention_mask=mask, return_dict=True).last_hidden_state
    return state * mask.unsqueeze(-1).to(state.dtype)


def build_lora(target):
    cfg = ModelConfig(
        latent_dim=4, model_dim=32, num_layers=2, num_heads=4,
        text_vocab_size=64, text_dim=16, text_layers=1, text_heads=2,
        text_encoder_type="pretrained", pretrained_projector_type="residual_mlp",
        use_caption_condition=True, caption_dim=16,
        use_speaker_condition=True, speaker_dim=16, speaker_layers=1,
        speaker_heads=2, timestep_embed_dim=16, adaln_rank=4,
        use_duration_predictor=True, duration_hidden_dim=32,
        duration_layers=1, duration_attention_heads=2, duration_dropout=0,
        duration_architecture="token_sum_dual_adarn_zero_no_aux",
    )
    raw = TextToLatentRFDiT(
        cfg, pretrained_backbone_config=backbone_config(), load_pretrained_backbone_weights=False,
    )
    initialize(raw)
    wrapped = apply_lora(raw, TrainConfig(
        lora_enabled=True, lora_target_modules=target, lora_r=4, lora_alpha=8,
        lora_dropout=0, lora_modules_to_save="auto",
    ))
    return wrapped


class FrozenBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_freeze_removes_existing_hooks_and_keeps_projector_gradients(self):
        backbone = PretrainedTextBackbone(
            "unused", config_dict=backbone_config(), load_pretrained_weights=False,
        )
        initialize(backbone)
        backbone.set_gradient_checkpointing(True)
        self.assertTrue(backbone.backbone.is_gradient_checkpointing)
        backbone.requires_grad_(False)
        backbone.set_gradient_checkpointing(True)
        self.assertFalse(backbone.backbone.is_gradient_checkpointing)
        self.assertFalse(getattr(backbone.backbone, "_require_grads_hooks", []))
        ids = torch.tensor([[1, 3, 4, 0]])
        mask = ids.ne(0)
        encoded = backbone(ids, mask)
        self.assertFalse(encoded.requires_grad)
        self.assertIsNone(encoded.grad_fn)
        torch.testing.assert_close(encoded, legacy_forward(backbone, ids, mask), rtol=0, atol=0)
        projector = PretrainedConditionProjector(16, 8, projector_type="linear")
        projector(backbone, ids, mask).square().mean().backward()
        self.assertGreater(projector.projector.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        # A later unfreeze is detected, rather than using a cached frozen flag.
        backbone.requires_grad_(True)
        backbone.set_gradient_checkpointing(True)
        backbone(ids, mask).square().mean().backward()
        self.assertTrue(backbone.backbone.is_gradient_checkpointing)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters()))
        with torch.no_grad():
            self.assertFalse(backbone(ids, mask).requires_grad)

    def test_full_lora_gradients_updates_and_recompute(self):
        for target in ("diffusion_attn_mlp", "pretrained_backbone_attn"):
            with self.subTest(target=target):
                torch.manual_seed(123)
                optimized = build_lora(target)
                legacy = copy.deepcopy(optimized)
                raw_new, raw_old = optimized.get_base_model(), legacy.get_base_model()
                raw_new.set_gradient_checkpointing(True)
                raw_old.set_gradient_checkpointing(True)
                old_backbone, new_backbone = raw_old.pretrained_text_backbone, raw_new.pretrained_text_backbone
                # Reproduce the old frozen encoder graph and checkpoint setup.
                old_backbone.forward = types.MethodType(legacy_forward, old_backbone)
                old_backbone.backbone.gradient_checkpointing_enable()
                frozen = target == "diffusion_attn_mlp"
                self.assertEqual(new_backbone.has_trainable_parameters(), not frozen)
                self.assertEqual(new_backbone.backbone.is_gradient_checkpointing, not frozen)
                self.assertTrue(raw_new.gradient_checkpointing)  # DiT checkpointing stays enabled.
                self.assertEqual(list(optimized.state_dict()), list(legacy.state_dict()))
                calls_new, calls_old = [], []
                h_new = new_backbone.backbone.layers[0].register_forward_pre_hook(lambda *args: calls_new.append(1))
                h_old = old_backbone.backbone.layers[0].register_forward_pre_hook(lambda *args: calls_old.append(1))
                optimizers = [torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=1e-3)
                              for m in (legacy, optimized)]
                for captions in (["", ""], ["calm", ""]):
                    batch = collate(samples(captions))
                    calls_new.clear()
                    calls_old.clear()
                    outputs = []
                    for model, optimizer in zip((legacy, optimized), optimizers):
                        optimizer.zero_grad(set_to_none=True)
                        outputs.append(forward(model, batch))
                        sum(t.square().mean() for t in outputs[-1]).backward()
                    if frozen:
                        expected_forward_calls = 1 if not any(captions) else 2
                        self.assertEqual(len(calls_new), expected_forward_calls)
                        self.assertGreater(len(calls_old), len(calls_new))
                        encoded = new_backbone(batch["text_ids"], batch["text_mask"])
                        self.assertFalse(encoded.requires_grad)
                    else:
                        self.assertEqual(len(calls_old), len(calls_new))
                        self.assertGreater(len(calls_new), 1 if not any(captions) else 2)
                    for old, new in zip(*outputs):
                        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                    has_nonzero_lora_gradient = False
                    for (name, old), (_, new) in zip(legacy.named_parameters(), optimized.named_parameters()):
                        self.assertEqual(old.grad is None, new.grad is None, name)
                        if old.grad is not None:
                            torch.testing.assert_close(new.grad, old.grad, rtol=2e-4, atol=1e-6, msg=name)
                            if "lora_" in name and new.grad.abs().sum() > 0:
                                has_nonzero_lora_gradient = True
                    self.assertTrue(has_nonzero_lora_gradient)
                    for optimizer in optimizers:
                        optimizer.step()
                    for name, old in legacy.state_dict().items():
                        torch.testing.assert_close(optimized.state_dict()[name], old, rtol=2e-4, atol=1e-6, msg=name)
                h_new.remove()
                h_old.remove()
                optimized.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
