"""Numerical checks for variable-length masked CUDA Attention dispatch."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend

import irodori_tts.attention as attention
from test_empty_captions import collate, forward, samples
from test_frozen_backbone import build_lora

OLD_PRIORITY = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA regression requires a GPU")
class AttentionPriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dispatch_outputs_and_qkv_gradients(self):
        torch.manual_seed(25)
        for sq, sk in ((7, 19), (16, 29)):
            with self.subTest(sq=sq, sk=sk):
                originals = [torch.randn(2, n, 4, 64, device="cuda", dtype=torch.bfloat16)
                             for n in (sq, sk, sk)]
                mask = torch.zeros(2, 1, 1, sk, device="cuda", dtype=torch.bool)
                mask[0, :, :, :5] = True
                mask[0, :, :, 9:12] = True  # independently padded context segments
                mask[1, :, :, :11] = True
                runs = []
                for priority in (OLD_PRIORITY, attention._SDPA_PRIORITY):
                    inputs = [value.detach().clone().requires_grad_() for value in originals]
                    with patch.object(attention, "_SDPA_PRIORITY", priority):
                        output = attention.masked_sdpa(*inputs, mask)
                    if priority[0] == SDPBackend.EFFICIENT_ATTENTION:
                        # masked_sdpa returns a transpose of the SDPA output.
                        backend_grad_fn = output.grad_fn.next_functions[0][0]
                        self.assertIn("EfficientAttention", type(backend_grad_fn).__name__)
                    output.float().square().mean().backward()
                    runs.append((output.detach(), [value.grad for value in inputs]))
                torch.testing.assert_close(runs[1][0], runs[0][0], rtol=0.025, atol=0.005)
                for old, new in zip(runs[0][1], runs[1][1]):
                    self.assertTrue(torch.isfinite(new).all())
                    torch.testing.assert_close(new, old, rtol=0.04, atol=3e-5)

    def test_empty_prefix_is_finite_and_math_fallback_remains_available(self):
        q = torch.zeros(2, 8, 2, 16, device="cuda")
        mask = torch.zeros(2, 8, device="cuda", dtype=torch.bool)
        mask[1, :3] = True
        with patch.object(attention, "fa3_usable", return_value=False):
            output = attention.prefix_key_mask_attention(q, q, q, mask)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        # FP64 is unsupported by the fused backends; math must still work.
        d = torch.randn(1, 3, 1, 8, device="cuda", dtype=torch.float64)
        result = attention.masked_sdpa(d, d, d, torch.ones(1, 1, 1, 3, device="cuda", dtype=torch.bool))
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.isfinite(result).all())

    def test_lora_checkpointing_gradients_updates_and_reload(self):
        torch.manual_seed(26)
        old = build_lora("diffusion_attn_mlp").cuda()
        new = copy.deepcopy(old)
        for model in (old, new):
            model.set_gradient_checkpointing(True)
        optimizers = [torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
                      for model in (old, new)]
        for captions in (["", ""], ["calm", ""], ["calm", "happy"]):
            batch = {key: value.cuda() for key, value in collate(samples(captions)).items()}
            outputs = []
            for model, optimizer, priority in zip((old, new), optimizers, (OLD_PRIORITY, attention._SDPA_PRIORITY)):
                optimizer.zero_grad(set_to_none=True)
                # Keep the priority context through checkpoint recomputation.
                with patch.object(attention, "_SDPA_PRIORITY", priority), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs.append(forward(model, batch))
                    sum(value.float().square().mean() for value in outputs[-1]).backward()
            for a, b in zip(*outputs):
                torch.testing.assert_close(b, a, rtol=0.025, atol=0.005)
            for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    self.assertTrue(torch.isfinite(b.grad).all(), name)
                    torch.testing.assert_close(b.grad, a.grad, rtol=0.06, atol=3e-4, msg=name)
            for optimizer in optimizers:
                optimizer.step()
            for name, a in old.state_dict().items():
                torch.testing.assert_close(new.state_dict()[name], a, rtol=0.005, atol=1e-4, msg=name)
        new.load_state_dict(old.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
