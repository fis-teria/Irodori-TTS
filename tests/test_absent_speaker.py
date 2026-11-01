"""Absent references must not affect RF/duration outputs or optimizer updates."""
import copy
import unittest
from unittest.mock import patch

import torch

from test_empty_captions import collate, samples
from test_frozen_backbone import build_lora


def run(model, batch, skip=False):
    return model(
        x_t=batch['latent_patched'], t=torch.full((2,), .5),
        text_input_ids=batch['text_ids'], text_mask=batch['text_mask'],
        ref_latent=batch['ref_latent_patched'], ref_mask=batch['ref_latent_mask_patched'],
        caption_input_ids=batch['caption_ids'], caption_mask=batch['caption_mask'],
        latent_mask=batch['latent_mask_patched'], duration_features=batch['duration_features'],
        duration_has_speaker=batch['has_speaker'], duration_has_caption=batch['has_caption'],
        speaker_condition_dropout=~batch['has_speaker'], duration_backprop_to_condition=True,
        no_reference_audio=skip,
    )


class AbsentSpeakerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_reference_skips_encoder_preserves_updates_and_reload(self):
        for checkpointing in [False, True]:
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(47)
                old=build_lora('diffusion_attn_mlp')
                new=copy.deepcopy(old)
                for m in [old,new]:m.set_gradient_checkpointing(checkpointing)
                opts=[torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-3,weight_decay=.1)
                      for m in [old,new]]
                # Populate optimizer state using real references, then omit them.
                for labels in [(True,True),(False,False),(True,False)]:
                    rows=samples(['',''])
                    for row,label in zip(rows,labels):row['has_speaker']=label
                    batch=collate(rows)
                    outputs=[]
                    for model,opt,optimized in zip([old,new],opts,[False,True]):
                        opt.zero_grad(set_to_none=True)
                        skip=optimized and not any(labels)
                        if skip:
                            with patch.object(model.speaker_encoder,'forward',side_effect=AssertionError('speaker encoder ran')):
                                out=run(model,batch,skip)
                        else:out=run(model,batch,skip)
                        outputs.append(out)
                        sum(t.square().mean() for t in out).backward()
                    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                    for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
                        self.assertEqual(a.grad is None,b.grad is None,name)
                        if a.grad is not None:
                            self.assertTrue(torch.isfinite(b.grad).all(),name)
                            torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=1e-6,msg=name)
                    for opt in opts:opt.step()
                    for name,a in old.state_dict().items():
                        torch.testing.assert_close(a,new.state_dict()[name],rtol=2e-4,atol=1e-6,msg=name)
                new.load_state_dict(old.state_dict(),strict=True)

    def test_trainable_encoder_and_speaker_inversion_are_not_skipped(self):
        model=build_lora('diffusion_attn_mlp')
        batch=collate(samples(['','']))
        model.speaker_encoder.requires_grad_(True)
        with patch.object(model.speaker_encoder,'forward',wraps=model.speaker_encoder.forward) as called:
            run(model,batch,True)
            self.assertEqual(called.call_count,1)
        model.speaker_encoder.requires_grad_(False)
        model.enable_speaker_inversion(num_tokens=2,init_std=.01)
        with patch.object(model.speaker_encoder,'forward',side_effect=AssertionError('speaker encoder ran')):
            out=run(model,batch,True)
            sum(t.square().mean() for t in out).backward()
        self.assertTrue(any(p.grad is not None for p in model.speaker_inversion.parameters()))

    def test_cpu_metadata_survives_prefetch_without_cuda_scalar_read(self):
        import train
        from irodori_tts.config import TrainConfig
        batch=collate(samples(['','']))
        moved=train._move_batch_to_device(batch,torch.device('cpu'))
        self.assertIs(moved['_no_reference_audio'],True)
        self.assertTrue(train._skip_absent_speaker_encoder(moved,TrainConfig()))
        self.assertFalse(train._skip_absent_speaker_encoder(moved,TrainConfig(speaker_inversion_enabled=True)))
        batch['has_speaker'][0]=True
        self.assertFalse(train._batch_has_no_reference_audio(batch))


if __name__=='__main__':unittest.main()
