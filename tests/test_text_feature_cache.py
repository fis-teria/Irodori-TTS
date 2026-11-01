"""Frozen feature reuse must preserve trainable projections and safe fallbacks."""
import copy
import unittest
from unittest.mock import patch

import torch

from irodori_tts.model import PretrainedConditionProjector, PretrainedTextBackbone
from irodori_tts.text_cache import make_text_feature_keys
from test_frozen_backbone import backbone_config, build_lora, initialize
from test_empty_captions import collate, samples


def backbone():
    m=PretrainedTextBackbone('unused',config_dict=backbone_config(),load_pretrained_weights=False)
    initialize(m)
    m.requires_grad_(False)
    return m


class TextFeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reuse_with_variable_padding_preserves_projector_updates(self):
        torch.manual_seed(49)
        old=backbone()
        new=copy.deepcopy(old)
        new.set_feature_cache(2**20)
        projectors=[PretrainedConditionProjector(16,8,projector_type='residual_mlp')]
        projectors.append(copy.deepcopy(projectors[0]))
        opts=[torch.optim.AdamW(p.parameters(),lr=.001) for p in projectors]
        ids=torch.tensor([[1,3,0,0],[1,4,5,2]])
        calls=[]
        handle=new.backbone.register_forward_pre_hook(lambda *args:calls.append(1))
        for step in range(4):
            current=ids if step%2==0 else torch.nn.functional.pad(ids,(0,3))
            mask=current.ne(0)
            keys=make_text_feature_keys(current,mask)
            outputs=[]
            for m,p,opt in zip([old,new],projectors,opts):
                opt.zero_grad(set_to_none=True)
                out=p(m,current,mask,**({'cache_keys':keys} if m is new else {}))
                outputs.append(out)
                out.square().mean().backward()
            torch.testing.assert_close(outputs[0],outputs[1],rtol=2e-5,atol=2e-6)
            for a,b in zip(projectors[0].parameters(),projectors[1].parameters()):
                self.assertEqual(a.grad is None,b.grad is None)
                if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=2e-6)
            self.assertGreater(projectors[1].projector.weight.grad.abs().sum(),0)
            for opt in opts:opt.step()
            for a,b in zip(projectors[0].parameters(),projectors[1].parameters()):
                torch.testing.assert_close(a,b,rtol=3e-4,atol=2e-6)
        self.assertEqual(len(calls),1)
        self.assertTrue(all(p.grad is None for p in new.parameters()))
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        handle.remove()

    def test_keys_distinguish_tokens_masks_and_ignore_only_right_padding(self):
        ids=torch.tensor([[1,3,0],[1,4,0]])
        mask=ids.ne(0)
        keys=make_text_feature_keys(ids,mask)
        padded=make_text_feature_keys(torch.nn.functional.pad(ids,(0,4)),torch.nn.functional.pad(mask,(0,4)))
        self.assertEqual(keys,padded)
        self.assertNotEqual(keys[0],keys[1])
        changed=mask.clone();changed[0,0]=False
        self.assertNotEqual(keys[0],make_text_feature_keys(ids,changed)[0])
        with self.assertRaises(ValueError):
            make_text_feature_keys(ids.to('meta'),mask.to('meta'))

    def test_unfreeze_dropout_and_weight_reload_invalidate_features(self):
        m=backbone();m.set_feature_cache(2**20)
        ids=torch.tensor([[1,3,0]]);mask=ids.ne(0);keys=make_text_feature_keys(ids,mask)
        with patch.object(m.backbone,'forward',wraps=m.backbone.forward) as called:
            m(ids,mask,cache_keys=keys);m(ids,mask,cache_keys=keys)
            self.assertEqual(called.call_count,1)
            m.load_state_dict(m.state_dict(),strict=True)
            m(ids,mask,cache_keys=keys)
            self.assertEqual(called.call_count,2)
            m.backbone.config.attention_dropout=.1
            m(ids,mask,cache_keys=keys);m(ids,mask,cache_keys=keys)
            self.assertEqual(called.call_count,4)
            self.assertEqual(m._frozen_text_cache.stats()['entries'],0)
            m.backbone.config.attention_dropout=0
            m.requires_grad_(True)
            m(ids,mask,cache_keys=keys).square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in m.parameters()))
            self.assertEqual(m._frozen_text_cache.stats()['entries'],0)

    def test_lru_bound_and_mode_dtype_separation(self):
        m=backbone();m.set_feature_cache(128) # two fp32 tokens x 16 hidden = 128 bytes
        ids=torch.tensor([[1,3],[1,4]])
        mask=ids.ne(0);keys=make_text_feature_keys(ids,mask)
        with patch.object(m.backbone,'forward',wraps=m.backbone.forward) as called:
            m(ids,mask,cache_keys=keys)
            self.assertLessEqual(m._frozen_text_cache.stats()['payload_bytes'],128)
            m(ids[:1],mask[:1],cache_keys=keys[:1])
            self.assertEqual(called.call_count,2) # first entry was evicted
        m.set_feature_cache(2**20)
        with patch.object(m.backbone,'forward',wraps=m.backbone.forward) as called:
            m(ids,mask,cache_keys=keys)
            m.eval();m(ids,mask,cache_keys=keys)
            self.assertEqual(called.call_count,2)
            with torch.autocast('cpu',dtype=torch.bfloat16):m(ids,mask,cache_keys=keys)
            self.assertEqual(called.call_count,3)
        m.to(dtype=torch.float64)
        self.assertEqual(m._frozen_text_cache.stats()['entries'],0)

    def test_full_lora_duration_and_condition_dropout_keep_gradients(self):
        for checkpointing in [False,True]:
            torch.manual_seed(53)
            old=build_lora('diffusion_attn_mlp');new=copy.deepcopy(old)
            new.pretrained_text_backbone.set_feature_cache(2**20)
            for m in [old,new]:m.set_gradient_checkpointing(checkpointing)
            opts=[torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.001) for m in [old,new]]
            batch=collate(samples(['','']))
            keys=make_text_feature_keys(batch['text_ids'],batch['text_mask'])
            with patch.object(new.pretrained_text_backbone.backbone,'forward',wraps=new.pretrained_text_backbone.backbone.forward) as called:
                for _ in range(2):
                    outputs=[]
                    for m,opt in zip([old,new],opts):
                        opt.zero_grad(set_to_none=True)
                        result=m(x_t=batch['latent_patched'],t=torch.full((2,),.5),
                            text_input_ids=batch['text_ids'],text_mask=batch['text_mask'],
                            ref_latent=batch['ref_latent_patched'],ref_mask=batch['ref_latent_mask_patched'],
                            caption_input_ids=batch['caption_ids'],caption_mask=batch['caption_mask'],
                            latent_mask=batch['latent_mask_patched'],duration_features=batch['duration_features'],
                            duration_has_speaker=batch['has_speaker'],duration_has_caption=batch['has_caption'],
                            text_condition_dropout=torch.tensor([False,True]),duration_backprop_to_condition=True,
                            text_cache_keys=keys if m is new else None)
                        outputs.append(result)
                        sum(x.square().mean() for x in result).backward()
                    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                    for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
                        self.assertEqual(a.grad is None,b.grad is None,name)
                        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=2e-6,msg=name)
                    for opt in opts:opt.step()
                self.assertEqual(called.call_count,1)
            new.load_state_dict(old.state_dict(),strict=True)

    def test_cpu_metadata_and_compile_or_modified_mask_fallback(self):
        import train
        from irodori_tts.config import ModelConfig, TrainConfig
        batch=collate(samples(['','']))
        moved=train._move_batch_to_device(batch,torch.device('cpu'),cache_text_features=True)
        self.assertEqual(moved['_text_cache_keys'],make_text_feature_keys(batch['text_ids'],batch['text_mask']))
        self.assertEqual(train._batch_text_cache_keys(moved,enabled=True),moved['_text_cache_keys'])
        self.assertIsNone(train._batch_text_cache_keys(batch,enabled=False))
        self.assertFalse(train._training_text_cache_enabled(ModelConfig(use_duration_predictor=False),TrainConfig(text_condition_dropout=.1)))
        self.assertFalse(train._training_text_cache_enabled(ModelConfig(),TrainConfig(compile_model=True)))
        self.assertFalse(train._training_text_cache_enabled(ModelConfig(),TrainConfig(train_mode='meanflow_distill')))


if __name__=='__main__':unittest.main()
