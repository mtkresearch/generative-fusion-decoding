import torch
import numpy as np
import unittest
from unittest.mock import MagicMock
from transformers import AutoModelForCausalLM, Cache, DynamicCache

from gfd.tokenizer import LlamaByteTokenizer

class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.llm = AutoModelForCausalLM.from_pretrained(
                    'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    attn_implementation=None
                )

    def test_kv_cache(self):
        with torch.no_grad():
            ids_for_test = [1, 29537, 886, 275, 90, 154, 4548, 1154 , 4354, 3484 , 4844, 4848, 4]

            # expected logits
            expected_logits = self.llm(
                input_ids=torch.tensor([ids_for_test], device=self.device),
                use_cache=True,
                return_dict=True).logits

            cache = DynamicCache()
            step = 2
            for i in range(step, len(ids_for_test), step):
                model_kwargs = self.llm.prepare_inputs_for_generation(
                    torch.tensor([ids_for_test[:i]], device=self.device),
                    past_key_values=cache,
                    attention_mask=None,
                    inputs_embeds=None,
                    cache_position=None,
                    use_cache=True,
                )

                outputs = self.llm(**model_kwargs)
                cache = outputs.past_key_values
                last_logits = outputs.logits

                expected_last_logits = expected_logits[:,i-step:i,:]

                torch.testing.assert_close(last_logits.cpu().argmax(-1), expected_last_logits.cpu().argmax(-1), atol=1e-6, rtol=0.0)
                torch.testing.assert_close(torch.topk(last_logits.cpu(), 2), torch.topk(expected_last_logits.cpu(), 2), atol=1e-6, rtol=0.0)

        
    

  