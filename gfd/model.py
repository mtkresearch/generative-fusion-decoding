import os
import time
import json
from collections import Counter

import torch
import numpy as np
from transformers import AutoModelForCausalLM, RepetitionPenaltyLogitsProcessor, DynamicCache

DEBUG = 0
 
class KVCache:
    def __init__(self):
        self._id_to_cache = {}
        self._id_to_logits = {}
        self._query_count = Counter()
        self._newly_added = set()

    def add(self, decode_ids, kv, logits):
        if isinstance(decode_ids, torch.Tensor):
            decode_ids = tuple(decode_ids.squeeze().tolist())

        if decode_ids not in self._id_to_cache:
            self._id_to_cache[decode_ids] = kv
            self._id_to_logits[decode_ids] = logits
            self._query_count[decode_ids] = 0
            self._newly_added.add(decode_ids)

    def query(self, decode_ids):
        # Find the longest match of decoding ids then return, should increment the number that the entry is queried
        if isinstance(decode_ids, torch.Tensor):
            decode_ids = tuple(decode_ids.squeeze().tolist())

        for i in range(len(decode_ids), 0, -1):
            prefix = decode_ids[:i]
            if prefix in self._id_to_cache:
                self._query_count[prefix] += 1
                return self._id_to_cache[prefix], self._id_to_logits[prefix], len(prefix)

        return None, None, 0

    def remove_unused(self):
        # Delete the entry that is not queried nor newly added
        keys_to_remove = [key for key in self._id_to_cache.keys() if self._query_count[key] == 0 and key not in self._newly_added]
        if DEBUG >= 2:
            print(keys_to_remove)
            print(self._query_count)
            print(self._newly_added)
        for key in keys_to_remove:
            del self._id_to_cache[key]
            del self._id_to_logits[key]
            del self._query_count[key]
        # Update min_length after removal
        self._query_count = Counter()
        self._newly_added = set()
        torch.cuda.empty_cache()
        if DEBUG >= 1:
            print(len(self._decode_ids_to_kv))
        if DEBUG >= 2:
            print(self._decode_ids_to_kv.keys())


class ByteModel:
    def _get_logprob(self, standard_ids, alternative_ids, logits, start_token_id=1):
        # standard_seq       [A  B       C       D       E       F]
        # standard_seq_probs [1, B_prob, C_prob, D_prob, E_prob, F_prob]
        # alternative_ids    [                   [D_alts][E_alts][F_alts]]
        # nseq_vocab_probs                  (bs,  (D_lg,   E_lg)        n_vocab)
        # sheilded_token_length x   x     x        x

        shifted_standard_ids = standard_ids[0, 1:]
        shifted_alternative_ids = alternative_ids[1:]
        shifted_logits = logits[0, :-1, :]
        shifted_start_token_id = start_token_id - 1
        all_logprobs = torch.log(torch.softmax(shifted_logits, dim=-1))
        
        standard_logprobs = all_logprobs[
            torch.arange(0, shifted_standard_ids.size(0)), 
            shifted_standard_ids
        ]
        standard_logprobs[:shifted_start_token_id] = 0.

        rolling_logprobs = torch.cumsum(standard_logprobs, dim=0)

        logprob_collect = []
        for i in range(shifted_start_token_id, len(all_logprobs)):
            if len(shifted_alternative_ids[i]) == 0:
                continue
            alternatives = torch.tensor(shifted_alternative_ids[i],
                device=self.device, dtype=torch.long)

            prev_logprob = rolling_logprobs[i-1]
            selected_logprobs = all_logprobs[i, alternatives]
            logprob_collect.append(prev_logprob + torch.logsumexp(selected_logprobs, dim=0))
        logprob_collect.append(rolling_logprobs[-1])
        logprob_collect_stacked = torch.stack(logprob_collect)
        
        # length to normalize depends on the max contributed sequence length
        normalizer_adjust_n = min(torch.argmax(logprob_collect_stacked) - len(logprob_collect_stacked) + 2, 0)
        
        logprob = torch.logsumexp(logprob_collect_stacked, dim=0)
         
        return logprob, normalizer_adjust_n


class BreezeByte(ByteModel):
    def __init__(self, config):
        self.config = config
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_path, 
            device_map=self.config.llm_device,
            torch_dtype=torch.float16,
            attn_implementation=self.config.llm_attn_implementation
        )
        self.device = self.llm.device
        self.kv_cache = KVCache()
        self.static_cache = None

    def _process_prompt_in_batches(self, static_decoding_ids):
        past_key_values = None
        for i in range(0, len(static_decoding_ids), batch_size):
            batch_input_ids = input_ids[i:i]

    def _initialize_static_cache(self, static_decoding_ids):
        if self.static_cache is None:
            model_kwargs = self.llm.prepare_inputs_for_generation(
                        torch.tensor([static_decoding_ids], device=self.device),
                        attention_mask=None,
                        inputs_embeds=None,
                        cache_position=None,
                        use_cache=True,
                    )
            outputs = self.llm(**model_kwargs)
            self.static_cache = outputs.past_key_values

    def get_logprob(self, prefix_decoding_ids, llm_ids, llm_tokenizer):
        standard_ids = prefix_decoding_ids + llm_ids
        alternative_ids = llm_tokenizer.get_alternative_ids(standard_ids)
        
        with torch.no_grad():
            standard_ids = torch.tensor([standard_ids], device=self.device)
            logits = self.llm(
                        input_ids=standard_ids,
                        return_dict=True
                    ).logits
            logits = logits.float() / self.config.llm_temp

        if self.config.repetition_penalty > self.config.repetition_penalty_threshold:
            logits_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config.repetition_penalty)
            for i in range(1, min(logits.size(1), self.config.repetition_penalty_last + 1)):
                logits[:, -i, :] = logits_processor(standard_ids[:, -i - self.config.repetition_penalty_window:-i], logits[:, -i, :])

        logprob, normalizer_adjust_n = self._get_logprob(standard_ids, alternative_ids, logits,
            start_token_id=len(prefix_decoding_ids))

        return logprob.item(), normalizer_adjust_n

    def get_logprob_cache_static(self, prefix_decoding_ids, llm_ids, llm_tokenizer):
        self._initialize_static_cache(prefix_decoding_ids)

        standard_ids = prefix_decoding_ids + llm_ids
        alternative_ids = llm_tokenizer.get_alternative_ids(standard_ids)
        
        with torch.no_grad():
            standard_ids = torch.tensor([standard_ids], device=self.device)
            logits = self.llm(
                        input_ids=standard_ids,
                        past_key_values=self.static_cache,
                        return_dict=True
                    ).logits
            logits = logits.float() / self.config.llm_temp

        if self.config.repetition_penalty > self.config.repetition_penalty_threshold:
            logits_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config.repetition_penalty)
            for i in range(1, min(logits.size(1), self.config.repetition_penalty_last + 1)):
                logits[:, -i, :] = logits_processor(standard_ids[:, -i - self.config.repetition_penalty_window:-i], logits[:, -i, :])

        logprob, normalizer_adjust_n = self._get_logprob(standard_ids, alternative_ids, logits,
            start_token_id=len(prefix_decoding_ids))

        return logprob.item(), normalizer_adjust_n


    def get_logprob_cache_dynamic(self, prefix_decoding_ids, llm_ids, llm_tokenizer):
        standard_ids = prefix_decoding_ids + llm_ids
        alternative_ids = llm_tokenizer.get_alternative_ids(standard_ids)
        
        with torch.no_grad():
            standard_ids = torch.tensor([standard_ids], device=self.device)
            past_key_values, past_logits , matched_length = self.kv_cache.query(standard_ids)

            if matched_length == standard_ids.shape[1]:
                logits = past_logits
            else:
                model_kwargs = self.llm.prepare_inputs_for_generation(
                    standard_ids[:, matched_length:],
                    past_key_values=past_key_values,
                    attention_mask=None,
                    inputs_embeds=None,
                    cache_position=None,
                    use_cache=True,
                )
                outputs = self.llm(**model_kwargs)

                logits = torch.cat((past_logits, outputs.logits), dim=1) if past_logits is not None else outputs.logits
                
                if logits.size(1) != standard_ids.shape[1]: # The output logits size does not match the length of standard_id 
                    model_kwargs = self.llm.prepare_inputs_for_generation(
                        standard_ids,
                        attention_mask=None,
                        inputs_embeds=None,
                        cache_position=None,
                        use_cache=True,
                    )
                    outputs = self.llm(**model_kwargs)
                    logits = outputs.logits

                self.kv_cache.add(standard_ids, outputs.past_key_values, logits)

            logits = logits.float() / self.config.llm_temp


        if self.config.repetition_penalty > self.config.repetition_penalty_threshold:
            logits_processor = RepetitionPenaltyLogitsProcessor(penalty=self.config.repetition_penalty)
            for i in range(1, min(logits.size(1), self.config.repetition_penalty_last + 1)):

                logits[:, -i, :] = logits_processor(standard_ids[:, -i - self.config.repetition_penalty_window:-i], logits[:, -i, :])
        
        logprob, normalizer_adjust_n = self._get_logprob(standard_ids, alternative_ids, logits,
            start_token_id=len(prefix_decoding_ids))
        
        return logprob.item(), normalizer_adjust_n

