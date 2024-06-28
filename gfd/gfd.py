import os
import time

import torch
import unittest
import numpy as np
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

from gfd.beam import BeamsControler
from gfd.model import BreezeByte
from gfd.tokenizer import LlamaByteTokenizer, WhisperByteTokenizer

DEBUG = 1

class SuppressTokenWarper():
    def __init__(self, surpress_tokens, min_value):
        self.surpress_tokens = surpress_tokens
        self.min_value = min_value
    
    def __call__(self, scores):
        scores[...,self.surpress_tokens] = self.min_value
        return scores

class Breezper:
    def __init__(self, config):
        self.config = config
        self.asr = WhisperForConditionalGeneration.from_pretrained(
            self.config.asr_model_path, torch_dtype=torch.float16, device_map=self.config.asr_device, 
            attn_implementation=self.config.asr_attn_implementation)
        self.device = self.asr.device
        self.asr_processor = WhisperProcessor.from_pretrained(self.config.asr_model_path)
        self.asr_tokenizer = WhisperByteTokenizer.from_pretrained(self.config.asr_model_path)

        self.breeze_byte = BreezeByte(config)
        self.llm_tokenizer = LlamaByteTokenizer.from_pretrained(self.config.llm_model_path)

        self.asr_prefix_prompt_template = "<|startofprev|> {prompt} "
        if self.config.lang == 'en':
            self.asr_prefix_template = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            self.asr_default_prompt = None
        elif self.config.lang == 'zh':
            self.asr_prefix_template = "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
            self.asr_default_prompt = '繁體中文'

        self.llm_prefix_template = "<s>{prompt}"

        asr_config = GenerationConfig().from_pretrained(self.config.asr_model_path)
        self.suppress_tokens = asr_config.suppress_tokens
        self.begin_suppress_tokens = asr_config.begin_suppress_tokens
        self.surpress_token_func = SuppressTokenWarper(self.suppress_tokens, min_value=float("-inf"))
        self.surpress_begin_token_func = SuppressTokenWarper(self.begin_suppress_tokens + self.suppress_tokens, min_value=float("-inf"))

    def _chunk_audio(self, y, sr):
        chunk_size = self.config.chunk_sec * sr
        stride_size = self.config.stride_sec * sr
        length = len(y)
        count = 1 + max(0, int(np.ceil((length - chunk_size) / (chunk_size - stride_size))))
        for i in range(count):
            start = i * (chunk_size - stride_size)
            end = min(start + chunk_size, length)
            chunked_y = y[start:end]
            yield chunked_y

    def _chunk_audio_by_whipser(self, y, sr, seg_with_overlap):
        input_features = self.asr_processor(y, sampling_rate=sr, 
            return_tensors="pt", truncation=False).input_features.half().to(self.device)
        res = self.asr.generate(prompt_condition_type='first-segment', input_features=input_features, num_beams=5,
                                return_segments=True, return_dict_in_generate=True)

        intervals = []
        # Return segments aggregated by target duration 
        if seg_with_overlap == True: 
            target_duration = 30
            sub_intervals = [(chunk['start'].item(), chunk['end'].item()) for chunk in res['segments'][0]]
            curr_start, curr_end = sub_intervals[0][0], sub_intervals[0][1]
            last_start = None
            for next_start, next_end in sub_intervals[1:]:
                if curr_end - curr_start + (next_end - next_start) > target_duration:
                    intervals.append((curr_start, curr_end))
                    curr_start = last_start
                else:
                    last_start = next_start
                curr_end = next_end
        elif seg_with_overlap == False:
            last_sequence = None
            for chunk in res['segments'][0]:
                curr_sequence = chunk['result']['sequences']
                if last_sequence is None or not torch.equal(last_sequence, curr_sequence):
                    intervals.append((chunk['start'].item(), chunk['end'].item()))
                    last_sequence = curr_sequence
                else:
                    old_start, old_end = intervals.pop()
                    intervals.append((min(old_start, chunk['start'].item()), max(old_end, chunk['end'].item())))
        else:
            raise NotImplementedError

        for start, end in intervals:
            yield y[int(start*sr): int(end*sr)]
               
    def get_transcription(self, fpath_or_audio, sr=16000, num_beams=5, asr_prompt='', llm_prompt=''):
        if isinstance(fpath_or_audio, str):
            y, sr = librosa.load(fpath_or_audio, sr = sr)
        else:
            y = fpath_or_audio
            sr = sr

        if len(y) <= sr * 30:
            transcription = self._get_transcription(y, sr, num_beams, asr_prompt=asr_prompt, llm_prompt=llm_prompt, use_cache=self.config.use_cache)
        else:
            transcription = ''
            for chunked_y in self._chunk_audio_by_whipser(y, sr, seg_with_overlap=self.config.seg_with_overlap):
                last_transcription = self._get_transcription(chunked_y, sr, num_beams, asr_prompt=asr_prompt, llm_prompt=llm_prompt+transcription[-self.config.transcription_cutoff:], use_cache=self.config.use_cache)
                transcription += last_transcription + ' '
                asr_prompt = last_transcription

                if DEBUG:
                    print('current transcription:', transcription)
                    time.sleep(3)
                    if DEBUG > 2:
                        input("Enter to continue ...")

        return transcription

    def fuse(self, asr_score, llm_score):
        if self.config.fuse_strategy == 'simple':  
            return (1 - self.config.fusing_r) * asr_score + self.config.fusing_r * llm_score 
        else:
            raise NotImplementedError()

    def _get_prefix_decoding_ids(self, asr_prompt, llm_prompt):
        # asr
        asr_prompt = asr_prompt if asr_prompt else self.asr_default_prompt
        asr_prefix = (
            self.asr_prefix_prompt_template.format(prompt=asr_prompt)
            + self.asr_prefix_template
        )
        asr_prefix_decoding_ids = self.asr_tokenizer(
            asr_prefix,
            add_special_tokens=False
        ).input_ids

        # llm
        llm_prefix_decoding_ids = self.llm_tokenizer.tokenize_from_byte(
            self.llm_prefix_template.format(
                prompt=llm_prompt
            ).encode('utf8')
        )

        return asr_prefix_decoding_ids, llm_prefix_decoding_ids

    def _asr_forward(self, encoder_outputs, decoder_input_ids, k, supress_func=None):
        with torch.no_grad():
            logits = self.asr(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=torch.tensor(decoder_input_ids, device=self.device), 
                return_dict=True
            ).logits
            if supress_func is not None:
                logits = supress_func(logits)
            logprobs = torch.log(torch.softmax(logits, dim=-1))
            next_logprobs, inds = torch.topk(logprobs[0, -1, :], k, dim=-1)

        return next_logprobs, inds

    def _get_transcription(self, y, sr, num_beams, asr_prompt, llm_prompt, use_cache=None):
        input_features = self.asr_processor(y, sampling_rate=sr, 
            return_tensors="pt").input_features.half().to(self.device)
        encoder_outputs = self.asr.get_encoder()(input_features, return_dict=True)

        beams = BeamsControler(
            config=self.config,
            n_beam=num_beams,
            asr_eos_id=self.asr_tokenizer.eos_token_id)
        
        asr_prefix_decoding_ids, llm_prefix_decoding_ids = self._get_prefix_decoding_ids(asr_prompt, llm_prompt)
        print(encoder_outputs,asr_prefix_decoding_ids,self.surpress_begin_token_func)
        next_asr_logprobs, asr_inds = self._asr_forward(
            encoder_outputs,
            asr_prefix_decoding_ids,
            k=1,
            supress_func=self.surpress_begin_token_func
        )
        print("start",self.asr_tokenizer.convert_ids_to_bytes(asr_inds))
        print(next_asr_logprobs)
        print(asr_prompt, llm_prompt)
        for ind, next_asr_logprob in zip(asr_inds, next_asr_logprobs):
            next_id = ind.item()
            next_asr_logprob = next_asr_logprob.item()
            asr_score, llm_score = self._calcualte_asr_llm_score(
                asr_normalized_len=1,
                asr_logprob=next_asr_logprob,
                llm_normalized_len=1,
                llm_logprob=None
            )

            fuse_score = self.fuse(asr_score, llm_score)
            beams.add(
                asr_score=asr_score,
                llm_score=llm_score,
                fuse_score=fuse_score,
                asr_prefix_ids=asr_prefix_decoding_ids,
                asr_ids=[next_id],
                asr_logprob=next_asr_logprob,
                llm_prefix_ids=llm_prefix_decoding_ids,
                llm_ids=[],
                llm_logprob=None,
            )
        self._update_asr_llm_mean_and_std(beams._next_beams)
        beams.update()

        while True:
            for beam in beams.list():
                if beam.reach_end:
                    beams.add_beam(beam)
                else:
                    next_asr_logprobs, asr_inds = self._asr_forward(
                        encoder_outputs,
                        beam.asr_prefix_ids + beam.asr_ids,
                        k=num_beams,
                        supress_func=self.surpress_token_func
                    )

                    asr_inds = [x.item() for x in asr_inds]
                    next_asr_logprobs = [x.item() for x in next_asr_logprobs]

                    # important: check if asr respond "stop" at top
                    # if not, go back normal operation
                    if asr_inds[0] == self.asr_tokenizer.eos_token_id:
                        next_asr_logprobs = next_asr_logprobs[0:1]
                        asr_inds = asr_inds[0:1]

                    # drop "ending at not top"
                    elif self.asr_tokenizer.eos_token_id in asr_inds:
                        p = asr_inds.index(self.asr_tokenizer.eos_token_id)
                        next_asr_logprobs = next_asr_logprobs[:p] + next_asr_logprobs[p+1:]
                        asr_inds = asr_inds[:p] + asr_inds[p+1:]
                    

                    asr_new_content = self.asr_tokenizer.convert_ids_to_bytes(
                        beam.asr_ids, skip_special_tokens=True
                    )
                    new_content = b''.join(asr_new_content)

                    llm_ids = self.llm_tokenizer.tokenize_from_byte(new_content)
                    if use_cache == 'dynamic':
                        llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob_cache_dynamic(
                            prefix_decoding_ids=llm_prefix_decoding_ids,
                            llm_ids=llm_ids,
                            llm_tokenizer=self.llm_tokenizer
                        )
                    elif use_cache == 'static':
                        llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob_cache_static(
                            prefix_decoding_ids=llm_prefix_decoding_ids,
                            llm_ids=llm_ids,
                            llm_tokenizer=self.llm_tokenizer
                        )
                    else:
                        llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob(
                            prefix_decoding_ids=llm_prefix_decoding_ids,
                            llm_ids=llm_ids,
                            llm_tokenizer=self.llm_tokenizer
                        )
                        
                    assert normalizer_adjust_n <= 0
                    
                    for next_id, next_asr_logprob in zip(asr_inds, next_asr_logprobs):
                        asr_logprob = next_asr_logprob + beam.asr_logprob
                        asr_score, llm_score = self._calcualte_asr_llm_score(
                            asr_normalized_len=len(beam.asr_ids)+1,
                            asr_logprob=asr_logprob,
                            llm_normalized_len=len(llm_ids) + normalizer_adjust_n,
                            llm_logprob=llm_logprob
                        )
                        fuse_score = self.fuse(asr_score, llm_score)
                        beams.add(
                            asr_score=asr_score,
                            llm_score=llm_score,
                            fuse_score=fuse_score,
                            asr_prefix_ids=beam.asr_prefix_ids,
                            asr_ids=beam.asr_ids + [next_id],
                            asr_logprob=asr_logprob,
                            llm_prefix_ids=beam.llm_prefix_ids,
                            llm_ids=llm_ids,
                            llm_logprob=llm_logprob,
                        )
            self._update_asr_llm_mean_and_std(beams._next_beams)
            beams.update()
            self.breeze_byte.kv_cache.remove_unused()

            if DEBUG > 1:
                for k, beam in enumerate(beams.list()):
                    print(f'''[{k}] asr_score={beam.asr_score}, llm_score={beam.llm_score},fuse_score={beam.fuse_score},
{self.asr_tokenizer.decode(beam.asr_ids)}''')
                print()
            elif DEBUG > 0:
                beam = beams.list()[0]
                print(f'''[0] asr_score={beam.asr_score}, llm_score={beam.llm_score},fuse_score={beam.fuse_score},
{self.asr_tokenizer.decode(beam.asr_ids)}
''')

            if beams.is_terminated():
                break

        transcription = beams.get_result(self.asr_tokenizer)
        return transcription

    def _update_asr_llm_mean_and_std(self, beams_list):
        if self.config.fuse_strategy == 'normalized' or self.config.fuse_strategy == 'normalized_2':
            asr_scores = np.array([b.asr_score for b in beams_list])
            llm_scores = np.array([b.llm_score for b in beams_list])

            self.asr_mean = self._moving_average(self.asr_mean, np.mean(asr_scores))
            asr_std = self._moving_average(self.asr_std, np.std(asr_scores - np.mean(asr_scores)))
            if np.isnan(asr_std):
                asr_std = self.min_std
            asr_std = np.clip(asr_std, self.min_std, np.inf)
            self.asr_std = asr_std

            self.llm_mean = self._moving_average(self.llm_mean, np.mean(llm_scores))
            llm_std = self._moving_average(self.llm_std, np.std(llm_scores -np.mean(llm_scores)))
            if np.isnan(llm_std):
                llm_std = self.min_std
            llm_std = np.clip(llm_std, self.min_std, np.inf)
            self.llm_std = llm_std

    def _calcualte_asr_llm_score(self, asr_normalized_len, asr_logprob, llm_normalized_len, llm_logprob):
        if not (asr_logprob > self.config.logprob_min):
            asr_logprob = self.config.logprob_min
            
        if llm_logprob is None or  not (llm_logprob > self.config.logprob_min):
            llm_logprob = self.config.logprob_min
        asr_score = asr_logprob / asr_normalized_len if asr_normalized_len > 0 else self.config.logprob_min
        llm_score = llm_logprob / llm_normalized_len if llm_normalized_len > 0 else self.config.logprob_min

        return asr_score, llm_score
