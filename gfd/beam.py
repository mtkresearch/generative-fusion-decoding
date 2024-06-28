from collections import namedtuple

import numpy as np


DecodingBeam = namedtuple(
    'DecodingBeam', 
    ['asr_score', 'llm_score', 'fuse_score', 'reach_end',
     'asr_ids', 'llm_ids', 'asr_prefix_ids', 'llm_prefix_ids', 'asr_logprob', 'llm_logprob']
)

class BeamsControler:
    def __init__(self, config, n_beam, asr_eos_id):
        self.config = config
        self.n_beam = n_beam
        self.asr_eos_id = asr_eos_id

        self.beams = []
        self._next_beams = []

    def list(self):
        return self.beams

    def add(self, asr_score, llm_score, fuse_score,
            asr_prefix_ids, asr_ids, asr_logprob,
            llm_prefix_ids, llm_ids, llm_logprob):
       
        reach_end = (asr_ids[-1] == self.asr_eos_id) or \
            len(asr_prefix_ids + asr_ids) > self.config.beam_max_decode_len

        beam = DecodingBeam(
            asr_score=asr_score,
            llm_score=llm_score,
            fuse_score=fuse_score,
            reach_end=reach_end,
            asr_ids=asr_ids,
            llm_ids=llm_ids,
            asr_prefix_ids=asr_prefix_ids,
            llm_prefix_ids=llm_prefix_ids,
            asr_logprob=asr_logprob,
            llm_logprob=llm_logprob
        )
        self._next_beams.append(beam)

    def add_beam(self, beam):
        self._next_beams.append(beam)

    def update(self):        
        self.beams = sorted(self._next_beams, key=lambda beam: beam.fuse_score, reverse=True)[:self.n_beam]
        self._next_beams = []

    def is_terminated(self):
        min_len = self.config.beam_min_len
        max_len = self.config.beam_max_len
        for beam in self.beams:
            min_len = min(min_len, len(beam.asr_ids))
            max_len = max(max_len, len(beam.asr_ids))
        if max_len - min_len > self.config.beam_max_len_diff:
            return True

        if self.config.beam_terminated_strategy == 'when_all_end':
            return all([beam.reach_end for beam in self.beams])
        else:
            raise NotImplementedError()

    def get_result(self, asr_tokenizer):
        if self.config.beam_select_strategy == 'best':
            transcription = asr_tokenizer.decode(self.beams[0].asr_ids, skip_special_tokens=True)
        elif self.config.beam_select_strategy == 'longest':
            max_len = 0
            transcription = ''
            for beam in self.beams:
                tmp = asr_tokenizer.decode(beam.asr_ids, skip_special_tokens=True)
                if len(tmp) > max_len:
                    transcription = tmp
                    max_len = len(tmp)
        else:
            raise NotImplementedError()
        return transcription
