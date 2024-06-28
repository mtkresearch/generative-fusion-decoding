from collections import namedtuple

import numpy as np
import unittest
from unittest.mock import MagicMock

from gfd.beam import BeamsControler, DecodingBeam


class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(map(str, ids))

class TestBeamsControler(unittest.TestCase):
    def setUp(self):
        # Create a mock config with necessary attributes
        self.mock_config = MagicMock()
        self.mock_config.beam_max_decode_len = 10
        self.mock_config.beam_min_len = 1
        self.mock_config.beam_max_len = 10
        self.mock_config.beam_max_len_diff = 20
        self.mock_config.beam_terminated_strategy = 'when_all_end'
        self.mock_config.beam_select_strategy = 'best'
        
        # Initialize BeamsControler with the mock config
        self.controller = BeamsControler(config=self.mock_config, n_beam=3, asr_eos_id=1)

    def test_add(self):
        self.controller.add(
            asr_score=0.5, llm_score=0.5, fuse_score=1.0,
            asr_prefix_ids=[1, 2], asr_ids=[3, 4], asr_logprob=-10,
            llm_prefix_ids=[5, 6], llm_ids=[7, 8], llm_logprob=-20
        )

        beam = self.controller._next_beams[0]
        assert len(self.controller._next_beams) == 1
        assert beam.asr_score == 0.5
        assert beam.llm_score == 0.5
        assert beam.fuse_score == 1.0
        assert beam.reach_end == False

    def test_add_beam(self):
        beam = DecodingBeam(
            asr_score=0.5, llm_score=0.5, fuse_score=1.0, reach_end=False,
            asr_ids=[3, 4], llm_ids=[7, 8], asr_prefix_ids=[1, 2], llm_prefix_ids=[5, 6],
            asr_logprob=-10, llm_logprob=-20
        )

        self.controller.add_beam(beam)

        assert len(self.controller._next_beams) == 1
        assert self.controller._next_beams[0] == beam

    def test_update(self):
        beam1 = DecodingBeam(asr_score=0.6, llm_score=0.4, fuse_score=0.5, reach_end=False,
                             asr_ids=[1, 2], llm_ids=[3, 4], asr_prefix_ids=[], llm_prefix_ids=[],
                             asr_logprob=-10, llm_logprob=-20)
        beam2 = DecodingBeam(asr_score=0.7, llm_score=0.5, fuse_score=0.6, reach_end=False,
                             asr_ids=[1, 2], llm_ids=[3, 4], asr_prefix_ids=[], llm_prefix_ids=[],
                             asr_logprob=-10, llm_logprob=-20)
        self.controller._next_beams = [beam1, beam2]

        self.controller.update()

        assert len(self.controller.beams) == 2
        assert self.controller.beams[0] == beam2
        assert self.controller.beams[1] == beam1

    def test_is_terminated_is_True_when_all_beams_reach_end(self):
        beam1 = DecodingBeam(0.5, 0.5, 1.0, True, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        beam2 = DecodingBeam(0.6, 0.4, 1.0, True, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [], [], -10, -10)
        beam3 = DecodingBeam(0.7, 0.3, 1.0, True, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [], [], -10, -10)
        self.controller.add_beam(beam1)
        self.controller.add_beam(beam2)
        self.controller.add_beam(beam3)

        self.controller.update()

        assert self.controller.is_terminated() == True

    def test_is_terminated_is_True_exceed_length_diff(self):
        beam1 = DecodingBeam(0.5, 0.5, 1.0, False, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        beam2 = DecodingBeam(0.6, 0.4, 1.0, False, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [], [], -10, -10)
        beam3 = DecodingBeam(0.7, 0.3, 1.0, False, [i for i in range(1, 30)], [i for i in range(1, 30)], [], [], -10, -10)
        self.controller.add_beam(beam1)
        self.controller.add_beam(beam2)
        self.controller.add_beam(beam3)

        self.controller.update()

        assert self.controller.is_terminated() == True

    def test_is_terminated_is_False(self):
        beam1 = DecodingBeam(0.5, 0.5, 1.0, False, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        beam2 = DecodingBeam(0.6, 0.4, 1.0, False, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [], [], -10, -10)
        beam3 = DecodingBeam(0.7, 0.3, 1.0, False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [], [], -10, -10)
        self.controller.add_beam(beam1)
        self.controller.add_beam(beam2)
        self.controller.add_beam(beam3)

        self.controller.update()
        
        assert self.controller.is_terminated() == False

    def test_get_result_select_strategy_best(self):
        self.mock_config.beam_select_strategy = 'best'
        beam1 = DecodingBeam(0.5, 0.5, 0.7, True, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        beam2 = DecodingBeam(0.6, 0.4, 0.9, True, [4, 5, 6], [4, 5, 6], [], [], -10, -10)
        beam3 = DecodingBeam(0.7, 0.3, 0.8, True, [7, 8, 9], [7, 8, 9], [], [], -10, -10)
        self.controller.add_beam(beam1)
        self.controller.add_beam(beam2)
        self.controller.add_beam(beam3)
        self.controller.update()
        
        mock_asr_tokenizer = MockTokenizer()
        result = self.controller.get_result(mock_asr_tokenizer)

        assert result == '456'

    def test_get_result_select_strategy_longest(self):
        self.mock_config.beam_select_strategy = 'longest'
        beam1 = DecodingBeam(0.5, 0.5, 1.0, True, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        beam2 = DecodingBeam(0.6, 0.4, 1.0, True, [4, 5, 6, 7, 8], [4, 5, 6, 7, 8], [], [], -10, -10)
        beam3 = DecodingBeam(0.7, 0.3, 1.0, True, [7, 8, 9], [7, 8, 9], [], [], -10, -10)

        self.controller.add_beam(beam1)
        self.controller.add_beam(beam2)
        self.controller.add_beam(beam3)
        self.controller.update()
        
        mock_asr_tokenizer = MockTokenizer()
        result = self.controller.get_result(mock_asr_tokenizer)

        assert result == '45678'

    def test_get_result_select_strategy_unsupported(self):
        self.mock_config.beam_select_strategy = 'unsupported'
        beam1 = DecodingBeam(0.5, 0.5, 1.0, True, [1, 2, 3], [1, 2, 3], [], [], -10, -10)
        self.controller.add_beam(beam1)
        self.controller.update()
        mock_asr_tokenizer = MockTokenizer()

        with self.assertRaises(NotImplementedError):
            self.controller.get_result(mock_asr_tokenizer)