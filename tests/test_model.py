import os
import time
import json

import torch
import numpy as np
from transformers import RepetitionPenaltyLogitsProcessor
import unittest
from unittest.mock import patch, MagicMock

from gfd.tokenizer import LlamaByteTokenizer
from gfd.model import BreezeByte

class TestBreezeByte(unittest.TestCase):
    @patch('gfd.model.AutoModelForCausalLM.from_pretrained')
    def test_get_logprob(self, MockModel):
        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_alternative_ids.return_value = [[0], [1], [2], [3], [4]]

        # Mock the model
        mock_model = MockModel.return_value
        mock_model.device = "cuda:0"
        mock_model.return_dict = True

        mock_logits = torch.randn(1, 5, 10).to(mock_model.device)
        mock_model.return_value.logits = mock_logits

        # Mock config
        mock_config = MagicMock()
        mock_config.llm_model_path = 'fakepath/to/model'
        mock_config.llm_device = 'cuda:0'
        mock_config.llm_attn_implementation = 'default'
        mock_config.llm_temp = 1.0
        mock_config.repetition_penalty = 1.2
        mock_config.repetition_penalty_threshold = 1.1
        mock_config.repetition_penalty_last = 2
        mock_config.repetition_penalty_window = 1

        # Initialize BreezeByte with the mock config
        breeze_byte = BreezeByte(mock_config)
        breeze_byte.llm = mock_model

        prefix_decoding_ids = [1, 2]
        llm_ids = [3, 4, 5]

        logprob = breeze_byte.get_logprob(prefix_decoding_ids, llm_ids, mock_tokenizer)

        self.assertIsInstance(logprob, float)
