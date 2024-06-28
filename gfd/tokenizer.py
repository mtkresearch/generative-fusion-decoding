from collections import defaultdict

import numpy as np
from transformers import LlamaTokenizerFast, WhisperTokenizer
from transformers.models.llama.tokenization_llama import SPIECE_UNDERLINE


class ByteTokenizer:
    def tokenize_from_byte(self, byte_str):
        str_part = byte_str.decode('utf8', errors='ignore')
        return self(str_part, add_special_tokens=False).input_ids

    def convert_ids_to_bytes(self, ids):
        raise NotImplementedError

    def get_matched_ids_from_prefix(self, byte_prefix):
        if not hasattr(self, '_prefix_to_ids'):
            self._prefix_to_ids = defaultdict(list)
            for i in range(self.vocab_size):
                b = self.convert_ids_to_bytes(i)
                for j in range(1,len(b)):
                    self._prefix_to_ids[b[:j]].append(i)
        
        return self._prefix_to_ids.get(byte_prefix, [])

    def get_alternative_ids(self, seq_ids):
        alternative_ids = [None] * len(seq_ids)
        prefix_from_last = b''
        pointer_from_last = 1
        while pointer_from_last <= len(seq_ids):
            prefix_from_last = self.convert_ids_to_bytes(seq_ids[-pointer_from_last]) + prefix_from_last
            alternative_ids[-pointer_from_last] = self.get_matched_ids_from_prefix(prefix_from_last)
            pointer_from_last += 1

        return alternative_ids


class LlamaByteTokenizer(LlamaTokenizerFast, ByteTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bytetokens_to_ids = {}
        for s,i in self.get_vocab().items():
            b = self._convert_token_to_byte(s)
            if b in self.bytetokens_to_ids:
                if self.bytetokens_to_ids[b] < i:
                    self.bytetokens_to_ids[b] = i
            else:
                self.bytetokens_to_ids[b] = i

    def convert_ids_to_bytes(self, ids):
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=False)
        if isinstance(tokens, str):
            return self._convert_token_to_byte(tokens)
        return [self._convert_token_to_byte(t) for t in tokens]

    def _convert_token_to_byte(self, token):    
        SPIECE_UNDERLINE = "▁"
        if token.startswith(SPIECE_UNDERLINE) and len(token) > 1:
            token = " " + token.lstrip(SPIECE_UNDERLINE)

        if token.startswith("<0x"): # '<0xAB>' -> 'AB' -> b'\xAB'
            bs = bytes.fromhex(f'{token[3:5]}')
        else:
            bs = token.encode("utf8")
        return bs

    def tokenize_from_byte(self, byte_str):
        str_part = byte_str.decode('utf8', errors='ignore')
        encoded_str_part = str_part.encode('utf8')
        
        str_part_tokenized = self(str_part, add_special_tokens=False).input_ids
        leftover_string = byte_str[len(encoded_str_part):]
        for byte_int in leftover_string:
            byte_character = bytes([byte_int])
            str_part_tokenized.append(self.bytetokens_to_ids[byte_character])

        return str_part_tokenized


class WhisperByteTokenizer(WhisperTokenizer, ByteTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_ids_to_bytes(self, ids, skip_special_tokens=True):
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        return [bytes([self.byte_decoder[c] for c in s]) for s in tokens]
