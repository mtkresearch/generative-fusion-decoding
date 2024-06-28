import pytest
from collections import defaultdict

from gfd.tokenizer import LlamaByteTokenizer

@pytest.fixture(scope="class")
def byte_tokenizer():
    return LlamaByteTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")

class TestLlamaByteTokenizer:
    def test_tokenize_from_byte(self, byte_tokenizer):
        byte_str = b'hello world'
        expected_ids = byte_tokenizer.encode('hello world', add_special_tokens=False)
        result = byte_tokenizer.tokenize_from_byte(byte_str)

        assert result == expected_ids

    def test_convert_ids_to_bytes_english_without_special_characters(self, byte_tokenizer):
        input_ids = [6312, 28709, 1526] # hello world
        tokens = byte_tokenizer.convert_ids_to_tokens(input_ids)
        expected_bytes_lists = [b' hell', b'o', b' world']
        result = byte_tokenizer.convert_ids_to_bytes(input_ids)

        assert result == expected_bytes_lists

    def test_convert_ids_to_bytes_english_with_special_characters(self, byte_tokenizer):
        # Original Input String: Hello! I stayed up late last night and I felf like dying...?
        input_ids = [22557, 28808, 315, 10452, 582, 3909, 1432, 2125, 304, 315, 2770, 737, 13074, 1101, 28804]
        tokens = byte_tokenizer.convert_ids_to_tokens(input_ids)
        expected_bytes_lists = [b' Hello', b'!', b' I', b' stayed', b' up', b' late', b' last', b' night', 
                                b' and', b' I', b' felt', b' like', b' dying', b'...', b'?']
        result = byte_tokenizer.convert_ids_to_bytes(input_ids)

        assert result == expected_bytes_lists

        
    def test_convert_ids_to_bytes_chinese_without_special_characters(self, byte_tokenizer):
        # Original Input String: 我每天都好累
        input_ids = [28705, 29242, 29513, 43136, 51557, 31719] 
        tokens = byte_tokenizer.convert_ids_to_tokens(input_ids)
        expected_bytes_lists = [s.encode('utf-8') for s in tokens]
        result = byte_tokenizer.convert_ids_to_bytes(input_ids)

        assert result == expected_bytes_lists
    
    def test_convert_ids_to_bytes_chinese_with_special_characters(self, byte_tokenizer):
        # Original Input String: # '最近梅雨季，一直下雨真的超煩...!!!!別下了，真心拜託。'
        input_ids = [28705, 42529, 31223, 31115, 31740, 28924, 42405, 48282, 42398, 29800, 33781, 1101, 19010, 
                     30798, 46562, 28924, 45930, 47542, 28944]
        tokens = byte_tokenizer.convert_ids_to_tokens(input_ids)
        expected_bytes_lists = [s.encode('utf-8') for s in tokens]
        result = byte_tokenizer.convert_ids_to_bytes(input_ids)

        assert result == expected_bytes_lists

    def test_get_matched_ids_from_prefix_result_matched(self, byte_tokenizer):
        byte_prefix_match = b'\xe8\x9f\x8b'
        matched_ids = byte_tokenizer.get_matched_ids_from_prefix(byte_prefix_match)
        expected_matched_ids = [61871]

        assert matched_ids == expected_matched_ids

    def test_get_matched_ids_from_prefix_result_no_matched(self, byte_tokenizer):
        byte_prefix_no_match = b'\xe3\x96\x87'
        matched_ids = byte_tokenizer.get_matched_ids_from_prefix(byte_prefix_no_match)
        expected_matched_ids = []

        assert matched_ids == expected_matched_ids



