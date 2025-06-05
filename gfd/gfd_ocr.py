import torch
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from gfd.beam import BeamsControler
from gfd.model import BreezeByte
from gfd.tokenizer import LlamaByteTokenizer, ByteTokenizer


class TrOCRByteTokenizer(ByteTokenizer):
    """Byte-level tokenizer wrapper for TrOCR."""

    def __init__(self, pretrained_model_name_or_path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.bytetokens_to_ids = {}
        for token, idx in self.tokenizer.get_vocab().items():
            b = token.encode("utf8")
            if b in self.bytetokens_to_ids:
                if self.bytetokens_to_ids[b] < idx:
                    self.bytetokens_to_ids[b] = idx
            else:
                self.bytetokens_to_ids[b] = idx

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_ids_to_bytes(self, ids, skip_special_tokens=True):
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        if isinstance(tokens, str):
            tokens = [tokens]
        return [t.encode("utf8") for t in tokens]

    def tokenize_from_byte(self, byte_str):
        str_part = byte_str.decode("utf8", errors="ignore")
        encoded_str_part = str_part.encode("utf8")
        token_ids = self.tokenizer(str_part, add_special_tokens=False).input_ids
        leftover = byte_str[len(encoded_str_part):]
        for b in leftover:
            token_ids.append(self.bytetokens_to_ids.get(bytes([b]), 0))
        return token_ids


class BreezperOCR:
    def __init__(self, config):
        self.config = config
        self.recognizer = VisionEncoderDecoderModel.from_pretrained(
            self.config.ocr_model_path,
            torch_dtype=torch.float16,
            device_map=self.config.ocr_device,
        )
        self.device = self.recognizer.device
        self.processor = TrOCRProcessor.from_pretrained(self.config.ocr_model_path)
        self.asr_tokenizer = TrOCRByteTokenizer(self.config.ocr_model_path)

        self.breeze_byte = BreezeByte(config)
        self.llm_tokenizer = LlamaByteTokenizer.from_pretrained(self.config.llm_model_path)

        self.llm_prefix_template = "<s>{prompt}"

    def fuse(self, asr_score, llm_score):
        return (1 - self.config.fusing_r) * asr_score + self.config.fusing_r * llm_score

    def _get_prefix_decoding_ids(self, asr_prompt, llm_prompt):
        asr_prefix_decoding_ids = self.asr_tokenizer(asr_prompt, add_special_tokens=False).input_ids
        llm_prefix_decoding_ids = self.llm_tokenizer.tokenize_from_byte(
            self.llm_prefix_template.format(prompt=llm_prompt).encode("utf8")
        )
        return asr_prefix_decoding_ids, llm_prefix_decoding_ids

    def _asr_forward(self, encoder_outputs, decoder_input_ids, k):
        with torch.no_grad():
            logits = self.recognizer(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=torch.tensor(decoder_input_ids, device=self.device),
                return_dict=True,
            ).logits
            logprobs = torch.log(torch.softmax(logits, dim=-1))
            next_logprobs, inds = torch.topk(logprobs[0, -1, :], k, dim=-1)
        return next_logprobs, inds

    def get_transcription(self, image_or_path, num_beams=5, asr_prompt="", llm_prompt=""):
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        encoder_outputs = self.recognizer.get_encoder()(pixel_values, return_dict=True)

        beams = BeamsControler(
            config=self.config,
            n_beam=num_beams,
            asr_eos_id=self.asr_tokenizer.tokenizer.eos_token_id,
        )

        asr_prefix_decoding_ids, llm_prefix_decoding_ids = self._get_prefix_decoding_ids(asr_prompt, llm_prompt)
        next_asr_logprobs, asr_inds = self._asr_forward(encoder_outputs, asr_prefix_decoding_ids, k=1)

        for ind, next_asr_logprob in zip(asr_inds, next_asr_logprobs):
            next_id = ind.item()
            next_asr_logprob = next_asr_logprob.item()
            asr_score = next_asr_logprob
            llm_score = 0.0
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
                    )
                    asr_inds = [x.item() for x in asr_inds]
                    next_asr_logprobs = [x.item() for x in next_asr_logprobs]

                    asr_new_content = self.asr_tokenizer.convert_ids_to_bytes(
                        beam.asr_ids, skip_special_tokens=True
                    )
                    new_content = b"".join(asr_new_content)

                    llm_ids = self.llm_tokenizer.tokenize_from_byte(new_content)
                    llm_logprob, normalizer_adjust_n = self.breeze_byte.get_logprob(
                        prefix_decoding_ids=llm_prefix_decoding_ids,
                        llm_ids=llm_ids,
                        llm_tokenizer=self.llm_tokenizer,
                    )

                    for next_id, next_asr_logprob in zip(asr_inds, next_asr_logprobs):
                        asr_logprob = next_asr_logprob + beam.asr_logprob
                        asr_score = asr_logprob / (len(beam.asr_ids) + 1)
                        llm_score = (
                            llm_logprob / (len(llm_ids) + normalizer_adjust_n)
                            if llm_logprob is not None
                            else 0.0
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
            beams.update()
            self.breeze_byte.kv_cache.remove_unused()

            if beams.is_terminated():
                break

        transcription = beams.get_result(self.asr_tokenizer.tokenizer)
        return transcription
