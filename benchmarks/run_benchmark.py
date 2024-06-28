import os
import json
import time
import argparse
import subprocess
from copy import deepcopy

from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import pipeline

from gfd.gfd import Breezper
from gfd.utils import process_config, combine_config

class Evaluator:
    def __init__(self, model_name, config, transcription_column_name, temp_output_dir):
        self.config = config
        self.transcription_column_name = transcription_column_name
        self.temp_output_dir = temp_output_dir
        self.model_name = model_name
        
        if self.model_name == 'gfd':
            self.model = Breezper(config)
        if self.model_name == 'whisper':
            self.model = pipeline(task='automatic-speech-recognition',
                model=self.config.asr_model_path,
                device=self.config.asr_device
                )

    def evaluate(self, example, idx):
        if self.model_name == 'gfd':
            return self.get_breezper_prediction(example, idx)
        elif self.model_name == 'whisper':
            return self.get_whisper_beam_prediction(example, idx)
        else:
            raise ValueError("Unsupported model type. Please choose 'gfd' or 'whisper'.")

    def get_breezper_prediction(self, example, idx):
        if os.path.exists(os.path.join(self.temp_output_dir, f'prediction_{idx}.json')):
            with open(os.path.join(self.temp_output_dir, f'prediction_{idx}.json'), 'r') as f:
                js = json.load(f)
            example['prediction'] = js["prediction"]
            return example

        breezper_transcription = self.model.get_transcription(example['audio']['array'], asr_prompt=self.config.asr_prompt, 
                                                     llm_prompt=self.config.llm_prompt)
        example['prediction'] = breezper_transcription
        result = {
            'id': idx,
            'transcription': example[self.transcription_column_name],
            'prediction': breezper_transcription 
        }

        with open(os.path.join(self.temp_output_dir, f'prediction_{idx}.json'), 'w') as f:
            json.dump(result, f, ensure_ascii=False)

        return example

    def get_whisper_beam_prediction(self, example, idx):
        whisper_transcription = self.model(example['audio'], generate_kwargs={'task': 'transcribe', 
            'num_beams': 5, 'language':f'<|{self.config.lang}|>'})['text']
        example['prediction'] = whisper_transcription
        result = {
            'id': idx,
            'transcription': example[self.transcription_column_name],
            'prediction': whisper_transcription 
        }

        with open(os.path.join(self.temp_output_dir, f'prediction_{idx}.json'), 'w') as f:
            json.dump(result, f, ensure_ascii=False)

        return example


def test_benchmark(ds, model_name, config, transcription_column_name, output_dir):
    temp_result_dir = f'{output_dir}/temp_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(temp_result_dir)
    evaluator = Evaluator(model_name, config, transcription_column_name, temp_result_dir)
    ds = ds.map(evaluator.evaluate, with_indices=True)
    ds.save_to_disk(os.path.join(output_dir, 'ds_result'))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Benchmark datasets.")
    parser.add_argument('--dataset_name', type=str, help='The benchmark dataset for testing')
    parser.add_argument('--model_name', type=str, help='The model for testing the benchmark dataset')
    parser.add_argument('--setting', type=str, help='benchmark dataset settings for specified model')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save results of the model output')

    return parser.parse_args()

def main():
    setting_configs = {
        'gfd': {'asr-en': process_config('config_files/model/gfd-asr-en.yaml'),
            'asr-zhtw': process_config('config_files/model/gfd-asr-zhtw.yaml'),
            'asr-en-lmoff': process_config('config_files/model/gfd-asr-en.yaml', args=argparse.Namespace(**{'fusing_r': 0.0})),
            'asr-zhtw-lmoff': process_config('config_files/model/gfd-asr-zhtw.yaml', args=argparse.Namespace(**{'fusing_r': 0.0}))
            },
        'whisper': { 'whisper-en': process_config('config_files/model/whisper-en.yaml'),
            'whisper-zhtw': process_config('config_files/model/whisper-zhtw.yaml')
        } 
    }

    args = parse_args()

    setting_config = setting_configs[args.model_name][args.setting]
    if args.dataset_name == 'ml-lecture-2021-long':
        prompt_config = process_config('config_files/prompt/ml-lecture-long-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        ds = load_dataset('generative-fusion-decoding/ml-lecture-2021-long', split='test') # Use Internet Download
        # ds = load_from_disk('../benchmark_dataset/ml_lecture_long', keep_in_memory = True)
        transcription_column_name = 'transcription'
    elif args.dataset_name == 'formosa-long':
        prompt_config = process_config('config_files/prompt/formosa-long-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        ds = load_dataset('Mediatek-Research/formosaspeech', split='test') # Use Internet Download
        # ds = load_from_disk('../../../data/speech/formosaspeech/test', keep_in_memory = True)
        transcription_column_name = 'text'
    elif args.dataset_name == 'fleurs-hk':
        prompt_config = process_config('config_files/prompt/fleurs-hk-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        ds = load_dataset('google/fleurs', 'yue_hant_hk', split='test') # Use Internet Download
        # ds = load_from_disk('../../../data/speech/fleurs/yue_hant_hk/test', keep_in_memory = True)
        transcription_column_name = 'transcription'
    elif args.dataset_name.startswith('noisy-librispeech'):
        signal_to_noise_ratio = args.dataset_name.split('-')[-1]
        prompt_config = process_config('config_files/prompt/noisy-librispeech-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise")[signal_to_noise_ratio] # Use Internet Download # DBG: to add
        transcription_column_name = 'text'
    elif args.dataset_name == 'atco2':
        prompt_config = process_config('config_files/prompt/atco2-asr-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        # ds = load_dataset(from somewhere) # Use Internet Download # DBG: to add
        ds = load_from_disk('../../../data/speech/atco2-asr/train/train', keep_in_memory = True)
        transcription_column_name = 'text'
    elif args.dataset_name == 'atco2-asr-only':
        prompt_config = process_config('config_files/prompt/atco2-asr-only-prompt.yaml')
        combined_config = combine_config(prompt_config, setting_config)
        # ds = load_dataset(from somewhere) # Use Internet Download # DBG: to add
        ds = load_from_disk('../../../data/speech/atco2-asr/train/train', keep_in_memory = True)
        transcription_column_name = 'text'

    if args.model_name == 'gfd':
        test_benchmark(ds, args.model_name, combined_config, transcription_column_name, args.output_dir)
    elif args.model_name == 'whisper':
        test_benchmark(ds, args.model_name, setting_config, transcription_column_name, args.output_dir)

if __name__== '__main__':
    main()
