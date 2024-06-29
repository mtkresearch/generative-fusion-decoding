import numpy as np

from gfd.gfd import Breezper
from gfd.utils import process_config

def parse_args():
    parser = argparse.ArgumentParser(description="Override config settings with command-line arguments.")
    parser.add_argument('--config_file_path', type=str, default='config_files/default_config.yaml', help='The default config for running single file')
    parser.add_argument('--model_name', type=str, help='The model for testing the benchmark dataset')
    parser.add_argument('--setting', type=str, help='benchmark dataset settings for specified model')
    parser.add_argument('--audio_file_path', type=str, help='Path to the audio file sample')
    parser.add_argument('--result_output_path', type=str, help='Path to save dataset with predictions from the model')
    return parser.parse_args()

def main():
    args = parse_args()
    config = process_config(args.config_file_path, args)
    model = Breezper(config)
    result = model.get_transcription(args.audio_file_path)
    print(f'Result: {result}')

    with open(args.result_output_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False)

if __name__== '__main__':
    main()    
