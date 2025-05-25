# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-10 --model_name gfd --setting asr-en --output_dir result/
# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-10 --model_name whisper --setting whisper-en --output_dir result/
# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-5 --model_name gfd --setting asr-en --output_dir result/
# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-5 --model_name whisper --setting whisper-en --output_dir result/
python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-0 --model_name gfd --setting asr-en --output_dir result-temp/
python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-0 --model_name whisper --setting whisper-en --output_dir result/
# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-minus5 --model_name gfd --setting asr-en --output_dir result/
# python benchmarks/run_benchmark.py --dataset_name noisy-librispeech-minus5 --model_name whisper --setting whisper-en --output_dir result/
