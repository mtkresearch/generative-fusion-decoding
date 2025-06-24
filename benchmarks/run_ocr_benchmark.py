import os
import json
import argparse
from datasets import load_dataset

from gfd.gfd_ocr import BreezperOCR
from gfd.utils import process_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR benchmark dataset")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Dataset identifier on the HuggingFace hub"
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument(
        "--image_column_name", type=str, default="image", help="Column name for images"
    )
    parser.add_argument(
        "--text_column_name", type=str, default="text", help="Column name for reference text"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_files/model/gfd-ocr-en.yaml",
        help="Path to GFD OCR configuration",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="Directory to save results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset_name, split=args.split)
    config = process_config(args.config)
    model = BreezperOCR(config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = []
    for idx, sample in enumerate(ds):
        prediction = model.get_transcription(sample[args.image_column_name])
        result = {
            "id": idx,
            "transcription": sample[args.text_column_name],
            "prediction": prediction,
        }
        results.append(result)
        with open(os.path.join(args.output_dir, f"prediction_{idx}.json"), "w") as f:
            json.dump(result, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
