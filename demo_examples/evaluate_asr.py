#from dataset import get_dataset, normal_mapping, v3_mapping
from model import get_whisper
from accelerate import Accelerator
from tqdm import tqdm
import os
import pandas as pd
import torch
from wer import get_wer
import librosa

accelerator = Accelerator(mixed_precision='fp16')
device = accelerator.device
print(device)

num_beams = 25
args = {"num_beams": num_beams, "num_return_sequences": num_beams, "early_stopping": True, "use_cache": True}

def generate(asr, loader):
    asr, loader = accelerator.prepare(asr, loader)
    asr = asr.to(device)
    references = []
    greedies = []
    n_bests = []
    for batch in tqdm(loader):
        input_features = batch["input_features"].to(device).half()
        texts = batch["text"]
        beams = asr_tokenizer.batch_decode(asr.generate(input_features, language='en', **args), skip_special_tokens=True)
        prediction = asr_tokenizer.batch_decode(asr.generate(input_features, language='en'), skip_special_tokens=True)
        n_best = [beams[i:i+num_beams] for i in range(0, len(beams), num_beams)]
        n_bests.extend(n_best)
        greedies.extend(prediction)
        print(n_best)
        references.extend(texts)

    df = {"text": references, "greedy": greedies, "n_bests": n_bests}
    df = pd.DataFrame(df)

    return df


if __name__ == "__main__":
    asr, asr_tokenizer = get_whisper("whisper-large-v3")
    
    #dataset, loader = get_dataset(batch_size=1, mapping=v3_mapping)
    #df = generate(asr, loader)
    #df.to_csv("large_v3_result.csv", index=False)
