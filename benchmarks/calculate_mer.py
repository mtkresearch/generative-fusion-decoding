import argparse
import os
import regex
import re

import json
import jiwer
import pandas as pd
import hanzidentifier
from opencc import OpenCC
from whisper.normalizers import EnglishTextNormalizer
from datasets import load_from_disk
import Levenshtein

cc = OpenCC("s2t")
normalizer = EnglishTextNormalizer()
greek_to_phonetic = {
    'α': 'alpha',
    'β': 'beta',
    'γ': 'gamma',
    'δ': 'delta',
    'ε': 'epsilon',
    'ζ': 'zeta',
    'η': 'eta',
    'θ': 'theta',
    'ι': 'iota',
    'κ': 'kappa',
    'λ': 'lambda',
    'μ': 'mu',
    'ν': 'nu',
    'ξ': 'xi',
    'ο': 'omicron',
    'π': 'pi',
    'ρ': 'rho',
    'σ': 'sigma',
    'τ': 'tau',
    'υ': 'upsilon',
    'φ': 'phi',
    'χ': 'chi',
    'ψ': 'psi',
    'ω': 'omega',
    'Α': 'Alpha',
    'Β': 'Beta',
    'Γ': 'Gamma',
    'Δ': 'Delta',
    'Ε': 'Epsilon',
    'Ζ': 'Zeta',
    'Η': 'Eta',
    'Θ': 'Theta',
    'Ι': 'Iota',
    'Κ': 'Kappa',
    'Λ': 'Lambda',
    'Μ': 'Mu',
    'Ν': 'Nu',
    'Ξ': 'Xi',
    'Ο': 'Omicron',
    'Π': 'Pi',
    'Ρ': 'Rho',
    'Σ': 'Sigma',
    'Τ': 'Tau',
    'Υ': 'Upsilon',
    'Φ': 'Phi',
    'Χ': 'Chi',
    'Ψ': 'Psi',
    'Ω': 'Omega',
    # Add lowercase and uppercase variants for all Greek letters
}

def separate_text(x):
    newx = ""
    for i,c in enumerate(x):
        if ord(c) > 1000:
            newx += " " + c + " "
        elif c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and x[i-1] not in "ABCDEFGHIJKLMNOPQRSTYVWXYZ":
            newx += " " + c
        else:
            newx += c
    newx = re.sub(r'\s+',' ',newx)
    newx = newx.strip()
    
    return newx


def demathify(x):
    for k,v in greek_to_phonetic.items():
        x = x.replace(k," "+v+" ")
    x = x.replace("*", " star ")
    x = x.replace("′", " prime ")
    x = x.replace("%", " percent ")

    return x


def clean_repeating_end(text, repeat_threshold):
    x = list(text.split())[::-1]
    for k in range(1,len(x)//repeat_threshold):
        repeat = x[0:k] * repeat_threshold
        motiv = x[0:k]
        
        if repeat == x[0:k * repeat_threshold]:
            print(repeat)
            while x[0:k] == motiv:
                x = x[k:]
            idx = -1
            while len(x) and x[-1] == motiv[idx]:
                x = x[-1:]
                idx -= 1
            x = motiv + x
            ret = " ".join(x[::-1])
            print(ret)
            return ret

    return text


def predscleaner(x):
    x = x.strip()
    if "<|transcribe|>" in x:
        x = x.split("<|transcribe|>")[1]
    x = re.sub('<[^>]*>','', x)
    l = len(x)
    while True:
        x = clean_repeating_end(x, 3)
        if len(x) == l:
            break
        l = len(x)
    x = demathify(x)
    x = re.sub(r'[^\w\s]',' ',x)
    
    x = x.strip()
    if hanzidentifier.is_simplified(x):
        x = cc.convert(x)
    x = x.lower()
    x = separate_text(x)
    x = normalizer(x)

    return x


def goldcleaner_ml(x):
    x = demathify(x)
    x = re.sub(r'[^\w\s]','',x)
    x = x.strip()
    x = x.lower()
    x = normalizer(x)

    return x


def goldcleaner_formosa(x):
    x = demathify(x)
    x = re.sub(r'[^\w\s]','',x)
    x = x.strip()
    if hanzidentifier.is_simplified(x):
        x = cc.convert(x)
    x = x.lower()
    x = normalizer(x)
    x = separate_text(x)

    return x


def goldcleaner_librispeech(x):
    #x = re.sub(r'[^\w\s]','',x)
    x = x.strip()
    x = x.lower()
    x = normalizer(x)

    return x

def predscleaner_librispeech(x):
    x = x.strip()
    x = re.sub('<[^>]*>','', x)
    x = clean_repeating_end(x, 3)
    x = normalizer(x)
    x = x.strip()

    return x

def predscleaner_acto2(x):
    x = x.strip()
    if "<|transcribe|>" in x:
        x = x.split("<|transcribe|>")[1]
    x = re.sub('<[^>]*>','', x)
    l = len(x)
    while True:
        x = clean_repeating_end(x, 5)
        if len(x) == l:
            break
        l = len(x)
    x = re.sub(r'[^\w\s]',' ',x)
    
    x = x.strip()
    x = x.lower()
    x = normalizer(x)

    return x


# def edit_distance_with_whitespace(transcription, prediction):
#     def compute_edit_distance(s1, s2):
#         m = len(s1)
#         n = len(s2)
#         dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
        
#         for i in range(m + 1):
#             for j in range(n + 1):
#                 if i == 0:
#                     dp[i][j] = j
#                 elif j == 0:
#                     dp[i][j] = i
#                 elif s1[i - 1] == s2[j - 1]:
#                     dp[i][j] = dp[i - 1][j - 1]
#                 else:
#                     dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
        
#         return dp[m][n]
    
#     # Function to find English words in a string
#     def find_english_words(s):
#         return [match.span() for match in re.finditer(r'\b[a-zA-Z]+(?:\s+[a-zA-Z]+)*\b', s)]
    
#     # Compute the standard edit distance
#     standard_edit_distance = compute_edit_distance(transcription, prediction)
    
#     # Find English words in both strings
#     transcription_english_words = find_english_words(transcription)
#     prediction_english_words = find_english_words(prediction)
    
#     # Check for whitespace insertion
#     index_to_add_whitespace = []
#     index_to_delete_whitespace = []
    
#     # Check for whitespace insertion in prediction
#     for start, end in prediction_english_words:
#         for i in range(start + 1, end):
#             if prediction[i] != ' ':
#                 new_prediction = prediction[:i] + ' ' + prediction[i:]
#                 if standard_edit_distance > compute_edit_distance(new_prediction, transcription):
#                     index_to_add_whitespace.append(i)

#     # Check for whitespace deletion in prediction
#     for start, end in prediction_english_words:
#         for i in range(start, end - 1):
#             if prediction[i] == ' ':
#                 new_prediction = prediction[:i] + prediction[i+1:]
#                 if standard_edit_distance > compute_edit_distance(new_prediction, transcription):
#                     index_to_delete_whitespace.append(i)

#     # Sort the indexes
#     index_to_add_whitespace.sort()
#     index_to_delete_whitespace.sort()

#     # Apply deletions first
#     new_prediction = list(prediction)
#     offset = 0
#     for i in index_to_delete_whitespace:
#         del new_prediction[i - offset]
#         offset += 1

#     # Apply additions
#     offset = 0
#     for i in index_to_add_whitespace:
#         new_prediction.insert(i + offset, ' ')
#         offset += 1

#     new_prediction = ''.join(new_prediction)
    
#     return new_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Benchmark Result.")
    parser.add_argument('--dataset_name', type=str, help='The benchmark dataset for testing')
    parser.add_argument('--output_dir', type=str, help='The output directory for benchmark dataset')

    return parser.parse_args()

args = parse_args()
if args.dataset_name in 'ml-lecture-2021-long':
    transcription_column_name = 'transcription'
    goldcleaner_function = goldcleaner_ml
elif args.dataset_name == 'formosa-long':
    transcription_column_name = 'transcription'
    goldcleaner_function = goldcleaner_formosa
elif args.dataset_name == 'fleurs-hk':
    transcription_column_name = 'transcription'
    goldcleaner_function = goldcleaner_formosa
elif args.dataset_name.startswith('noisy-librispeech'):
    transcription_column_name = 'text'
    goldcleaner_function = goldcleaner_librispeech
    predscleaner = predscleaner_librispeech
elif args.dataset_name == 'acto2':
    transcription_column_name = 'text'
    predscleaner = predscleaner_acto2
    goldcleaner_function = goldcleaner_librispeech

# Try get the dataset with predictions, else use json files to evaluate
use_dataset = True
try:
    ds = load_from_disk(os.path.join(args.output_dir, 'ds_result'))
except:
    print(args.output_dir, 'Load from disk failed.')
    use_dataset = False

if use_dataset:
    try:
        sub_ds = ds.select_columns([transcription_column_name, 'prediction'])
        sub_ds = sub_ds.rename_column(transcription_column_name, 'transcription')
        df = sub_ds.to_pandas()
    except:
        print('Extracting dataset failed. Column name error.')
        use_dataset = False

if not use_dataset: # use json
    all_samples = []
    path = os.path.join(args.output_dir, 'temp_results')
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        with open(fpath, "r") as ff:
            js = json.load(ff)
        all_samples.append(js)
    df = pd.DataFrame.from_dict(all_samples)
    print('Number of samples loaded from json: ', len(df))

#print(df)
#df = df.head(1165)
df["preds"] = [predscleaner(x) for x in df["prediction"]]
df["gold"] = [goldcleaner_function(x) for x in df['transcription']]
df["wer"] = [jiwer.wer(gold, pred) for gold, pred in zip(df["gold"], df["preds"])]

# Save to CSV
df.to_csv("wer_results.csv", index=False)
wer = jiwer.wer(list(df["gold"]),list(df["preds"]))
print(args.output_dir, wer)
