import os
import fire
from tqdm import tqdm
from evaluator.data import stream_jsonl


def analyze_time(sample_file):
    latency, n_samples = 0.0, 0
    samples = stream_jsonl(sample_file)
    for i, sample in tqdm(enumerate(samples), colour='green', desc="Analyzing time"):
        if i == 0:
            continue  # skip the first sample since it includes GPU kernel startup time
        try:
            latency += sample['result'][0]['decode_time'] / sample['result'][0]['num_gen']
        finally:
            n_samples += 1
    return latency / n_samples if n_samples > 0 else float('nan')


def analyze_acceptance_rate(sample_file):
    acc_rate, n_samples = 0.0, 0
    samples = stream_jsonl(sample_file)
    for sample in tqdm(samples, colour='green', desc="Analyzing acceptance rate"):
        try:
            acc_rate += sample['result'][0]['acc_rate']
        finally:
            n_samples += 1
    return acc_rate / n_samples if n_samples > 0 else float('nan')
