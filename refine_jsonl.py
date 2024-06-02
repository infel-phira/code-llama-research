import os
import fire
from tqdm import tqdm
from evaluator.data import stream_jsonl, write_jsonl


def refine(sample_jsonl, output_jsonl=None):
    if output_jsonl is None:
        output_jsonl = os.path.splitext(sample_jsonl)[0] + "__refine.jsonl"

    samples = []
    for sample in tqdm(stream_jsonl(sample_jsonl), colour='green'):
        samples.extend([
            dict(
                task_id=sample['task_id'],
                completion=run['completion']
            ) for run in sample['result']
        ])

    write_jsonl(output_jsonl, samples)


if __name__ == '__main__':
    fire.Fire(refine)