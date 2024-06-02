import os
import shutil
import sys
import fire
from evaluator.data import HUMAN_EVAL
from evaluator.evaluation import evaluate_functional_correctness
from refine_jsonl import refine
from analyze import analyze_time, analyze_acceptance_rate


def entry_point(
    sample_file: str,
    k: str = "1",  # "1,10,100"
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    tmp_dir = "temp/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    tmp_file = os.path.join(tmp_dir, "refined.jsonl")
    refine(sample_jsonl=sample_file, output_jsonl=tmp_file)

    k = list(map(int, str(k).split(",")))
    results = evaluate_functional_correctness(tmp_file, k, n_workers, timeout, problem_file)
    results['avg_latency'] = analyze_time(sample_file)
    results['avg_acc_rate'] = analyze_acceptance_rate(sample_file)
    print(results)

    results_dir = os.path.join(os.path.dirname(sample_file), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    result_file = os.path.join(
        results_dir,
        os.path.basename(sample_file) + "__results.jsonl"
    )

    with open(result_file, 'w') as fout:
        fout.write(repr(results).replace("'", '"') + "\n")
        with open(tmp_file + "_results.jsonl", 'r') as fin:
            fout.write(fin.read())

    os.remove(tmp_file)
    os.remove(tmp_file + "_results.jsonl")
    os.rmdir(tmp_dir)


def main():
    fire.Fire(entry_point)

sys.exit(main())
