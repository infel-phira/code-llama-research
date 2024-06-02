import os
import torch
from datetime import datetime
from tqdm import tqdm
from code_generator import CodeGenerator
from evaluator.data import read_problems, write_jsonl


def load_models(target_model, draft_model=None):
    return CodeGenerator(
        target_model=target_model, draft_model=draft_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def pass_at_k(
    target_model, draft_model=None, method=None,
    k=1, max_len=512, temperature=0.2, output_jsonl=None, **kwargs
):
    if output_jsonl is None:
        output_jsonl = f"outputs/{os.path.splitext(os.path.basename(target_model))[0]}__"
        if method == 'speculative' and draft_model is not None:
            output_jsonl += f"{os.path.splitext(os.path.basename(draft_model))[0]}__"
        output_jsonl += f"pass@{k}__{datetime.now().strftime('%y%m%d_%H%M%S')}.jsonl"

    print(f"Outputs will be stored in {output_jsonl}.")

    code_llama = load_models(target_model, draft_model)
    problems = read_problems()

    samples = []
    for task_id in tqdm(problems, colour="green", total=len(problems), desc=f"Running pass@{k}"):
        result = [
            code_llama.infer(
                problems[task_id]['prompt'],
                method=method, max_len=max_len, temperature=temperature, **kwargs
            ) for _ in range(k)
        ]
        samples.append( dict(task_id=task_id, result=result) )
    
    write_jsonl(output_jsonl, samples)


if __name__ == '__main__':
    # for _ in range(1):
    #     pass_at_k(
    #         target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-7b-Python-hf',
    #         method='autoregressive',
    #         k=1, max_len=512, temperature=0.2
    #     )

    # for _ in range(1):
    #     pass_at_k(
    #         target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-13b-Python-hf',
    #         method='autoregressive',
    #         k=1, max_len=512, temperature=0.2
    #     )

    # for _ in range(1):
    #     pass_at_k(
    #         target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-34b-Python-hf',
    #         method='autoregressive',
    #         k=1, max_len=512, temperature=0.2
    #     )
    
    # for _ in range(1):
    #     pass_at_k(
    #         target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-13b-Python-hf',
    #         draft_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-7b-Python-hf',
    #         method='speculative', gamma=2,
    #         k=1, max_len=512, temperature=0.2
    #     )

    for _ in range(1):
        pass_at_k(
            target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-34b-Python-hf',
            draft_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-7b-Python-hf',
            method='speculative', gamma=3,
            k=1, max_len=512, temperature=0.2
        )