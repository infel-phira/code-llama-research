import sys
sys.path.append("..")

from evaluator.data import stream_jsonl

CMP_FILE_1 = "240528/CodeLlama-13b-Python-hf__CodeLlama-7b-Python-hf__pass@1__240527_183544__GAMMA-4__results.jsonl"
CMP_FILE_2 = "240528/CodeLlama-13b-Python-hf__CodeLlama-7b-Python-hf__pass@1__240527_190341__GAMMA-8__results.jsonl"

result_1 = list(stream_jsonl(CMP_FILE_1))
result_2 = list(stream_jsonl(CMP_FILE_2))

assert len(result_1) == len(result_2), "Bad apple"

for i in range(1, len(result_1)):
    assert result_1[i]['task_id'] == result_2[i]['task_id'], "Bad apple"

    if result_1[i]['passed'] == result_2[i]['passed']:
        continue

    print("TASK ID:")
    print(result_1[i]['task_id'])
    print("COMPLETION 1:", result_1[i]['passed'])
    print(result_1[i]['completion'])
    print("COMPLETION 2:", result_2[i]['passed'])
    print(result_2[i]['completion'])

    input("Press enter to continue...")
