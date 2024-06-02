import sys
sys.path.append("..")

from evaluator.data import stream_jsonl

FILE = "240529/results/___time_summary_gamma-8___"

time_summary = list(stream_jsonl(FILE))

draft, target, val = 0.0, 0.0, 0.0

for run in time_summary:
    d, t, v = run['draft'], run['target'], run['val']
    denom = d + t + v

    d /= denom
    t /= denom
    v /= denom

    draft += d
    target += t
    val += v

draft /= len(time_summary)
target /= len(time_summary)
val /= len(time_summary)

print(draft, target, val)