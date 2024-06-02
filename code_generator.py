import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

__epsilon__ = 1e-6


def enable_timer(func):  # descriptor
    def inner(*args, **kwargs):
        begin_time = time.perf_counter()
        retval = func(*args, **kwargs)
        return time.perf_counter() - begin_time, retval
    return inner


def exec_once(func):  # descriptor
    flag = False
    def inner(*args, **kwargs):
        nonlocal flag
        if not flag:
            flag = True
            return func(*args, **kwargs)
    return inner


def update_total(bar, total):
    counter = bar.n
    bar.reset(total=total)
    bar.update(counter)


@exec_once
def check_kvcache(outputs):
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        print("KV Cache is enabled. #layers in KV Cache:", len(outputs.past_key_values))
    else:
        print("KV Cache is not enabled.")


class CodeGenerator:
    def __init__(self, target_model, draft_model, device='cpu'):
        self.target = self.model_from_pretrained(target_model)
        self.draft = self.model_from_pretrained(draft_model) if draft_model is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.device = device

        self._eofn = self.tokenizer.encode("\n\n")[-2:]
        self._eol = self.tokenizer.encode("\n")[-1]
        self._comm = self.tokenizer.encode("#")[-2]


    @staticmethod
    def model_from_pretrained(model, dtype=torch.float16):
        print(f"Loading model {model}...")
        return AutoModelForCausalLM.from_pretrained(model, device_map='auto', torch_dtype=dtype)
    

    @staticmethod
    def score_to_prob(scores, temperature=0.0):
        if temperature <= __epsilon__:
            return F.one_hot(torch.argmax(scores, dim=1), num_classes=scores.shape[1]).float()  # use argmax
        else:
            return torch.softmax(scores / temperature, dim=1)  # use softmax with temperature


    @enable_timer
    def encode(self, prompt):
        return self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)


    @enable_timer
    def decode(self, tokens):
        return self.tokenizer.decode(tokens[0, :], skip_special_tokens=True)
    

    def check_eofn(self, tokens):
        return tokens[0, -1] == self._eofn[-1] and tokens[0, -2] == self._eofn[-2]


    def check_comment(self, tokens):
        for i in range(tokens.shape[1] - 1, -1, -1):
            if tokens[0, i] == self._eol:
                return False
            elif tokens[0, i] == self._comm:
                return True
        return False


    @enable_timer
    @torch.no_grad()
    def autoregressive(self, tokens, max_len=512, temperature=0.0):
        ctx_len = tokens.shape[1]
        seq_len = ctx_len + max_len

        while tokens.shape[1] < seq_len:
            outputs = self.target(tokens)
            output_logits = outputs.logits[:, -1, :]  # shape: (1, num_tokens, vocab_size)
            check_kvcache(outputs)
            
            output_prob = self.score_to_prob(output_logits, temperature=temperature)  # convert scores to probabilities
            next_token = torch.multinomial(output_prob, num_samples=1)  # select one token under softmax'd probability distribution
            tokens = torch.cat((tokens, next_token), dim=1)  # append the new token to the token list

            if self.check_eofn(tokens):
                break

        return tokens, (tokens.shape[1] - ctx_len,)


    @enable_timer
    @torch.no_grad()
    def speculative(self, tokens, gamma=3, max_len=512, temperature=0.0):
        ctx_len = tokens.shape[1]
        seq_len = ctx_len + max_len

        acc_rate = []

        ___total_draft___ = 0.0
        ___total_target___ = 0.0
        ___total_val___ = 0.0
        ___total_resamp___ = 0.0

        while tokens.shape[1] < seq_len:
            prefix_len = tokens.shape[1]
            
            ___begin_time___ = time.perf_counter()  ######## DRAFT BEGIN

            for _ in range(gamma):
                outputs = self.draft(tokens)
                output_logits = outputs.logits[:, -1, :]
                check_kvcache(outputs)
                
                output_prob = self.score_to_prob(output_logits, temperature=temperature)
                next_token = torch.multinomial(output_prob, num_samples=1)
                tokens = torch.cat((tokens, next_token), dim=1)
            
            ___total_draft___ += time.perf_counter() - ___begin_time___  ######## DRAFT END
            ___begin_time___ = time.perf_counter()  ######## TARGET BEGIN

            target_logits = self.target(tokens).logits[:, -gamma - 1:, :]
            draft_logits = outputs.logits[:, -gamma:, :]

            ___total_target___ += time.perf_counter() - ___begin_time___  ######## TARGET END
            ___begin_time___ = time.perf_counter()  ######## VALIDATION BEGIN

            n_accepted = gamma  # initialize as all accepted
            reach_eofn = False

            for i in range(gamma):
                if self.check_comment(tokens[:, :prefix_len + i + 1]):
                    continue

                if self.check_eofn(tokens[:, :prefix_len + i + 1]):
                    tokens = tokens[:, :prefix_len + i + 1]
                    reach_eofn = True
                    break

                curr_token = tokens[0, prefix_len + i]

                target_prob = self.score_to_prob(target_logits[:, i, :], temperature=temperature)
                draft_prob = self.score_to_prob(draft_logits[:, i, :], temperature=temperature)
                ratio = target_prob[0, curr_token] / draft_prob[0, curr_token]

                if torch.rand(1, device=self.device) > ratio:
                    n_accepted = i
                    break
            
            acc_rate.append(n_accepted / gamma)

            ___total_val___ += time.perf_counter() - ___begin_time___  ######## VALIDATION END
            ___begin_time___ = time.perf_counter()  ######## RESAMPLING BEGIN

            if reach_eofn:
                break

            sample_prob = self.score_to_prob(target_logits[:, n_accepted, :], temperature=temperature)
            
            if n_accepted < gamma:
                tokens = tokens[:, :prefix_len + n_accepted]

                sample_prob = sample_prob - self.score_to_prob(draft_logits[:, n_accepted, :], temperature=temperature)
                sample_prob = torch.where(sample_prob > 0.0, sample_prob, 0.0)
                sample_prob = sample_prob / torch.sum(sample_prob, dim=1, keepdim=True)
            
            next_token = torch.multinomial(sample_prob, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

            ___total_resamp___ += time.perf_counter() - ___begin_time___  ######## RESAMPLING END
        
        with open("___time_summary___", 'a') as ___time_summary___:
            ___time_summary___.write(json.dumps({
                "draft": ___total_draft___,
                "target": ___total_target___,
                "val": ___total_val___,
                "resamp": ___total_resamp___
            }) + '\n')

        return tokens, (tokens.shape[1] - ctx_len, sum(acc_rate) / len(acc_rate))


    def infer(self, input, method=None, silent=True, **kwargs):
        if method is None or method == 'autoregressive' or method == 'auto-regressive':
            decoder = self.autoregressive
        elif method == 'speculative':
            decoder = self.speculative
        else:
            raise ValueError(f"Invalid method: `{method}`!")
        
        prefill_time, tokens = self.encode(input)
        decode_time, (tokens, metadata) = decoder(tokens, **kwargs)
        _, output = self.decode(tokens)

        num_gen = metadata[0]
        acc_rate = metadata[1] if len(metadata) > 1 else None

        if not silent:
            print(f"\nPrefilling time:  {prefill_time:.6f} s")
            print(f"Decoding time:    {decode_time:.6f} s\n")

        return dict(
            prefill_time=prefill_time,
            decode_time=decode_time,
            num_gen=num_gen,
            acc_rate=acc_rate,
            completion=output
        )


if __name__ == '__main__':
    code_llama = CodeGenerator(
        target_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-13b-Python-hf',
        draft_model='/scratch/bcjw/shihan3/models/codellama/CodeLlama-7b-Python-hf',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(code_llama.infer("def fibonacci():", max_len=128, temperature=0.2, silent=False)['completion'])