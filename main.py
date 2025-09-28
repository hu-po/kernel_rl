import ast
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import signal
import statistics
import sys
import sysconfig
import time
import types

from datasets import Dataset
import numpy as np
import torch
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

max_seq_length = 640 # Can increase for longer RL output
lora_rank = 4 # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    offload_embedding = True, # Reduces VRAM by 1GB
)



model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

 
def generate_random_matrices(seed = 3407, n = 256):
    random_state = np.random.RandomState(seed)
    n, k, m = random_state.randint(1, n+1, size = 3)
    A = np.random.uniform(-10, 10, size = (n, k))
    B = np.random.uniform(-10, 10, size = (k, m))
    return A, A.tolist(), B, B.tolist()



A, A_list, B, B_list = generate_random_matrices(seed = 42, n = 5)
print(A)
print(B)
print(np.matmul(A, B))



def calculate_difference(pred, real):
    if pred is None: return 3, 3
    assert real is not None
    difference = pred - real
    amax_error = float(np.amax(difference))
    mse_error  = float(np.mean(np.square(difference)))
    return amax_error, mse_error

def matmul(A, B):
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]



prediction = matmul(A_list, B_list)
calculate_difference(prediction, np.matmul(A, B))

def _stdlib_names():
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__")  # special-case

    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        pass

    return names

_STDLIB_SET = _stdlib_names()

def check_only_stdlib_imports(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])
        def visit_ImportFrom(self, node: ast.ImportFrom):
            nonlocal relative_count
            if (node.level or 0) > 0:
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }



sample = """
def matmul(A, B):
    import numpy as np
    from torch import matmul
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]
"""
ok, info = check_only_stdlib_imports(sample)
print("Only stdlib imports?", ok)
print(info)



output_function = {}
exec(sample, {}, output_function)
output_function["matmul"]


output_function["matmul"] = types.FunctionType(output_function["matmul"].__code__, {})

def import_numpy():
    np.matmul
    print("Success")

import_numpy()
import_numpy = types.FunctionType(import_numpy.__code__, {})
try:
    import_numpy()
except Exception as e:
    print(str(e))

def create_locked_down_function(function):
    output_function = {}
    exec(function, {}, output_function)
    new_matmul = output_function["matmul"]
    new_matmul = types.FunctionType(new_matmul.__code__, {})
    return new_matmul


class TimeoutError(Exception): pass

@contextmanager
def time_limit(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)

class Benchmarker:
    def __init__(self, trials = 3, loops = 1, timeout = 30):
        self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype = np.uint8)
        self.trials = trials
        self.loops = loops
        assert timeout > 0 # Cannot be 0 since it won't work!
        self.timeout = timeout
    def thrash(self):
        self.buffer ^= 1
        return int(self.buffer[::4096].sum())

    def benchmark(self, function, arguments):
        assert len(arguments) == self.loops
        samples = []
        exceptions = []
        timed_out = 0
        for _ in range(self.trials):
            gc.collect(); gc.disable(); self.thrash()
            t_start = time.perf_counter_ns()
            for i in range(self.loops):
                try:
                    with time_limit(self.timeout):
                        function(*arguments[i])
                except TimeoutError as e:
                    timed_out += 1
                except Exception as e:
                    exceptions.append(str(e))
            t_end = time.perf_counter_ns()
            gc.enable()
            samples.append((t_end - t_start) // max(1, self.loops))
        return {
            "median_ns": int(statistics.median(samples)),
            "mean_ns": int(statistics.fmean(samples)),
            "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
            "exceptions" : exceptions,
            "timeouts" : timed_out,
        }



A, A_list, B, B_list = generate_random_matrices(seed = 0, n = 256)
Benchmarker(trials = 1, timeout = 10).benchmark(output_function["matmul"], [(A_list, B_list)])



prompt = """
Create a new fast matrix multiplication function using only native Python code.
You are given a list of list of numbers.
Output your function in backticks using the format below:
```python
def matmul(A, B):
    return [[sum(a*b for a, b in zip(row, col)) for col in list(zip(*B))] for row in A]
```
""".strip()
print(prompt)



text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 512,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)



def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first : second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def matmul(A, B):"): return fx
    return None
print(extract_function(prompt))



ok, info = check_only_stdlib_imports("def a")
ok, info

def function_works(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        print(function)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_matmul = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores



def no_cheating(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        else:
            ok = False
        scores.append(1.0 if ok else -20.0) # Penalize heavily!
    return scores



np.finfo(np.float64).eps

def correctness_check(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        A, A_list, B, B_list = generate_random_matrices(seed = np.random.randint(10000), n = 128)
        try:
            pred = new_matmul(A_list, B_list)
        except:
            scores.append(-2.0)
            continue
        true = np.matmul(A, B)
        amax_error, mse_error = calculate_difference(pred, true)

        machine_epsilon = 100*np.finfo(np.float64).eps
        if   amax_error >= 3:   score = -3.0
        elif amax_error >= 2:   score = -2.5
        elif amax_error >= 1:   score = -2.0
        elif amax_error >= 0.5: score = -1.0
        elif amax_error >= 100*machine_epsilon: score = 0.0
        elif amax_error >= machine_epsilon: score = 1.0
        else: score = 3.0

        if   mse_error >= 3:   score += -3.0
        elif mse_error >= 2:   score += -2.5
        elif mse_error >= 1:   score += -2.0
        elif mse_error >= 0.5: score += -1.0
        elif mse_error >= 100*machine_epsilon: score += 0.0
        elif mse_error >= machine_epsilon: score += 1.0
        else: score += 3.0
        scores.append(score)
    return scores



A, A_list, B, B_list = generate_random_matrices(seed = 0, n = 256)
benchmarker = Benchmarker(trials = 3, timeout = 10)
numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
numpy_results

new_matmul = create_locked_down_function(extract_function(prompt))
new_results = benchmarker.benchmark(new_matmul, [(A_list, B_list)])
new_results



negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
reward = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
reward

new_results["median_ns"] = 3
numpy_results["median_ns"] = 1000
negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
reward = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
reward

def speed_check(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        A, A_list, B, B_list = generate_random_matrices(seed = np.random.randint(10000), n = 256)
        numpy_results = benchmarker.benchmark(np.matmul,  [(A, B)])
        new_results   = benchmarker.benchmark(new_matmul, [(A_list, B_list)])

        negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
        positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
        score = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
        if score >= 10:  score = 10
        if score <= -10: score = -10
        scores.append(score)
    return scores


dataset = Dataset.from_list([{"prompt" : [{"role": "user", "content": prompt.strip()}], "answer" : 0}]*1000)
maximum_length = len(tokenizer(prompt.strip())["input_ids"])
print(maximum_length)
dataset[0]



max_prompt_length = maximum_length + 1 # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 100,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",

)



trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,
        no_cheating,
        correctness_check,
        speed_check,
    ],
    args = training_args,
    train_dataset = dataset,
)



trainer.train()



text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)



if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "mxfp4")
if False: model.push_to_hub_merged("repo_id/repo_name", tokenizer, token = "hf...", save_method = "mxfp4")

if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "merged_16bit")
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method = "merged_16bit", token = "")
