"""
This script trains a model with GRPO using a composite reward made up of four
independent components:

- function_works:
  Goal: Ensure the completion forms a valid, lockable Python function.
  Logic: Extract, static-check imports, attempt to build the function.
  Range: Approximately [-2.0, +1.0].
    - -2.0 if missing/invalid function or AST parse error
    - -0.5 if creation fails at runtime
    - +1.0 if creation succeeds

- no_cheating:
  Goal: Discourage importing non-stdlib modules (e.g., NumPy, Torch).
  Logic: 1.0 if only stdlib imports; -20.0 otherwise (or if no function).
  Range: {-20.0, +1.0}. Heavy penalty dominates to strongly prevent cheating.

- correctness_check:
  Goal: Encourage numerically correct outputs on random test cases.
  Logic: Compare predicted matrix product vs. NumPy ground truth using absolute
  max error and mean squared error with thresholds near machine precision.
  Range: Approximately [-6.0, +6.0], combining two sub-scores.

- speed_check:
  Goal: Reward faster-than-NumPy implementations; penalize slower ones.
  Logic: Benchmark median time vs. NumPy and transform the ratio into a score
  scaled by 1/100 and clipped to [-10, +10]. Faster gets positive, slower
  negative. Multiple trials and cache-thrashing reduce noise.

"""

import ast
from dataclasses import dataclass
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
import logging

from unsloth import FastLanguageModel
from datasets import Dataset
import numpy as np
from trl import GRPOConfig, GRPOTrainer
import wandb

os.environ["WANDB_ENTITY"] = "hug"
os.environ["WANDB_PROJECT"] = "kernel_rl"
wandb.login()

@dataclass
class TrainingConfig:
    prompt: str = """
Create a new fast matrix multiplication function using only native Python code.
You are given a list of list of numbers.
Output your function in backticks using the format below:
```python
def matmul(A, B):
    return [[sum(a*b for a, b in zip(row, col)) for col in list(zip(*B))] for row in A]
```
""".strip()

    max_seq_length: int = 640 # increase for longer output
    lora_rank: int = 4 # larger rank = smarter, but slower (> 0 ! Suggested 8, 16, 32, 64, 128)
    model_name: str = "unsloth/gpt-oss-20b"
    load_in_4bit: bool = True # false for LoRA 16bit
    offload_embedding: bool = True # reduces VRAM by 1GB
    random_state: int = 334
    num_trials: int = 3 # number of trials per benchmark 
    temperature: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1  # Increase to 4 for smoother training
    num_generations: int = 2 # Decrease if out of memory
    max_steps: int = 1000
    save_steps: int = 100
    report_to: str = "wandb"
    output_dir: str = "outputs"

def main(config: TrainingConfig):
    log = logging.getLogger(__name__)
    log.info(f"Training config: {config}")

    log.info("Loading pretrained model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
        max_seq_length = config.max_seq_length,
        load_in_4bit = config.load_in_4bit,
        offload_embedding = config.offload_embedding,
    )

    log.info("Loading LoRA weights...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = config.lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = config.random_state,
    )

 
    def generate_random_matrices(seed = 3407, n = 256):
        random_state = np.random.RandomState(seed)
        n, k, m = random_state.randint(1, n+1, size = 3)
        A = np.random.uniform(-10, 10, size = (n, k))
        B = np.random.uniform(-10, 10, size = (k, m))
        return A, A.tolist(), B, B.tolist()

    def calculate_difference(pred, real):
        # Return a large penalty (3, 3) when we cannot compare.
        if pred is None:
            return 3, 3
        assert real is not None

        # Convert to numpy arrays to standardize types (lists, tensors, etc.).
        try:
            pred_arr = np.asarray(pred, dtype=float)
        except Exception:
            return 3, 3
        try:
            real_arr = np.asarray(real, dtype=float)
        except Exception:
            return 3, 3

        # Shapes must match for elementwise difference.
        if pred_arr.shape != real_arr.shape:
            return 3, 3

        # Guard against NaN/Inf outputs from model code.
        if not np.all(np.isfinite(pred_arr)):
            return 3, 3

        difference = pred_arr - real_arr
        # Use absolute max error; MSE already squares.
        amax_error = float(np.max(np.abs(difference)))
        mse_error  = float(np.mean(np.square(difference)))
        return amax_error, mse_error

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
        """
        Uses Python AST to list absolute imports and checks
        that they belong to the standard library. Relative imports are counted but not
        penalized here. Returns a boolean and details used by other rewards.
        """
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

    def create_locked_down_function(function):
        """
        Safely materializes the generated function by
        executing code with empty locals/globals and then re-creating the function
        with empty globals via `types.FunctionType`. This prevents access to global
        state and minimizes cheating avenues (e.g., importing optimized libraries or
        reading external variables).
        """
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
        def __init__(self, trials = config.num_trials, loops = 1, timeout = 30):
            self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype = np.uint8)
            self.trials = trials
            self.loops = loops
            assert timeout > 0 # Cannot be 0 since it won't work!
            self.timeout = timeout
        def thrash(self):
            self.buffer ^= 1
            return int(self.buffer[::4096].sum())

        def benchmark(self, function, arguments):
            """
            Runs a function under a strict timeout and with
            cache-thrashing between trials to reduce caching effects. Produces timing
            statistics (median/mean/stdev in nanoseconds), plus exception and timeout
            counts. Used by the speed reward to compare against NumPy.
            """
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

    benchmarker = Benchmarker()

    def extract_function(text):
        """
        Parses a completion and extracts a Python function enclosed
        in triple backticks. It expects a signature starting with `def matmul(A, B):`.
        If extraction fails, many rewards default to penalties or zero.
        """
        if text.count("```") >= 2:
            first = text.find("```") + 3
            second = text.find("```", first)
            fx = text[first : second].strip()
            fx = fx.removeprefix("python\n")
            fx = fx[fx.find("def"):]
            if fx.startswith("def matmul(A, B):"): return fx
        return None

    def function_works(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
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
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                ok, info = check_only_stdlib_imports(function)
            else:
                ok = False
            scores.append(1.0 if ok else -20.0) # Penalize heavily!
        return scores

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

    dataset = Dataset.from_list([{"prompt" : [{"role": "user", "content": config.prompt.strip()}], "answer" : 0}]*1000)
    maximum_length = len(tokenizer(config.prompt.strip())["input_ids"])
    max_prompt_length = maximum_length + 1 # + 1 just in case!
    max_completion_length = config.max_seq_length - max_prompt_length

    training_args = GRPOConfig(
        temperature = config.temperature,
        learning_rate = config.learning_rate,
        weight_decay = config.weight_decay,
        warmup_ratio = config.warmup_ratio,
        lr_scheduler_type = config.lr_scheduler_type,
        optim = config.optim,
        logging_steps = config.logging_steps,
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        num_generations = config.num_generations,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = config.max_steps,
        save_steps = config.save_steps,
        report_to = config.report_to,
        output_dir = config.output_dir,
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


if __name__ == "__main__":
    config = TrainingConfig()
    main(config)
