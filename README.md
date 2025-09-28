# kernel_rl

RL for kernel generation with Unsloth GRPO.

Reference: https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning

## Quickstart (uv)

- Prereqs: Python 3.9+ and `uv` installed. If you need `uv`:
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`

### Create and activate a virtualenv

- Create venv (choose a Python version you have installed):
  - `uv venv .venv --python 3.10`
- Activate:
  - Linux/macOS: `source .venv/bin/activate`
  - Windows (PowerShell): `.venv\Scripts\Activate.ps1`

### Install dependencies

- CPU/default install (from `pyproject.toml`):
  - `uv pip install -e .`

- Optional: install CUDA-enabled PyTorch wheels (pick the CUDA that matches your driver). Example for CUDA 12.1:
  - `uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`

Notes:
- `original.py` uses 4-bit loading via `bitsandbytes` and will require a compatible NVIDIA GPU + CUDA for acceleration.
- Pinned versions (from the notebook): `transformers==4.56.2`, `trl==0.22.2`; lower bounds: `torch>=2.8.0`, `triton>=3.4.0`.

### Run

- `python original.py`
