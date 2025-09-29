# Kernel RL

RL for kernel generation with Unsloth GRPO.

- Main training script: `main.py`
- Original Unsloth reference code: `reference.py`
- Original Unsloth notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
- Unsloth log post: https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning
- wandb workspace: https://wandb.ai/hug/kernel_rl
- Blog post: https://www.aleksagordic.com/blog/matmul

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# setup venv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r pyproject.toml
# gpu torch
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# run main training script
python main.py
```
