# Kernel RL

RL for kernel generation with Unsloth GRPO.

- Main training script: `main.py`
- Original reference code: `reference.py`
- Original notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
-Blog post: https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning

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
