# Kernel RL

RL for kernel generation with Unsloth GRPO.

- Main training script: `main.py`
- Original Unsloth reference code: `reference.py`
- Original Unsloth notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb
- Unsloth blog post: https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning
- wandb workspace: https://wandb.ai/hug/kernel_rl
- Blog post: https://www.aleksagordic.com/blog/matmul

```bash
# clone this repo
git clone https://github.com/hu-po/kernel_rl.git
cd kernel_rl
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# setup venv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r pyproject.toml
# check version of cuda on machine
nvidia-smi
# install cuda enabled torch (3090)
uv pip install  torch torchvision --index-url https://download.pytorch.org/whl/cu121
# install cuda enabled torch (H100)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# login to wandb
wandb login
huggingface-cli login
# run main training script
python main.py
```
