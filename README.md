# Toward Agents That Reason About Their Computation

Vectorised-training Atari: each run trains N independent agent-env pairs (N seeds) in one
process. The N networks are stacked into one TensorDict and driven with `torch.func.vmap`, so
inference and the update run as single batched kernels over the stack.

## Install (uv)

Requires Python 3.13, CUDA 12.8, and an Nvidia GPU.

```bash
uv venv --python 3.13
source .venv/bin/activate

# PyTorch (CUDA 12.8)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Training deps
uv pip install tensordict nvidia-nvcomp-cu12 gymnasium tyro tqdm numpy

# ALE with the multi-rom vector env (the feat/multi-rom branch of the fork)
uv pip install "ale-py @ git+https://github.com/AdrianOrenstein/Arcade-Learning-Environment.git@feat/multi-rom"

# Atari ROMs
uv pip install "autorom[accept-rom-license]"
mkdir -p "$HOME/.atari-roms"
AutoROM --accept-license --install-dir "$HOME/.atari-roms"
export ALE_ROMS_DIR="$HOME/.atari-roms"
```

`nvidia-nvcomp-cu12` powers the default `--replay-compress nvcomp` buffer. To run without it, pass
`--replay-compress bitpack`.

## Run

Each script trains `N` seeds at once (`--num-agent-env-pairs`, default 8). The env
uses `frameskip=5`, so 200M frames is 40M agent steps per seed (`--total-timesteps 40_000_000`).

The scripts import as `src.*`, so put the repo root on `PYTHONPATH` (e.g. the container path
`/app/projects/toward_camera_ready`, or `$(pwd)` from the repo root):

```bash
export PYTHONPATH=/app/projects/toward_camera_ready  # or: export PYTHONPATH=$(pwd)

# DQN
python src/dqn.py \
  --env-id ALE/Pong-v5 --total-timesteps 40_000_000 --num-agent-env-pairs 8 \
  --metrics-path results/dqn_pong.jsonl

# Compute-DQN (temporal options: each decision commits 1, 2, or 4 repeats)
python src/compute_dqn.py \
  --env-id ALE/Pong-v5 --total-timesteps 40_000_000 --num-agent-env-pairs 8 \
  --option-lengths 1 2 4 \
  --metrics-path results/compute_dqn_pong.jsonl

# PPO
python src/ppo.py \
  --env-id ALE/Pong-v5 --total-timesteps 40_000_000 --num-agent-env-pairs 8 \
  --metrics-path results/ppo_pong.jsonl
```

Metrics (per-step return and speed) stream as JSON lines to `--metrics-path`.

# Citation

```bibtex
@article{orensteinAgentsThatReason2026,
  title = {{{Toward Agents That Reason About Their Computation}}},
  author = {Orenstein, Adrian and Chen, Jessica and Santos, Gwyneth Anne Delos and Sapara, Bayley and Bowling, Michael},
  date = 2026,
  journal = "Reinforcement Learning Journal (RLJ)"
}
```
