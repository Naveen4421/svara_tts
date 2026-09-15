#!/bin/bash
# Source this (don't execute) to enter the training environment:
#   source training/setup_env.sh
#
# Creates (on first run) an isolated venv on /mnt/siet_llm_data - the repo's own
# filesystem is at 100% usage, so nothing training-related may land under /.
set -e

# The venv itself MUST be on local disk: /mnt/siet_llm_data is a CIFS mount
# (nounix) that rejects both symlink creation and the atomic rename-with-
# permission-set pattern `pip install` uses, even for a single small file.
# Confirmed empirically - see training/README.md. Plain data (HF cache, pip's
# wheel cache, datasets, checkpoints) works fine there; installed packages don't.
TRAIN_ROOT=/mnt/siet_llm_data/svara_training
VENV_DIR="/home/sietllm/svara_training_venv"

export HF_HOME="$TRAIN_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$TRAIN_ROOT/pip_cache"

if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3.12 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ "$1" == "--install" ]; then
    pip install --upgrade pip
    # Pin torch/CUDA first - same cu128 combo the inference stack already
    # validates against this host's driver (570.211).
    pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    # --no-deps so Unsloth can't silently pull a different torch build.
    pip install --no-deps unsloth unsloth_zoo
    # Pin to unsloth_zoo's actual supported ranges (checked via its own metadata,
    # not guessed) - letting pip pick "latest" here resolved transformers 5.17.0
    # and datasets 5.0.1 / trl 1.13.0, all outside what unsloth_zoo supports.
    # 4.55.4 also matches this base model's own config.json transformers_version.
    pip install "transformers==4.55.4" "datasets>=3.4.1,<4.4.0,!=4.0.*,!=4.1.0" "trl>=0.18.2,<=0.24.0,!=0.19.0"
    pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements-train.txt"
    echo "--- sanity check ---"
    python -c "
import torch, unsloth
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
import transformers, datasets, trl, peft, accelerate, bitsandbytes, snac, soundfile
print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'trl', trl.__version__, 'peft', peft.__version__)
"
fi
