# Kannada LoRA fine-tuning

Full plan: see the conversation this was built from, or re-derive from the scripts
below (each is self-contained and commented with its reasoning).

## Storage layout (important, don't move things)

`/mnt/siet_llm_data` is a CIFS mount (`nounix`) that rejects both symlink creation
and the atomic rename-with-permission-set pattern `pip install` uses, even for a
single small file — confirmed empirically, not a guess. It works fine for **plain
data** (HF model cache, pip's wheel cache, datasets, checkpoints, eval outputs) but
cannot host an installed Python environment.

- **Venv + installed packages**: `/home/sietllm/svara_training_venv` (local disk).
- **Everything else** (HF cache, dataset, checkpoints, eval outputs): under
  `/mnt/siet_llm_data/svara_training/`.

Root disk (`/`) had only 5.3GB free at the start of this work; freed to ~178GB via
`docker image prune -a` + `docker volume prune` (only removed images/volumes not
attached to any running container — nothing on other host services was touched).

## Usage

```bash
source training/setup_env.sh --install   # first time only
python training/prepare_dataset.py
python training/train_lora.py --smoke_test   # Stage A, run this first
python training/train_lora.py                # Stage B, full run
python training/merge_and_export.py --run_name <name>
python training/evaluate_samples.py --checkpoint <merged_checkpoint_path>
```
