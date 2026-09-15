"""
Merge a trained LoRA adapter into the base checkpoint and save a standalone
merged model. A single always-on adapter for a locked Kannada-only deployment
is simpler to operate as a merged checkpoint than via vLLM's --enable-lora
hot-swap machinery, which is built for a multi-adapter use case we don't have.

Usage:
    source training/setup_env.sh
    python training/merge_and_export.py --run_name kn_lora_full
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_and_export")

BASE_MODEL_NAME = "kenpath/svara-tts-v1"
RUNS_DIR = Path("/mnt/siet_llm_data/svara_training/runs")
CHECKPOINTS_DIR = Path("/mnt/siet_llm_data/svara_training/checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, help="Matches the --run_name used in train_lora.py")
    parser.add_argument("--output_name", default=None, help="Defaults to svara-tts-v1-kn-lora-<today>")
    args = parser.parse_args()

    adapter_dir = RUNS_DIR / args.run_name / "lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_dir} - did train_lora.py finish and save?")

    output_name = args.output_name or f"svara-tts-v1-kn-lora-{date.today().isoformat()}"
    output_dir = CHECKPOINTS_DIR / output_name

    log.info("Loading base model %s", BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")

    log.info("Loading LoRA adapter from %s", adapter_dir)
    merged = PeftModel.from_pretrained(base_model, str(adapter_dir))

    log.info("Merging adapter into base weights")
    merged = merged.merge_and_unload()

    log.info("Saving merged checkpoint to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))

    log.info("Done. Merged checkpoint at %s", output_dir)
    log.info("Next: python training/evaluate_samples.py --checkpoint %s --run_name %s_vs_base", output_dir, args.run_name)


if __name__ == "__main__":
    main()
