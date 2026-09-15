"""
LoRA fine-tune kenpath/svara-tts-v1 on the processed Kannada dataset built by
prepare_dataset.py. Uses Unsloth for the LoRA setup (matches how the base
checkpoint was itself trained) and plain transformers.Trainer - not
SFTTrainer, since our examples are already tokenized input_ids/labels pairs,
not raw text to be reformatted.

Usage:
    source training/setup_env.sh
    python training/train_lora.py --smoke_test          # Stage A: 300 rows, 1 epoch
    python training/train_lora.py                        # Stage B: full run
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_from_disk
from unsloth import FastLanguageModel
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_lora")

MODEL_NAME = "kenpath/svara-tts-v1"
DATASET_DIR = Path("/mnt/siet_llm_data/svara_training/processed_dataset")
RUNS_DIR = Path("/mnt/siet_llm_data/svara_training/runs")

MAX_SEQ_LENGTH = 1536
PAD_TOKEN_ID = 128263  # tts_engine.utils.PAD_TOKEN, <custom_token_7>
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class PretokenizedCollator:
    """Pads pre-built input_ids/labels pairs. No tokenizer call here - the
    format was already finalized by prepare_dataset.py."""

    def __init__(self, pad_token_id: int = PAD_TOKEN_ID):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_model():
    log.info("Loading base model %s (bf16, no quantization)", MODEL_NAME)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="Stage A: 300 rows, 1 epoch, batch size 2")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or ("smoke_test" if args.smoke_test else "kn_lora_full")
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading processed dataset from %s", DATASET_DIR)
    ds = load_from_disk(str(DATASET_DIR))
    train_ds, eval_ds = ds["train"], ds["validation"]

    if args.smoke_test:
        train_ds = train_ds.select(range(min(300, len(train_ds))))
        eval_ds = eval_ds.select(range(min(30, len(eval_ds))))
        log.info("Smoke test: %d train rows, %d eval rows", len(train_ds), len(eval_ds))
    else:
        log.info("Full run: %d train rows, %d eval rows", len(train_ds), len(eval_ds))

    model, tokenizer = load_model()
    collator = PretokenizedCollator(pad_token_id=PAD_TOKEN_ID)

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2 if args.smoke_test else 4,
        gradient_accumulation_steps=4,
        num_train_epochs=1 if args.smoke_test else 3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    log.info("Starting training: run_name=%s smoke_test=%s", run_name, args.smoke_test)
    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Saved LoRA adapter + tokenizer to %s", adapter_dir)


if __name__ == "__main__":
    main()
