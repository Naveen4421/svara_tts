# Kannada LoRA Fine-Tuning — Status

## Goal

Improve Kannada speech quality/accent for the `svara-tts-kn-test` deployment by
LoRA fine-tuning the base `kenpath/svara-tts-v1` checkpoint on a real Kannada TTS
dataset, rather than just prompting the stock multilingual model with a Kannada
speaker id (which is all the current deployment does).

Dataset: AI4Bharat/IndicTTS Kannada, at `/mnt/siet_llm_data/IndicTTS_Kannada`
(~7.35h studio-quality audio, 2 native speakers — 1 male, 1 female — with
transcripts, 9,694 utterances, avg ~2.7s each).

## Why LoRA / Unsloth

The base model's own `config.json` shows it was itself produced by an Unsloth
training run — Unsloth LoRA is the natural fit: small trainable adapter, much
lower memory/time than full fine-tuning, known-compatible with this exact
architecture (Llama-3.2-3B-shaped, `vocab_size=156940`).

## What's been decided (see full plan for details/citations)

- **Token format** for training examples, reverse-engineered directly from the
  repo's own inference code (`tts_engine/utils.py`, `tts_engine/transports.py`):
  ```
  prompt_ids = tokenize(svara_prompt(text, speaker_id))
  target_ids = [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH, END_OF_AI, EOT_ID]
  input_ids  = prompt_ids + target_ids
  labels     = [-100]*len(prompt_ids) + target_ids   # loss only on the audio span
  ```
  `audio_tokens` come from the repo's existing `SNACCodec.encode_audio(...,
  add_token_offsets=True)` — reused as-is, not reimplemented. `gender` 0/1 in the
  dataset maps directly onto the existing `kn_female`/`kn_male` voice ids already
  used by this deployment, so no API changes are needed to serve the result later.
- **No 4-bit/8-bit weight quantization** — full bf16 LoRA (GPU has plenty of
  headroom for a 3B model; avoids quantization noise on audio-token fidelity).
- **Staged execution**: a small smoke-test run (300 rows, 1 epoch) before
  committing to the full ~9.4k-row, 3-epoch run, and a human-listening comparison
  (base vs. fine-tuned, paired WAVs) as the final go/no-go gate before ever
  touching the live `docker-compose.kannada.yml`/redeploying.
- Full plan with all hyperparameters and file-by-file design:
  `/home/sietllm/.claude/plans/jiggly-plotting-snail.md`.

## Environment reality (discovered during setup, not assumed)

- Root disk (`/`) was at **100% usage, 5.3GB free** when this started. Freed to
  **~178GB free** via `docker image prune -a` (153.6GB reclaimed) + `docker volume
  prune` (10.45GB reclaimed) — both scoped to images/volumes not attached to any
  currently-running container. Verified afterward: no other host service (ai_
  attendance, ai_interviewer, openproject, etc.) broke; `svara-tts-kannada:test`
  image intact.
- `/mnt/siet_llm_data` is a CIFS mount (`nounix`) that **cannot host an installed
  Python environment** — confirmed empirically, it rejects both symlink creation
  and pip's atomic rename-with-permission-set pattern, even for one small file.
  It works fine for plain data (HF cache, dataset, checkpoints, eval outputs).
  So: **venv lives at `/home/sietllm/svara_training_venv`** (local disk); all data
  artifacts live under `/mnt/siet_llm_data/svara_training/`.

## Current status: environment setup complete and verified

- `training/setup_env.sh --install` ran: torch 2.10.0 (cu128) + unsloth +
  transformers installed successfully.
- Post-install sanity check (`import torch, unsloth`) **failed** at first:
  `ModuleNotFoundError: No module named 'PIL'` (`unsloth_zoo` needs Pillow,
  not pulled in via the `--no-deps` unsloth install), then after fixing that,
  surfaced a deeper problem: unpinned `pip install transformers/datasets/trl`
  had resolved versions (`transformers 5.17.0`, `datasets 5.0.1`, `trl 1.13.0`)
  outside what `unsloth_zoo` actually supports, and several of unsloth_zoo's
  own runtime deps (`hf_transfer`, `msgspec`, `protobuf`, `sentencepiece`,
  `torchao`, `tyro`, `wheel`, `cut_cross_entropy`) were never installed at all
  under `--no-deps`. Fixed by pinning `transformers==4.55.4` (also matches the
  base model's own `config.json`), `datasets>=3.4.1,<4.4.0`, `trl>=0.18.2,
  <=0.24.0`, and adding the missing deps to `requirements-train.txt`.
  Also tried installing `diffusers`/`torchvision`/`xformers` (also listed in
  unsloth's metadata) — reverted: `xformers` silently upgraded torch to an
  unpinned 2.14.0 build that broke `torchaudio`/`unsloth_zoo`'s own
  `torch<2.13` pin, and `diffusers>=0.40` needs `huggingface-hub>=1.23` which
  conflicts with the `<1.0` that `transformers==4.55.4` needs. None of the
  three are required for `import unsloth` or text-only LoRA training (they're
  vision-model / optional-speedup extras) — deliberately left out.
- **Verified working**: `torch 2.10.0+cu128`, CUDA available, GPU visible
  (`NVIDIA RTX 6000 Ada Generation`), plus `transformers 4.55.4`,
  `datasets 4.3.0`, `trl 0.22.2`, `peft 0.20.0`, `snac`, `soundfile` all
  import cleanly. (One benign warning: "Unsloth fused-forward install
  skipped: requires transformers >= 4.56.0" — a perf-only optimization,
  not a functional blocker; we're capped at 4.55.4 by unsloth_zoo's own
  upper bound.)
- Also hit and fixed a broken CUDA/cuDNN setup: leftover `nvidia-cudnn-cu13`/
  `cuda-toolkit`/other cu13 packages from the earlier reverted xformers
  install shared file paths with their cu12 counterparts, so uninstalling
  them deleted shared libraries torch 2.10.0+cu128 still needed (`libcudnn.
  so.9`, `libcusparseLt.so.0`) — surfaced as `CUDNN_STATUS_NOT_INITIALIZED`
  on the very first real GPU op (SNAC's resample). Fixed by force-reinstalling
  every cu12 nvidia package torch depends on. Verified with a bare
  `torch.nn.functional.conv1d` call on CUDA before trusting the pipeline again.
- **Dataset prep done**: `training/prepare_dataset.py` written and run to
  completion. Real duration distribution didn't match the dataset card's "avg
  ~2.7s" (measured: mean 8.29s, p95 16.11s, max 55.17s) — the plan's default
  `MAX_DURATION_SEC=15.0` would have dropped 6.76% of rows (over its own 5%
  budget), so it's set to 17.0s here instead (keeps duration-only drops under
  4%). Result: **9,058 kept / 636 dropped (6.56% total, duration + the
  1536-token sequence-length cap combined)** → 8,786 train / 272 validation
  rows, saved to `/mnt/siet_llm_data/svara_training/processed_dataset/`.
  Tokenizer/format round-trip check (verification step 2) passed - a 5-row
  spot check confirmed prompts and targets match the plan's exact format
  (`128000, 128259, 156939, ... 128258, 128262, 128009`).
- `training/train_lora.py`, `training/merge_and_export.py`,
  `training/evaluate_samples.py` are now written (Unsloth LoRA r=32/α=64,
  plain `Trainer` with a custom pre-tokenized collator; merge via
  `PeftModel.merge_and_unload()`; evaluation reuses `SvaraMapper`/
  `SNACCodec.decode_window` directly so decoding matches production).
- **Stage A smoke test done**: 300 rows, 1 epoch, batch size 2 (38 steps).
  Loss dropped 3.28 → 3.19 over the run, no NaNs/crashes/OOM. LoRA adapter
  saved to `runs/smoke_test/lora_adapter/` (194MB, 48.6M trainable params,
  1.45% of the 3.35B total - matches the r=32 config).
- **Terminator-learning check passed** (verification step 6): generated a
  validation sentence per gender with the smoke-test adapter via plain
  `.generate()`. Both runs stopped naturally at END_OF_SPEECH/END_OF_AI
  (653 and 961 new tokens respectively, well under the 2048 cap) instead of
  running to max length - confirms the target-token format is sound.
- Ran `evaluate_samples.py` properly (5 validation + 3 novel sentences ×
  2 genders = 10 samples) with the smoke-test adapter, saved WAVs + manifest
  to `eval_outputs/smoke_test_check/`. Sent 2 samples to the user directly -
  expected to sound rough (300 rows / 1 epoch is a pipeline check, not real
  training), but they render as real, EOS-terminated Kannada speech, not noise.
- **Stage B full run done** (user approved). 1,650 steps, 3 epochs over 8,786
  rows, ~2h13m runtime. Train loss 3.6 → ~3.3-3.35 by the end; eval_loss=3.45
  at the one eval checkpoint that ran (close to train loss, no overfitting).
  Adapter saved to `runs/kn_lora_full/lora_adapter/`.
- **Merged**: `merge_and_export.py --run_name kn_lora_full` →
  `checkpoints/svara-tts-v1-kn-lora-2026-09-11/` (6.6GB standalone checkpoint).
- **Evaluated base vs. fine-tuned**: same 16 sentences (5 validation/gender +
  3 novel/gender), same seed/decoding params, via `evaluate_samples.py`.
  Both hit EOS cleanly on all 16/16 samples (base avg 867 tokens, fine-tuned
  avg 839 tokens) - fine-tuning didn't break termination behavior. Saved to
  `eval_outputs/base_vs_kn_lora_full/` and `eval_outputs/finetuned_kn_lora_full/`.
  Sent the user a paired female + male sample (same novel sentence, base vs.
  fine-tuned) for the listening comparison.
- **Currently waiting on the human-listening go/no-go verdict** (verification
  step 9) before touching `docker-compose.kannada.yml` or the live
  `svara-tts-kn-test` service in any way.

## Not yet started

- The human-listening go/no-go decision (waiting on the user).
- Updating `docker-compose.kannada.yml` / redeploying `svara-tts-kn-test` -
  explicitly gated on a positive listening verdict, not started, and the live
  service is untouched and still serves the stock base model.
