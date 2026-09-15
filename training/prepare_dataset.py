"""
Build the LoRA training set for Kannada from the AI4Bharat/IndicTTS Kannada
parquet dataset, in the exact token format `tts_engine`'s own inference code
expects (see training/STATUS.md and the plan this was built from).

Reuses the repo's own prompt-building and SNAC-encoding code (svara_prompt,
create_speaker_id, SNACCodec.encode_audio) rather than reimplementing them, so
training targets stay bit-for-bit consistent with what production sends to
vLLM.

Usage:
    source training/setup_env.sh
    python training/prepare_dataset.py
"""
from __future__ import annotations

import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tts_engine.snac_codec import SNACCodec  # noqa: E402
from tts_engine.utils import create_speaker_id, load_audio_from_bytes, svara_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_dataset")

MODEL_NAME = "kenpath/svara-tts-v1"
DATASET_DIR = Path("/mnt/siet_llm_data/IndicTTS_Kannada/data")
OUTPUT_DIR = Path("/mnt/siet_llm_data/svara_training/processed_dataset")

# The dataset card's "avg ~2.7s" undersold the real per-utterance durations
# (measured: mean=8.29s, p95=16.11s, max=55.17s) - 15.0s would drop 6.76% of
# rows (over the plan's 5% budget); 17.0s keeps the drop rate under 4%.
MAX_DURATION_SEC = 17.0
MAX_SEQ_LEN = 1536
VAL_FRACTION = 0.03
SEED = 42

# Vocab bounds for audio tokens: 128266 is the lowest offset code id
# (128266 + code + pos*4096 with code=0, pos=0); AUDIO (156939) is the first
# id above the audio-code range, so every valid audio token must be strictly
# below it.
AUDIO_TOKEN_MIN = 128266
AUDIO_TOKEN_MAX = 156939  # exclusive

# 2 of the 6 source shards have transcripts missing terminal punctuation on
# ~78-82% of rows, even though the recorded speaker still pauses at the end
# of every utterance - that inconsistency taught the first fine-tune a noisy
# period->pause mapping (confirmed by listening test: lost sentence-final
# pausing, garbled words). Normalize so every example ends the same way.
SENTENCE_END_CHARS = (".", "?", "!", "।")


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text.endswith(SENTENCE_END_CHARS):
        text = text + "."
    return text


def find_shards() -> list[str]:
    shards = sorted(str(p) for p in DATASET_DIR.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {DATASET_DIR}")
    log.info("Found %d parquet shards", len(shards))
    return shards


def tokenizer_roundtrip_check(tokenizer) -> None:
    """Verification step 2 from the plan: fail fast if the token-format
    assumption is wrong, before touching any of the ~9.7k rows."""
    speaker_id = create_speaker_id("kn", "female")
    assert speaker_id == "Kannada (Female)", f"unexpected speaker id: {speaker_id!r}"

    prompt_text = svara_prompt("ನಮಸ್ಕಾರ", speaker_id)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids

    decoded = tokenizer.decode(prompt_ids)
    for marker in ("<|audio|>", "ನಮಸ್ಕಾರ", "<custom_token_5>"):
        assert marker in decoded, f"round-trip lost {marker!r}: {decoded!r}"

    # <custom_token_5> is START_OF_AI (128261) - must be the last id, since
    # svara_prompt ends the human turn and opens the AI turn there.
    start_of_ai_id = tokenizer.convert_tokens_to_ids("<custom_token_5>")
    assert start_of_ai_id == 128261, f"unexpected START_OF_AI id: {start_of_ai_id}"
    assert prompt_ids[-1] == start_of_ai_id, (
        f"expected prompt to end with START_OF_AI ({start_of_ai_id}), got {prompt_ids[-1]}"
    )
    log.info("Tokenizer/format round-trip check passed. Example prompt token count: %d", len(prompt_ids))


def duration_histogram(shards: list[str]) -> None:
    """Read only WAV headers (soundfile.info) across every row, cheap, so we
    pick MAX_DURATION_SEC/MAX_SEQ_LEN from real data rather than a guess."""
    durations = []
    ds = load_dataset("parquet", data_files=shards, split="train", columns=["audio"])
    # The "audio" column is auto-typed as a HF Audio feature, which would try
    # to fully decode each file (via torchcodec, not installed) just to
    # iterate rows. We only want the raw bytes here - disable that decoding.
    ds = ds.cast_column("audio", Audio(decode=False))
    for row in ds:
        info = sf.info(BytesIO(row["audio"]["bytes"]))
        durations.append(info.frames / info.samplerate)
    durations = np.array(durations)
    log.info(
        "Duration stats over %d rows: mean=%.2fs p50=%.2fs p95=%.2fs p99=%.2fs max=%.2fs",
        len(durations), durations.mean(),
        np.percentile(durations, 50), np.percentile(durations, 95),
        np.percentile(durations, 99), durations.max(),
    )
    drop_frac = (durations > MAX_DURATION_SEC).mean()
    log.info("Rows above MAX_DURATION_SEC=%.1fs: %.2f%%", MAX_DURATION_SEC, drop_frac * 100)
    if drop_frac > 0.05:
        raise RuntimeError(
            f"Duration cutoff would drop {drop_frac:.1%} of rows (>5%) - "
            "inspect the histogram above and adjust MAX_DURATION_SEC before proceeding."
        )


def build_example(row, tokenizer, codec: SNACCodec, speaker_cache: dict):
    duration_sec = None
    audio_bytes = row["audio"]["bytes"]
    info = sf.info(BytesIO(audio_bytes))
    duration_sec = info.frames / info.samplerate
    if duration_sec > MAX_DURATION_SEC:
        return None

    gender = "female" if row["gender"] == 0 else "male"
    speaker_id = speaker_cache.setdefault(gender, create_speaker_id("kn", gender))

    waveform, sr = load_audio_from_bytes(audio_bytes, device=codec.device)
    audio_tokens = codec.encode_audio(waveform, input_sample_rate=sr, add_token_offsets=True)

    if not audio_tokens:
        log.warning("Empty audio_tokens for %s - skipping", row["audio"]["path"])
        return None
    lo, hi = min(audio_tokens), max(audio_tokens)
    if lo < AUDIO_TOKEN_MIN or hi >= AUDIO_TOKEN_MAX:
        log.warning(
            "Audio token range [%d, %d] outside expected [%d, %d) for %s - skipping",
            lo, hi, AUDIO_TOKEN_MIN, AUDIO_TOKEN_MAX, row["audio"]["path"],
        )
        return None

    text = normalize_text(row["text"])
    prompt_text = svara_prompt(text, speaker_id)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids

    start_of_speech = tokenizer.convert_tokens_to_ids("<custom_token_1>")
    end_of_speech = tokenizer.convert_tokens_to_ids("<custom_token_2>")
    end_of_ai = tokenizer.convert_tokens_to_ids("<custom_token_6>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    target_ids = [start_of_speech] + audio_tokens + [end_of_speech, end_of_ai, eot_id]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > MAX_SEQ_LEN:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "gender": gender,
        "duration_sec": duration_sec,
        "source_path": row["audio"]["path"],
    }


def main():
    shards = find_shards()

    log.info("Loading tokenizer for %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_roundtrip_check(tokenizer)

    duration_histogram(shards)

    log.info("Loading full dataset for encoding")
    raw_ds = load_dataset("parquet", data_files=shards, split="train")
    raw_ds = raw_ds.cast_column("audio", Audio(decode=False))
    log.info("Total rows: %d", len(raw_ds))

    codec = SNACCodec()
    speaker_cache: dict = {}

    examples = []
    dropped = 0
    for i, row in enumerate(raw_ds):
        try:
            ex = build_example(row, tokenizer, codec, speaker_cache)
        except Exception:
            log.exception("Failed on row %d (%s) - skipping", i, row["audio"].get("path"))
            ex = None
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
        if (i + 1) % 500 == 0:
            log.info("Processed %d/%d rows (%d kept, %d dropped)", i + 1, len(raw_ds), len(examples), dropped)

    log.info("Done: %d kept, %d dropped (%.2f%% drop rate)", len(examples), dropped, 100 * dropped / len(raw_ds))

    lengths = np.array([len(e["input_ids"]) for e in examples])
    log.info(
        "input_ids length: mean=%.0f p50=%.0f p95=%.0f p99=%.0f max=%d",
        lengths.mean(), np.percentile(lengths, 50), np.percentile(lengths, 95),
        np.percentile(lengths, 99), lengths.max(),
    )

    rng = np.random.default_rng(SEED)
    by_gender: dict[str, list[int]] = {}
    for idx, ex in enumerate(examples):
        by_gender.setdefault(ex["gender"], []).append(idx)

    val_idx = set()
    for gender, idxs in by_gender.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * VAL_FRACTION))
        val_idx.update(idxs[:n_val].tolist())
        log.info("gender=%s: %d total, %d held out for validation", gender, len(idxs), n_val)

    train_examples = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val_examples = [ex for i, ex in enumerate(examples) if i in val_idx]

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "validation": Dataset.from_list(val_examples),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(OUTPUT_DIR))

    manifest = {
        "model_name": MODEL_NAME,
        "source_shards": shards,
        "max_duration_sec": MAX_DURATION_SEC,
        "max_seq_len": MAX_SEQ_LEN,
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "total_rows_in_source": len(raw_ds),
        "kept": len(examples),
        "dropped": dropped,
        "train_rows": len(train_examples),
        "validation_rows": len(val_examples),
        "input_ids_length_stats": {
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved processed dataset + manifest to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
