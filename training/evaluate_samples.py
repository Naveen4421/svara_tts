"""
Quality-check a checkpoint (base or fine-tuned) by generating held-out and
novel Kannada sentences and saving WAVs. Bypasses vLLM/FastAPI entirely -
runs plain transformers .generate() so base and fine-tuned checkpoints can be
compared side by side without touching the live service.

Reuses tts_engine's own prompt-building and audio-decoding code (svara_prompt,
SvaraMapper, SNACCodec.decode_window) so the decode path is identical to
production - only the model weights and the generate() call differ.

Usage:
    source training/setup_env.sh
    python training/evaluate_samples.py --checkpoint base --run_name base_eval
    python training/evaluate_samples.py --checkpoint /mnt/siet_llm_data/svara_training/checkpoints/<name> --run_name finetuned_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tts_engine.mapper import SvaraMapper  # noqa: E402
from tts_engine.snac_codec import SNACCodec  # noqa: E402
from tts_engine.utils import create_speaker_id, svara_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_samples")

BASE_MODEL_NAME = "kenpath/svara-tts-v1"
DATASET_DIR = Path("/mnt/siet_llm_data/svara_training/processed_dataset")
EVAL_OUTPUTS_DIR = Path("/mnt/siet_llm_data/svara_training/eval_outputs")

# Production's decoding defaults (tts_engine/transports.py).
GEN_KWARGS = dict(temperature=0.75, top_p=0.9, repetition_penalty=1.1, do_sample=True)
MAX_NEW_TOKENS = 2048
# Production's real stop condition is END_OF_SPEECH/END_OF_AI (128258/128262),
# not EOT_ID - .generate() has no separate stop_token_ids, so all three are
# passed as eos_token_id, matching the plan's terminator-learning check.
EOS_TOKEN_IDS = [128258, 128262, 128009]
PAD_TOKEN_ID = 128263
START_OF_SPEECH_ID = 128257

NOVEL_SENTENCES = [
    "ಈ ದಿನ ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
    "ಕರ್ನಾಟಕದ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಬಹಳ ಶ್ರೀಮಂತವಾಗಿದೆ.",
    "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ನಿಧಾನವಾಗಿ ಮಾತನಾಡಿ.",
]


def load_model(checkpoint: str):
    if checkpoint == "base":
        log.info("Loading base model %s", BASE_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    elif (Path(checkpoint) / "adapter_config.json").exists():
        # Unmerged LoRA adapter dir (e.g. runs/<name>/lora_adapter) - load base
        # + adapter directly, skip the merge/export step. Useful for a quick
        # diagnostic check without writing out a full merged checkpoint.
        log.info("Loading base model %s + LoRA adapter from %s", BASE_MODEL_NAME, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        log.info("Loading merged checkpoint from %s", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


# svara_prompt's fixed shape: "...<audio> {speaker_id}: {text}<|eot_id|>..."
# - pull the sentence back out from between the speaker-id prefix and EOT.
_PROMPT_TEXT_RE = re.compile(r":\s(.*?)<\|eot_id\|>", re.DOTALL)


def pick_validation_sentences(n_per_gender: int = 5):
    ds = load_from_disk(str(DATASET_DIR))["validation"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    picked = {"female": [], "male": []}
    for row in ds:
        gender = row["gender"]
        if len(picked[gender]) >= n_per_gender:
            continue
        ids = row["input_ids"]
        prompt_ids = [i for i in ids if i < 128266]  # strip audio-range tokens
        decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        m = _PROMPT_TEXT_RE.search(decoded)
        if not m:
            log.warning("Could not recover text from decoded prompt: %r - skipping row", decoded)
            continue
        picked[gender].append((m.group(1), gender))
        if all(len(v) >= n_per_gender for v in picked.values()):
            break
    return picked


def generate_and_decode(model, tokenizer, codec: SNACCodec, text: str, speaker_id: str) -> bytes:
    prompt = svara_prompt(text, speaker_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=EOS_TOKEN_IDS,
            pad_token_id=PAD_TOKEN_ID,
            **GEN_KWARGS,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    hit_eos = bool(new_ids) and new_ids[-1] in EOS_TOKEN_IDS
    log.info(
        "Generated %d new tokens (hit_eos=%s, first ids=%s)",
        len(new_ids), hit_eos, new_ids[:5],
    )

    mapper = SvaraMapper()
    pcm = bytearray()
    for token_id in new_ids:
        raw_num = token_id - 128256  # <custom_token_N> numbering, see tts_engine.mapper
        win = mapper.feed_raw(raw_num)
        if win is not None:
            pcm.extend(codec.decode_window(win))
    return bytes(pcm), hit_eos, len(new_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="'base' or a path to a merged fine-tuned checkpoint")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--n_per_gender", type=int, default=5)
    args = parser.parse_args()

    out_dir = EVAL_OUTPUTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.checkpoint)
    codec = SNACCodec()

    sentences = []
    val_picks = pick_validation_sentences(args.n_per_gender)
    for gender, items in val_picks.items():
        speaker_id = create_speaker_id("kn", gender)
        for text, _ in items:
            sentences.append({"text": text, "gender": gender, "speaker_id": speaker_id, "source": "validation"})
    for text in NOVEL_SENTENCES:
        for gender in ("female", "male"):
            speaker_id = create_speaker_id("kn", gender)
            sentences.append({"text": text, "gender": gender, "speaker_id": speaker_id, "source": "novel"})

    manifest = {"checkpoint": args.checkpoint, "samples": []}
    for i, s in enumerate(sentences):
        log.info("[%d/%d] gender=%s source=%s: %s", i + 1, len(sentences), s["gender"], s["source"], s["text"][:60])
        pcm, hit_eos, n_tokens = generate_and_decode(model, tokenizer, codec, s["text"], s["speaker_id"])
        wav_path = out_dir / f"sample_{i:03d}_{s['gender']}_{s['source']}.wav"
        if pcm:
            import numpy as np
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
            sf.write(str(wav_path), audio, codec.sample_rate)
        manifest["samples"].append({
            **s,
            "wav_path": str(wav_path) if pcm else None,
            "hit_eos": hit_eos,
            "generated_tokens": n_tokens,
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Saved %d samples + manifest to %s", len(sentences), out_dir)


if __name__ == "__main__":
    main()
