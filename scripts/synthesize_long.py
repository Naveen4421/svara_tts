#!/usr/bin/env python3
"""Synthesize long text by splitting it into sentence-sized chunks.

The model is trained on single utterances, so a whole paragraph in one request
does not merely truncate -- past a certain length it stops tracking the text and
emits speech-like babble that sounds like another language. This script keeps
every request inside the range the model handles, then joins the pieces.

Chunks are requested as raw PCM (24kHz mono 16-bit), so concatenating them is
plain byte concatenation; the WAV header is written once at the end.

    ./scripts/synthesize_long.py --file passage.txt --out passage.wav
    ./scripts/synthesize_long.py "ನಮಸ್ಕಾರ ..." --voice kn_male

Voice timbre is sampled independently per request, so it can drift slightly
between chunks. Lower --temperature if that is noticeable.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import wave

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2

# Sentence-final punctuation, including the Devanagari danda used in some
# Kannada typesetting. The lookbehind keeps the mark attached to its sentence.
SENTENCE_END = re.compile(r'(?<=[.;?!।])\s+')

# Fallback split points for a single sentence that is already too long, tried
# in order from most to least natural.
SOFT_BREAKS = [re.compile(r'(?<=[,:])\s+'), re.compile(r'\s+')]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def _force_split(piece: str, max_chars: int) -> list[str]:
    """Break a single over-long sentence at the most natural point available."""
    for pattern in SOFT_BREAKS:
        parts = [p.strip() for p in pattern.split(piece) if p.strip()]
        if len(parts) > 1:
            return _pack(parts, max_chars, joiner=" ")
    # A single unbroken token longer than the limit: cut it bluntly.
    return [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)]


def _pack(pieces: list[str], max_chars: int, joiner: str = " ") -> list[str]:
    """Greedily combine pieces into chunks of at most max_chars."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_force_split(piece, max_chars))
            continue
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks small enough for the model to read reliably."""
    return _pack(split_sentences(text), max_chars)


def synthesize(host: str, text: str, voice: str, temperature: float | None,
               timeout: int) -> bytes:
    """Request one chunk as raw PCM."""
    payload: dict[str, object] = {"text": text, "voice": voice, "format": "pcm"}
    if temperature is not None:
        payload["temperature"] = temperature
    request = urllib.request.Request(
        f"{host.rstrip('/')}/tts",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            truncated = response.headers.get("X-Truncated") == "true"
            pcm = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"  ✗ HTTP {exc.code} from /tts: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"  ✗ cannot reach {host}: {exc.reason}") from exc
    if truncated:
        print("  ! server reports this chunk hit the token cap", file=sys.stderr)
    return pcm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize long text in sentence-sized chunks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("text", nargs="?", help="Text to speak")
    source.add_argument("--file", help="Read the text from this file instead")
    parser.add_argument("--host", default="http://localhost:8030")
    parser.add_argument("--voice", default="kn_female")
    parser.add_argument("--out", default="speech.wav")
    parser.add_argument("--max-chars", type=int, default=300,
                        help="Chunk size ceiling (default: 300)")
    parser.add_argument("--gap-ms", type=int, default=150,
                        help="Silence inserted between chunks (default: 150)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-request timeout in seconds")
    args = parser.parse_args()

    text = open(args.file, encoding="utf-8").read() if args.file else args.text
    text = text.strip()
    if not text:
        raise SystemExit("No text to speak")

    chunks = chunk_text(text, args.max_chars)
    print(f"{len(text)} characters -> {len(chunks)} chunks (max {args.max_chars})")

    gap = b"\x00" * (
        (args.gap_ms * SAMPLE_RATE // 1000) * BYTES_PER_SAMPLE
    )

    audio: list[bytes] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}/{len(chunks)}] {len(chunk):>4} chars ... ",
              end="", flush=True)
        pcm = synthesize(args.host, chunk, args.voice, args.temperature,
                         args.timeout)
        seconds = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
        rate = len(chunk) / seconds if seconds else float("inf")
        # Healthy synthesis runs near 11 characters of text per second of audio.
        # Both directions signal a problem: too little audio means the model
        # stopped following the text, too much means it rambled or repeated
        # (which is what a chunk cut mid-sentence tends to produce).
        if rate > 14:
            flag = "  <-- too short, model may have lost the text"
        elif rate < 8:
            flag = "  <-- too long, model may be rambling"
        else:
            flag = ""
        print(f"{seconds:5.2f}s ({rate:.1f} chars/s){flag}")
        if index > 1:
            audio.append(gap)
        audio.append(pcm)

    body = b"".join(audio)
    with wave.open(args.out, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(body)

    total = len(body) / BYTES_PER_SAMPLE / SAMPLE_RATE
    print(f"wrote {args.out} - {total:.2f}s, {len(body):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
