# Kannada-only test deployment

A standalone stack that runs `kenpath/svara-tts-v1` locked to Kannada at the
application layer. Shares no network, volume, or container name with the main
`docker-compose.yml` stack, so both can coexist.

## Run it

```bash
docker compose -f docker-compose.kannada.yml up --build
./scripts/test_kannada.sh          # smoke test once healthy
```

Tear down (the `-v` also drops the model cache):

```bash
docker compose -f docker-compose.kannada.yml down
docker compose -f docker-compose.kannada.yml down -v
```

## Ports

The container runs **two** servers under supervisord. The app endpoints live on
the FastAPI server, not on vLLM:

| Host   | Container | Server  | Use                                          |
|--------|-----------|---------|----------------------------------------------|
| `8030` | `8080`    | FastAPI | `/health`, `/tts`, `/v1/*` — **this is the API** |
| `8040` | `8000`    | vLLM    | `/v1/models`, `/v1/completions` — debug only |

Host ports 8031-8035 are occupied by an unrelated `ai-interviewer` stack on
this machine, which is why vLLM is published on 8040.

Pointing `8030` at container port `8000` is the easy mistake: you reach raw
vLLM and every app route 404s.

## The language lock

`ALLOWED_VOICES` is a comma-separated allowlist of voice IDs, read at startup.
Unset or empty means all 38 v1 voices are served, which is the behaviour of the
main stack. The test stack sets:

```yaml
- ALLOWED_VOICES=kn_male,kn_female
- DEFAULT_VOICE=kn_female
```

It is enforced in three places:

- `POST /tts` — rejects anything outside the list with `400`
- `POST /v1/text-to-speech` — same check on the `voice` parameter
- `GET /v1/voices` — lists only allowed voices

Both spellings resolve, so you can use either the registry ID or the speaker ID
the model was trained on:

```
kn_female  ==  "Kannada (Female)"
kn_male    ==  "Kannada (Male)"
```

An unknown or invalid entry in `ALLOWED_VOICES` fails the server at startup
rather than silently serving nothing.

### Known gap: voice cloning bypasses the lock

Zero-shot cloning (`reference_audio`, `voice_clone_id`, `voice_clone_tokens` on
`/v1/text-to-speech`, and all of `/v1/voice-clone`) never selects a preset
voice, so the allowlist does not apply to those requests — the language comes
from the reference audio. `POST /tts` does not expose cloning at all, so it is
locked unconditionally. If this deployment should refuse cloning too, that
needs a separate switch.

## Requests

```bash
# WAV out (default)
curl -X POST http://localhost:8030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಧ್ವನಿ ಪರೀಕ್ಷೆ.", "voice": "kn_female"}' \
  --output speech.wav

# voice omitted -> DEFAULT_VOICE
curl -X POST http://localhost:8030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "ನಮಸ್ಕಾರ"}' --output speech.wav

# raw PCM16 (24kHz mono), matching /v1/text-to-speech
curl -X POST http://localhost:8030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "ನಮಸ್ಕಾರ", "format": "pcm"}' --output speech.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav

# rejected
curl -X POST http://localhost:8030/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "test", "voice": "hi_male"}'
# {"detail":"only ['kn_female', 'kn_male'] supported in this build"}
```

`/tts` is non-streaming and returns the complete clip. For streaming, use
`/v1/text-to-speech` with `stream=true`, which returns raw PCM chunks.

## Debugging

`restart: "no"` keeps Docker from respawning the container, and
`SUPERVISOR_AUTORESTART=false` stops supervisord from respawning vLLM or
uvicorn inside it — a crash stays crashed so you can read the traceback.

```bash
docker compose -f docker-compose.kannada.yml logs -f
docker exec svara-tts-kn-test tail -100 /var/log/supervisor/vllm.log
docker exec svara-tts-kn-test tail -100 /var/log/supervisor/fastapi.log
docker exec svara-tts-kn-test supervisorctl -c /etc/supervisor/conf.d/svara-tts.conf status
```

Note that with autorestart off, supervisord itself stays up after a child dies,
so the container keeps running while showing a `FATAL` process. Check
`supervisorctl status` before assuming the stack is healthy.

## GPU

`VLLM_GPU_MEMORY_UTILIZATION=0.25` is a fraction of **total** GPU memory, not of
what is free. On a 49GB card that reserves roughly 12GB, which leaves room
alongside other workloads. Raise it if vLLM reports insufficient KV cache
blocks; lower it if it collides with something else on the card.

`VLLM_MAX_MODEL_LEN=8192` bounds prompt + generated tokens together. Audio costs
about 82 tokens per second, so this fits roughly 90 seconds of speech. The server
subtracts the prompt length and clamps `max_tokens` to whatever remains; if
generation stops at that ceiling the response carries `X-Truncated: true`, since
a clipped clip is otherwise indistinguishable from a short one.

## Input length

`MAX_TTS_CHARS` (default 400) caps how much text `/tts` accepts in one request.
This is a **model** limit, not a context-window limit, and it is the reason long
passages fail in a way that looks like the wrong language.

The model is trained on single utterances. Past roughly 400 characters it stops
tracking the text and emits speech-like audio that is not Kannada — while still
returning `200`. Healthy synthesis runs about **11 characters of text per second
of audio**; a collapse shows up as far too little audio for the text:

| chars | chars/sec | |
|-------|-----------|--|
| 106   | 11.5      | ok |
| 224   | 11.0      | ok |
| 290   | 10.9      | ok |
| 380   | 11.0–11.7 | ok (3 runs) |
| 578   | 32.3      | collapsed |
| 832   | 44.7      | collapsed |
| 892   | 112.4     | collapsed |

**Cut chunks at sentence boundaries.** Where the text is split matters as much
as its length. The same passage cut mid-sentence makes the model ramble —
measured at 6.3–6.5 chars/sec, roughly 70% more audio than the text warrants,
reproducible across runs — while the sentence-boundary cut of the same length
gave a clean 11.0–11.7.

To read a long passage, split it and join the audio:

```bash
./scripts/synthesize_long.py --file passage.txt --out passage.wav
./scripts/synthesize_long.py "ನಮಸ್ಕಾರ ..." --voice kn_male --max-chars 300
```

It splits on sentence marks, packs chunks up to `--max-chars` (default 300),
requests each as PCM, and joins them with a short silence. It prints the
chars/sec of every chunk and flags any that look too short or too long, so a bad
chunk is visible rather than buried in the middle of the audio. Voice timbre is
sampled per request and can drift slightly between chunks; lower
`--temperature` if that is noticeable.

### Known gap: concurrent requests fail

Two simultaneous requests both return `500`, even though the vLLM engine
completes both generations successfully — the orchestrator holds per-request
state that is not safe to share. Drive the API one request at a time;
`synthesize_long.py` is sequential for this reason. Using the web UI while a
script is running will produce spurious failures.
