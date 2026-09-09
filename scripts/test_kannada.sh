#!/bin/bash
# Smoke-test the Kannada-only test stack (docker-compose.kannada.yml).
#
#   ./scripts/test_kannada.sh                  # default sentence, kn_female
#   ./scripts/test_kannada.sh "ನಮಸ್ಕಾರ" kn_male
#
# Writes speech.wav in the current directory.

set -euo pipefail

HOST=${HOST:-http://localhost:8030}
TEXT=${1:-"ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಧ್ವನಿ ಪರೀಕ್ಷೆ."}
VOICE=${2:-kn_female}
OUT=${OUT:-speech.wav}

echo "== health =="
curl -fsS "$HOST/health" && echo

echo
echo "== voices served (should be Kannada only) =="
curl -fsS "$HOST/v1/voices" \
  | python3 -c 'import json,sys; [print("  ", v["voice_id"], "->", v["name"]) for v in json.load(sys.stdin)["voices"]]'

echo
echo "== the lock: hi_male must be rejected with 400 =="
code=$(curl -sS -o /tmp/kn_reject.json -w '%{http_code}' \
  -X POST "$HOST/tts" -H 'Content-Type: application/json' \
  -d '{"text":"test","voice":"hi_male"}')
echo "  HTTP $code -> $(cat /tmp/kn_reject.json)"
if [ "$code" != "400" ]; then
  echo "  ✗ FAIL: expected 400, the language lock is not working"
  exit 1
fi
echo "  ✓ rejected as expected"

echo
echo "== synthesize ($VOICE) =="
echo "  text: $TEXT"
curl -fsS -X POST "$HOST/tts" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "voice": sys.argv[2], "format": "wav"}))' "$TEXT" "$VOICE")" \
  -D /tmp/kn_headers.txt \
  --output "$OUT"

grep -i '^x-speaker-id:' /tmp/kn_headers.txt | sed 's/^/  /' || true
echo "  wrote $OUT ($(stat -c%s "$OUT") bytes)"
python3 - "$OUT" <<'PY'
import sys, wave
with wave.open(sys.argv[1]) as w:
    dur = w.getnframes() / w.getframerate()
    print(f"  {w.getframerate()} Hz, {w.getnchannels()} ch, "
          f"{w.getsampwidth() * 8}-bit, {dur:.2f}s")
    if dur < 0.1:
        sys.exit("  ✗ FAIL: audio is suspiciously short")
print("  ✓ playable audio produced")
PY
