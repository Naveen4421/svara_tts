# Kannada Fine-Tuning Report

## Is this live in production?

**No.** The live `svara-tts-kn-test` service is untouched and is currently
serving the original, un-tuned `kenpath/svara-tts-v1` model. There have been
two fine-tuning attempts so far — the first was deployed, tested, and rolled
back after you reported problems; the second (fixed) attempt is trained and
evaluated but not yet deployed, waiting on your listening verdict.

## The story so far, in order

1. **Attempt 1 — trained and deployed.** Trained on the Kannada dataset you
   provided (~8,800 usable recordings, 3 passes, ~2h13m). Deployed it live so
   you could test it directly.
2. **You reported it was worse than the original**: garbled words, and no
   proper pause after sentences ending in a period. **Rolled back
   immediately** to the original model.
3. **Root cause found.** Checked the training data itself: 2 of the 6 data
   files (~27% of all 9,694 recordings) had transcripts with almost no
   ending punctuation (22% and 18% of sentences ended with a period, vs.
   96-100% in the other 4 files) — even though the speaker still paused
   naturally at the end of every recording. That mismatch taught the model
   an inconsistent rule for "period → pause," which lines up exactly with
   what you heard.
4. **Fixed and retrained (Attempt 2).** Added a step that makes every
   training sentence end with proper punctuation before training, regenerated
   the dataset, and ran the full training again from scratch (~2h9m).
5. **Merged and re-evaluated**, same 16 test sentences as before, same
   settings, for a fair before/after comparison. Sent you 3 samples from this
   version, including one specifically chosen to test mid-sentence pausing.

## Numbers: before vs. after, both attempts

### Training numbers (measure whether the model learned anything at all)

| Metric | Attempt 1 | Attempt 2 (fixed) |
|---|---|---|
| Recordings used for training | 8,786 | 8,786 |
| Recordings held out to check progress | 272 | 272 |
| Training passes (epochs) | 3 | 3 |
| Total training steps | 1,650 | 1,650 |
| Training time | 2h 13m | 2h 9m |
| Loss at the start | 3.72 | 3.73 |
| Loss at the end | ~3.30–3.35 | ~3.31–3.36 |
| Loss on held-out data | 3.45 | (not separately re-checked; training curve matched attempt 1 closely) |

"Loss" measures how surprised the model is by the correct answer — lower is
better. Both attempts learned equally well by this measure, which makes
sense: the punctuation fix changes *what* the model learned to associate
with pauses, not *how much* it learned overall. **This is exactly why the
loss numbers looked fine even in attempt 1, despite it sounding worse** — loss
doesn't know the difference between "learned a good pattern" and "learned a
confusing pattern," only whether the target was predictable at all.

### Generation numbers (same 16 test sentences, both attempts)

| Metric | Original model | Attempt 1 | Attempt 2 (fixed) |
|---|---|---|---|
| Correctly stopped talking on its own | 16 / 16 | 16 / 16 | 16 / 16 |
| Average length of generated audio (tokens) | 867 | 839 | 863 |

All three reliably know when to stop speaking. Attempt 2's output length is
much closer to the original (863 vs. 867) than attempt 1 was (839) — a mild
signal that it drifted less from the original model's behavior, consistent
with the training data being more internally consistent this time. This is
still not a quality score.

### What these numbers can't tell you

None of the above measures "does this sound like natural Kannada" or "is the
pausing correct" — that was exactly the problem with attempt 1: it looked
fine on paper and sounded wrong. The only real test is you listening to the
paired samples, which is what's pending right now for attempt 2.

## Where things are

| What | Path |
|---|---|
| Attempt 1 checkpoint (rolled back, not recommended) | `/mnt/siet_llm_data/svara_training/checkpoints/svara-tts-v1-kn-lora-2026-09-11/` |
| **Attempt 2 checkpoint (current candidate)** | `/mnt/siet_llm_data/svara_training/checkpoints/svara-tts-v1-kn-lora-v2-2026-09-15/` |
| Attempt 2 comparison samples | `/mnt/siet_llm_data/svara_training/eval_outputs/finetuned_kn_lora_full_v2/` |
| Original-model samples (for comparison) | `/mnt/siet_llm_data/svara_training/eval_outputs/base_vs_kn_lora_full/` |
| Full technical status log | `training/STATUS.md` |
| Training/data-prep scripts (includes the punctuation fix) | `training/*.py` |

## What happens next

Waiting on your verdict on the 3 attempt-2 samples sent to you. If they sound
right (proper pauses, clean pronunciation), I'll deploy attempt 2 the same
way as before. If something's still off, tell me specifically what, and
we'll dig further before a third attempt — no changes to the live service
either way until you approve.
