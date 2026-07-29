## Context

Accurate interpretation can emit draft translations before final ASR and final translation complete. If final translation is delayed, dropped, or skipped during shutdown, the browser can keep a gray draft or a "translating" row after the source is already final.

The stop flow currently sends `END_OF_AUDIO` and closes the WebSocket immediately. In faster-whisper accurate mode this can miss the final short ASR tail because normal realtime processing waits for `min_transcription_chunk_seconds`, and translation cleanup can set the translation client exit flag before queued final translations are drained.

Recent Chinese-to-English accurate-mode testing shows contextual output can be polluted by previous turns for short Chinese segments. The same meeting also showed repeated `utterance_id` values across multiple source time ranges, so terminal-state tracking cannot rely on utterance ID alone.

Project constraints: do not change FunASR, do not add model instances, do not increase GPU residency, do not add dependencies, and keep meeting log assembly on the backend.

## Goals / Non-Goals

**Goals:**

- Complete faster-whisper ASR tail processing after `END_OF_AUDIO` using the existing ASR thread.
- Resolve every completed source segment that enters translation processing to a successful final translation or explicit failed placeholder.
- Stop stale draft/translation-pending rows by using a 12-second frontend timeout placeholder.
- Finish user-initiated stop through `ASR finalization -> translation drain -> meeting log finish -> SESSION_FINALIZED -> close`.
- Use a 15-second backend budget across ASR finalization and translation drain, and a 20-second browser wait for `SESSION_FINALIZED`.
- Keep Chinese-to-English readability context for normal segments, but verify high-risk contextual output with one no-context direct translation.
- Preserve existing hard abnormal expansion rejection, glossary/hotword restoration, and `R&D` HTML entity normalization.

**Non-Goals:**

- Do not modify FunASR finalization.
- Do not tune model parameters, load additional models, change device defaults, or add GPU-resident caches.
- Do not correct ASR recognition mistakes, hotword bias, active audio compression, VAD behavior, or segmentation policy beyond finalizing the last valid faster-whisper segment.
- Do not add a frontend test framework.
- Do not run service restart or real browser/audio联调 unless explicitly authorized.

## Decisions

1. Faster-whisper finalization runs inside the existing ASR thread.

   `END_OF_AUDIO` will request finalization and wake the speech-to-text loop. The loop performs at most one final transcription pass for remaining audio, even if it is shorter than the realtime minimum, and reports completion. This avoids concurrent model access.

2. The last ASR segment can be force-completed only through existing safety checks.

   The final pass may mark the last segment completed and enqueue it for translation only after existing silence, hallucination, mixed-noise, RMS, and dedupe checks pass. If no remaining audio exists, finalization completes immediately. If finalization exceeds the budget, log `ASR_FINALIZE_TIMEOUT` and continue translation drain.

3. Translation terminal tracking uses composite segment identity.

   The internal key is `(utterance_id, start, end)` when `utterance_id` exists, otherwise `(start, end)`. Successful merged translations resolve pending segments by source IDs and time coverage. This handles repeated `utterance_id` values split into multiple rows.

4. Translation drain uses a FIFO sentinel and does not set `exit` first.

   Normal finalization sends a drain sentinel to the translation queue. The translation thread processes all earlier queued final segments, force-flushes translation and merge buffers, reports completion, then exits. Cleanup for unexpected disconnects keeps the existing interrupted behavior.

5. Timeout placeholders are terminal for the backend.

   If the 15-second backend finalization budget expires, unresolved known source segments become completed `翻译暂不可用` segments with `translation_warning: translation_drain_timeout`. Timed-out keys suppress duplicate late output.

6. Frontend timeout placeholders are in-memory only.

   The browser starts a 12-second timer for each completed source composite key. If no terminal translation arrives, it shows a non-pending `frontend_timeout` placeholder. A later backend completed segment matching by composite key or time coverage replaces it. Frontend placeholders do not claim backend log completion.

7. `SESSION_FINALIZED` is the only normal stop completion signal.

   The server sends `SESSION_FINALIZED` after meeting log finish with `session_id`, `session_status`, `asr_finalization`, `translation_drain`, and `translation_timeout_count`. The browser waits up to 20 seconds, then closes with a visible timeout status if the signal is missing.

8. Chinese-to-English direct verification is risk-triggered.

   Keep contextual Chinese-to-English translation by default. Run one no-context direct translation only if the contextual output is high risk: current source has at most 24 effective Chinese characters; output is at least 160 chars and over 4x source length; output repeats at least four consecutive English words from the previous translation; boundary extraction fails; existing output validation fails; or glossary placeholders cannot be restored. Use the direct result only if it passes existing validation. Otherwise emit a failed placeholder instead of suspected polluted context.

9. Existing hard guards stay active.

   Keep the current hard abnormal expansion rejection for Chinese-to-English outputs at least 240 chars and over 6x source length. Keep glossary/hotword restoration only for terms present in recognized source text, and keep final HTML entity normalization such as `R&D`.

## Risks / Trade-offs

- [Risk] Finalization can still time out under model stalls. -> Use explicit timeout statuses and placeholders instead of hanging indefinitely.
- [Risk] Direct verification adds latency for high-risk Chinese-to-English segments. -> Trigger only on high-risk outputs; normal segments remain single pass.
- [Risk] Late translations may arrive after timeout placeholders. -> Suppress backend duplicates for timed-out keys and replace frontend placeholders by composite key/time coverage.
- [Risk] Composite key matching can miss heavily merged segments. -> Resolve by both source IDs and time overlap coverage.
- [Risk] Keeping hard expansion guards can still produce failure placeholders. -> This is intentional for pathological output; logs make the reason visible.

## Migration Plan

Deploy as a normal code change. Existing meeting logs remain valid because failed placeholders use existing translated segment fields. No data migration is required.

After deployment and explicit restart authorization, validate in fresh faster-whisper accurate sessions that stop waits for `SESSION_FINALIZED`, tail translations appear in JSON/Markdown, stale drafts become terminal, and high-risk Chinese-to-English context pollution is avoided.
