## 1. OpenSpec Alignment

- [x] 1.1 Revise `fix-translation-finalization` artifacts to scope implementation to faster-whisper, fixed 12/15/20-second timeouts, composite segment identity, `SESSION_FINALIZED`, and high-risk Chinese-to-English direct verification.

## 2. Faster-Whisper ASR Finalization

- [x] 2.1 Add finalization request/completion state to the faster-whisper client path without introducing concurrent model calls.
- [x] 2.2 On `END_OF_AUDIO`, run one final ASR pass for remaining faster-whisper audio even when it is below the normal realtime minimum chunk duration.
- [x] 2.3 Extend the final pass to force-complete the last valid segment after existing silence, hallucination, mixed-noise, RMS, and dedupe checks.
- [x] 2.4 Report ASR finalization completion or `ASR_FINALIZE_TIMEOUT` without changing unexpected-disconnect interrupted behavior.

## 3. Translation Terminal State And Drain

- [x] 3.1 Track pending final translation segments with a composite key `(utterance_id, start, end)`, falling back to time range when no utterance ID exists.
- [x] 3.2 Mark successful emitted translations as terminal by source IDs plus covered time range, including merged translation segments.
- [x] 3.3 Convert per-segment translation exceptions, output validation failures, and drain timeouts into completed `翻译暂不可用` segments with stable `translation_warning` values.
- [x] 3.4 Add a FIFO drain sentinel/finalize method that flushes translation and merge buffers before the translation thread exits, without setting `exit` first.
- [x] 3.5 Enforce one 15-second backend finalization budget covering ASR finalization plus translation drain, and suppress duplicate late output for timed-out source segments.
- [x] 3.6 Add bounded logs for pending, resolved, failed, ASR timeout, drain start, drain complete, drain timeout, and late-suppressed states.

## 4. Chinese-To-English Risk Verification

- [x] 4.1 Keep Chinese-to-English readability context enabled for normal segments.
- [x] 4.2 Run one no-context direct translation only when a Chinese-to-English contextual result is high risk: current source has at most 24 effective Chinese characters, output is at least 160 chars and over 4x source length, output repeats at least four consecutive English words from previous translation, boundary extraction fails, existing validation fails, or glossary placeholders are lost.
- [x] 4.3 Use the direct result only if it passes existing length, character, and glossary validation; otherwise emit a failed placeholder instead of using suspected polluted context.
- [x] 4.4 Preserve the existing hard abnormal expansion rejection for outputs at least 240 chars and over 6x source length.
- [x] 4.5 Preserve configured glossary/hotword targets for recognized source terms only, and keep final HTML entity normalization such as `R&D`.
- [x] 4.6 Add bounded logs for risk detection, direct verification, direct fallback success/failure, glossary restoration failure, and hard expansion rejection.

## 5. Server Finalization Protocol

- [x] 5.1 Update `whisper_live/server.py` so user-initiated `END_OF_AUDIO` follows `ASR finalization -> translation drain -> meeting log finish -> SESSION_FINALIZED -> close`.
- [x] 5.2 Send `SESSION_FINALIZED` with `session_id`, `session_status`, `asr_finalization`, `translation_drain`, and `translation_timeout_count` before closing.
- [x] 5.3 Preserve existing interrupted-session behavior for connection closes that are not user-initiated `END_OF_AUDIO`.

## 6. Frontend Finalization And Draft Lifecycle

- [x] 6.1 Change `stopCapture()` to stop microphone capture, send one `END_OF_AUDIO`, keep the WebSocket open, and show that final recognition/translation is completing.
- [x] 6.2 Close the WebSocket and mark the session finished only after `SESSION_FINALIZED`; after 20 seconds, show finish-timeout status, convert remaining pending rows to frontend placeholders, and close without pretending backend finalization succeeded.
- [x] 6.3 Track completed source rows by composite key and start a 12-second translation wait timer for each final source row.
- [x] 6.4 Replace stale draft/translation-pending rows with in-memory `frontend_timeout` placeholders, and allow later backend completed translations covering the same key or time range to replace those placeholders without duplicates.
- [x] 6.5 Clear finalization, pending-source, and timer state on new session, reconnect, clear-source, clear-translation, expected stop completion, and unexpected disconnect.

## 7. Tests And Verification

- [x] 7.1 Add tests for faster-whisper ASR finalization with no tail audio, successful tail completion, and timeout.
- [x] 7.2 Add `tests.test_translation_backend` coverage for success terminal state, failure placeholder emission, drain-before-exit behavior, drain timeout placeholders, duplicate late suppression, and repeated utterance IDs with different time ranges.
- [x] 7.3 Add `tests.test_translation_backend` coverage for Chinese-to-English short-source direct verification, abnormal expansion, previous-translation leakage, normal long contextual translation, glossary preservation, HTML entity normalization, and existing hard expansion rejection.
- [x] 7.4 Add `tests.test_server_extended` coverage for `END_OF_AUDIO` finalization ordering and `SESSION_FINALIZED` payload.
- [x] 7.5 Run deployment-container `py_compile` for changed Python files and targeted unittest modules.
- [x] 7.6 Run deployment-container `node --check web/app.js`.
- [x] 7.7 Run `git diff --check`, inspect `git status --short`, and review relevant diffs before delivery.
- [x] 7.8 Leave real audio/browser联调 and service restart as待部署机验证 unless explicitly authorized.
