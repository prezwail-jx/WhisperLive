## 1. Mixed-Language ASR Stabilization

- [x] 1.1 Add high-accuracy mixed-interpretation language retry configuration to `ServeClientFasterWhisper` and wire it from `server.py` only for faster-whisper accurate mixed interpretation.
- [x] 1.2 Materialize faster-whisper automatic decode results so language, text, average log probability, no-speech evidence, and bounded text previews can be scored before segment emission.
- [x] 1.3 Implement suspicious language-switch detection using previous stable `zh` or `en` language, automatic language probabilities, candidate text-language consistency, and bounded score thresholds.
- [x] 1.4 Implement one previous-language retry for the normal faster-whisper path and for the batch `BatchRequest` path without adding model instances or persistent GPU resources.
- [x] 1.5 Select the ASR candidate before transcript append, translation queue enqueue, Admin updates, and meeting log persistence.
- [x] 1.6 Adjust `server.py` and translation source-language inference so trusted `zh` or `en` segment language metadata is preserved instead of being overwritten by character heuristics.
- [x] 1.7 Add bounded logs for switch evaluation, retry execution, selected candidate, and accepted real language switches.

## 2. Translation Incomplete Fragment Handling

- [x] 2.1 Extend English incomplete-ending phrase detection with narrow dangling connector phrases such as `and they`, `but they`, `and we`, `assuming you`, and `we need to`.
- [x] 2.2 Preserve the existing `translation_incomplete_max_wait_seconds` behavior so high-accuracy mode still waits at most 4 seconds.
- [x] 2.3 Ensure `incomplete_timeout` flushes continue to avoid `translation_warning` and only emit bounded diagnostic logs.
- [x] 2.4 Allow incomplete-timeout candidates to enter completeness recovery and low-confidence selection without treating timeout itself as a warning reason.

## 3. Translation Safety, Completeness, and Candidate Model

- [x] 3.1 Split translation validation into safety failures and completeness failures while preserving existing guards for source echo, residual CJK, hallucination phrases, repetition, empty output, and abnormal expansion.
- [x] 3.2 Add an internal translation candidate structure that tracks candidate text, language pair, stage, safety reason, completeness reason, and bounded coverage metrics.
- [x] 3.3 Implement bidirectional fact-anchor extraction for digits, percentages, currency, units, acronyms, quantity words, hotword terms, and translation glossary terms.
- [x] 3.4 Implement accepted fact-anchor equivalences such as `tons` to `吨`, `RMB` to `人民币`, `AUD` to `澳元`, and percentage wording to `%` or `百分之`.
- [x] 3.5 Lower the English-to-Chinese undertranslation word threshold from 16 to 12 while allowing fact-anchor validation to trigger below that threshold.
- [x] 3.6 Add Chinese-to-English completeness checks for source character coverage, fact anchors, multi-clause tail omission risk, and residual source-language output.
- [x] 3.7 Implement deterministic candidate ranking that prefers safe candidates, higher fact-anchor coverage, reasonable length ratios, and whole-segment translations when coverage is equivalent.

## 4. Staged Translation Recovery

- [x] 4.1 Refactor strict current-segment translation so it returns a validated candidate instead of immediately finalizing recoverable failures.
- [x] 4.2 Reuse existing readability context with boundary extraction as the first recovery stage when bounded same-direction context is available.
- [x] 4.3 Add a conservative relaxed NLLB generation profile with `num_beams=3`, `max_new_tokens=320`, and `length_penalty=1.1` for recovery candidates only.
- [x] 4.4 Extend NLLB single and batch translation calls to accept the generation profile while keeping default strict behavior unchanged.
- [x] 4.5 Add bidirectional chunked recovery with at most three chunks and require chunked candidates to pass safety checks and improve completeness evidence.
- [x] 4.6 Emit safe but incomplete best candidates with `translation_confidence: "low"` and without `translation_warning`.
- [x] 4.7 Emit target-language unavailable placeholders only when no safe candidate remains, retaining `translation_warning` for those final failures.
- [x] 4.8 Add bounded logs for context retry, relaxed retry, chunked retry, low-confidence output, and terminal unavailable output.

## 5. Segment Metadata, Merge, Logs, and Frontend

- [x] 5.1 Add optional `translation_confidence` to final translation segments and preserve it through WebSocket output.
- [x] 5.2 Preserve `translation_confidence: "low"` in merge-buffer output when any merged child segment is low confidence.
- [x] 5.3 Preserve `translation_confidence` in meeting JSON export without changing historical logs.
- [x] 5.4 Ensure frontend warning rendering remains driven by `translation_warning` and does not show `!` for `translation_confidence: "low"` alone.
- [x] 5.5 Localize backend unavailable placeholders to `翻译暂不可用` for Chinese targets and `Translation unavailable` for English targets.
- [x] 5.6 Localize frontend timeout fallback placeholders using the resolved translation target and update `web/index.html` cache-busting for `web/app.js`.

## 6. Tests

- [x] 6.1 Add faster-whisper unit tests for conditional language retry enablement, suspicious switch retry, real switch acceptance, one-retry limit, and batch retry compatibility.
- [x] 6.2 Add server tests proving high-accuracy mixed interpretation enables language retry and trusted segment languages are not overwritten.
- [x] 6.3 Add translation buffer tests for new dangling English endings, strict 4-second timeout behavior, and no warning for `incomplete_timeout`.
- [x] 6.4 Add translation tests for safety-versus-completeness classification and fact-anchor coverage in both `en -> zh` and `zh -> en`.
- [x] 6.5 Add staged recovery tests for strict success, context recovery, relaxed recovery, chunked recovery, low-confidence safe output, and unavailable final failure.
- [x] 6.6 Add meeting-log tests proving `translation_confidence: "low"` is persisted and merge output preserves low confidence.
- [x] 6.7 Add frontend or static checks for localized timeout placeholder behavior when Node is available in the deployment container.

## 7. Verification and Handoff

- [x] 7.1 Run container `py_compile` for `whisper_live/backend/faster_whisper_backend.py`, `whisper_live/backend/translation_backend.py`, and `whisper_live/server.py`.
- [x] 7.2 Run container unit tests for `tests.test_faster_whisper_backend`, `tests.test_translation_backend`, `tests.test_server_extended`, and `tests.test_meeting_logs`.
- [x] 7.3 Run `docker exec whisperlive-gpu0 node --check web/app.js` when Node exists in the deployment container; otherwise record the first environment failure and stop that check.
- [x] 7.4 Run `git diff --check`, inspect `git status --short`, and review relevant diffs.
- [ ] 7.5 After deployment reload is explicitly approved, retest the two provided meeting samples and verify mixed-language retry, fact-anchor recovery, low-confidence JSON metadata, and warning marker behavior.
