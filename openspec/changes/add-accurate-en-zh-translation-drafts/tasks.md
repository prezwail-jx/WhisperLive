## 1. Configuration And Propagation

- [x] 1.1 Inspect current ASR segment, translation queue, and WebSocket config flow in `whisper_live/backend/base.py`, `whisper_live/backend/translation_backend.py`, `whisper_live/server.py`, `run_server.py`, and `web/app.js` before editing.
- [x] 1.2 Add backend normalization for `translation_draft_enabled`, `translation_draft_interval_seconds`, `translation_draft_min_delta_chars`, and `translation_draft_max_source_chars` with safe defaults and bounds.
- [x] 1.3 Add backend normalization for `translation_readability_context_sentences` and `translation_readability_context_max_chars` with defaults of `2` and `220`, disabled when draft translation is not eligible, and bounded so old clients keep current behavior.
- [x] 1.4 Propagate normalized draft and readability context config from WebSocket options into the translation client without changing GPU, model, batch, precision, or device defaults.
- [x] 1.5 Update frontend connection config so only high-accuracy English-to-Chinese mode sends draft translation enabled with interval `1.2`, min delta `8`, max source chars `220`, readability context sentences `2`, and readability context max chars `220`; other modes send false or omit the fields.

## 2. Backend Draft Scheduler

- [x] 2.1 Add per-client draft state keyed by source `utterance_id`, including latest source text, revision, finalization state, last translated draft text, last inference time, pending flag, and in-flight flag.
- [x] 2.2 Capture eligible incomplete English ASR draft segments in high-accuracy English-to-Chinese sessions without enqueueing each update into the existing final translation FIFO.
- [x] 2.3 Implement latest-slot coalescing so newer draft text replaces pending older draft text for the same utterance.
- [x] 2.4 Enforce interval, min-delta, empty/punctuation, English-source, Chinese-target, and bounded source-length checks before draft inference.
- [x] 2.5 Trim long draft source text to the latest bounded suffix without cutting through an English word when possible.
- [x] 2.6 Ensure one draft inference per client at a time and prioritize final translation over not-yet-started drafts.
- [x] 2.7 Invalidate pending draft state when the final ASR segment arrives for the utterance.

## 3. Backend Draft Translation And Emission

- [x] 3.1 Implement draft translation inference by reusing existing translator/tokenizer/device/output-guard logic and the shared model inference lock.
- [x] 3.2 Ensure draft inference does not mutate final translation buffers, merge buffers, final dedup state, `last_translated_source_text`, warning state, or formal Admin counters.
- [x] 3.3 Recheck revision, finalization state, client liveness, and English-to-Chinese direction immediately before emitting a draft result.
- [x] 3.4 Emit valid draft results as WebSocket-only `translated_segments` entries with `completed: false`, stable `translation_id`, current `revision`, `utterance_id`, `source_utterance_ids`, source/target language metadata, model name, source text, and draft text.
- [x] 3.5 Drop invalid draft results silently without showing failure placeholder text or affecting final translation behavior.
- [x] 3.6 Add bounded diagnostic logs for scheduled, coalesced, emitted, finalized, and stale-dropped draft events without logging full long text.
- [x] 3.7 Clean up draft scheduler state when the client exits or the translation client is cleaned up.

## 4. Readability Context Translation

- [x] 4.1 Add per-client readability context history for the most recent successful final English-to-Chinese translation units, capped by `translation_readability_context_sentences`.
- [x] 4.2 Gate readability context strictly to eligible high-accuracy English-to-Chinese sessions; standard interpretation, conversation translation, transcription-only mode, Chinese-to-English, non-English source, non-Chinese target, and old clients must use current-sentence-only translation.
- [x] 4.3 Ensure only successful final translations update readability context; drafts, failed placeholders, warning outputs, source-echo fallbacks, and invalid translations must not enter history.
- [x] 4.4 Build contextual translation input for eligible draft and final translations as recent English context plus a protected boundary marker plus the current source sentence.
- [x] 4.5 Trim readability context from the oldest side first to `translation_readability_context_max_chars` without cutting through an English word when possible, and never shorten the current source sentence to fit history.
- [x] 4.6 Extract only the current-sentence Chinese translation from contextual output using the protected boundary marker or equivalent reliable delimiter handling.
- [x] 4.7 If boundary extraction is missing, ambiguous, or fails current-output validation, retry current source alone; draft retry failure must drop silently, while final retry failure must follow existing final translation error handling.
- [x] 4.8 Ensure contextual draft inference does not mutate final buffers, merge buffers, final dedup state, `last_translated_source_text`, warning state, Admin callbacks, or context history.
- [x] 4.9 Add bounded diagnostic logs for contextual-input use, fallback-to-current-only, extraction failure, and context history updates without logging full long text.

## 5. Frontend Draft Handling

- [x] 5.1 Update `translationSegmentStoreKey()` to prefer stable `translation_id` when present while preserving fallback compatibility for historical final translations.
- [x] 5.2 Track draft revisions so higher revisions replace lower revisions and late lower/equal revisions are ignored.
- [x] 5.3 Remove incomplete draft translations whose source IDs intersect a newly received final translation, including finals that merge multiple source utterances.
- [x] 5.4 Render `completed: false` translations with the existing incomplete style in split, stacked, interleaved, and single-language layouts.
- [x] 5.5 Preserve `utterance_id` and `source_utterance_ids` binding so interleaved mode updates the corresponding source row.
- [x] 5.6 Clear draft revision and finalization state on translation clear, transcript clear, disconnect, reconnect/session reset, and meeting end.
- [x] 5.7 Do not add frontend typewriter or artificial reveal animation for draft revisions.

## 6. Log And Admin Isolation

- [x] 6.1 Ensure `completed: false` draft translation segments do not call `meeting_logs.append_segments` and cannot appear in meeting JSON, Markdown, DOCX, or summary input.
- [x] 6.2 Ensure draft emissions do not update formal Admin translation message counts, final translation text fields, or persisted translation statistics.
- [x] 6.3 Confirm final `completed: true` translation segments still follow existing persistence, merge, download, and summary behavior.
- [x] 6.4 Confirm readability context is not persisted as extra source or translation text in meeting JSON, Markdown, DOCX, summary input, or Admin status fields.

## 7. Tests And Verification

- [x] 7.1 Add backend tests covering eligible high-accuracy English-to-Chinese draft generation from incomplete ASR text.
- [x] 7.2 Add backend tests proving non-high-accuracy modes, Chinese-to-English, missing client flag, empty text, pure punctuation, and non-English drafts do not generate drafts.
- [x] 7.3 Add scheduler tests for 1.2-second coalescing, min-delta suppression, latest text retention, revision stale drop, and finalization invalidation.
- [x] 7.4 Add tests proving old in-flight draft results cannot override final translations and final translations take priority over pending drafts.
- [x] 7.5 Add tests proving drafts do not enter meeting logs, summary input, formal Admin stats, final dedup state, or merge buffers, while final translations still persist normally.
- [x] 7.6 Add tests or assertions proving no second translation model instance is created and draft/final inference uses the shared model lock.
- [x] 7.7 Add tests covering readability context history updates from successful final translations only, with drafts, warnings, failures, and source-echo fallbacks excluded.
- [x] 7.8 Add tests covering context input construction, latest-two-unit limit, 220-character context trimming, word-boundary trimming, and current sentence not being truncated by context.
- [x] 7.9 Add tests covering boundary marker extraction, fallback to current-only translation when extraction is unreliable, draft silent drop on fallback failure, and final fallback preserving existing error behavior.
- [x] 7.10 Add tests proving readability context never appears in emitted draft/final current rows, meeting logs, summary input, Admin stats, final dedup state, or merge buffers.
- [x] 7.11 Add frontend-oriented tests or code-level checks for `translation_id` revision replacement, final-clears-draft behavior, stale draft ignore behavior, and shared state across layouts where the project test structure supports it.
- [x] 7.12 Run the required minimal checks once: `docker exec whisperlive-gpu0 python3 -m py_compile whisper_live/backend/base.py whisper_live/backend/translation_backend.py whisper_live/server.py`, `docker exec whisperlive-gpu0 python3 -m unittest tests.test_base_backend tests.test_translation_backend tests.test_server_extended`, `docker exec whisperlive-gpu0 node --check web/app.js`, `git diff --check`, `git status --short`, and relevant `git diff`; record any first environment failure without installing dependencies.
