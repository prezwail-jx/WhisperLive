## Context

High-accuracy interpretation currently translates only completed ASR segments. English source text can update as a draft while the speaker is still talking, but Chinese translation appears only after the utterance completes and final translation processing finishes. A frontend-only typewriter animation would not expose any earlier information, so the feature needs backend-generated draft translations based on the latest English ASR draft.

Completed English-to-Chinese translation calls are also mostly sentence-local today. That keeps latency predictable, but it weakens readability when the speaker uses pronouns, short references, or repeated domain terms across adjacent utterances. Draft and final translation should therefore share a bounded readability context from recently successful final English units, while displaying and persisting only the current utterance translation.

The implementation must preserve the existing final-translation path, meeting logs, summary inputs, Admin statistics, and model resource behavior. Draft inference and contextual final translation may increase translation call count or input length, but they must reuse the existing translation model, tokenizer, device, and shared inference lock and remain opt-in for high-accuracy English-to-Chinese sessions only. Bounded extra GPU time and transient memory use are acceptable; new model instances, default parallel inference, batch changes, and persistent GPU caches are not.

## Goals / Non-Goals

**Goals:**

- Provide ASR-like Chinese draft translation updates while an English utterance is still incomplete in high-accuracy mode.
- Improve high-accuracy English-to-Chinese readability by giving draft and final translations access to the most recent successful English context without showing that context in the current row.
- Coalesce frequent ASR draft updates into one latest draft slot per utterance rather than enqueueing every update.
- Use monotonic revisions so stale draft results cannot overwrite newer drafts or final translations.
- Ensure final translation takes priority and invalidates pending or late draft results.
- Keep draft translations out of meeting logs, summary input, formal Admin translation counters, and persistent translation buffers.
- Keep readability context history final-only: drafts, failed placeholders, warning outputs, and source-echo fallbacks must not enter context.
- Keep model memory bounded by reusing the existing translator instance, tokenizer, device, and inference lock.

**Non-Goals:**

- This is not token streaming from NLLB `generate()` and does not expose decoder tokens directly.
- This does not enable draft translation for standard interpretation, conversation mode, transcription mode, or Chinese-to-English output.
- This does not create a second translation model, change GPU selection, increase batch size, change model precision, or modify vLLM summary behavior.
- This does not change WebSocket endpoint paths, meeting log file formats, or Admin REST APIs.
- This does not provide document-level translation memory. Readability context is limited to the most recent successful final translation units.

## Decisions

1. Use ASR draft text snapshots instead of decoder token streaming.

   NLLB token streaming would require reworking model generation and still would not incorporate later ASR context until a new source snapshot is translated. Translating the latest English ASR draft at bounded intervals better matches the desired ASR-like behavior: the whole Chinese draft line may revise as more source context arrives.

2. Add explicit opt-in connection configuration.

   The frontend will send `translation_draft_enabled`, `translation_draft_interval_seconds`, `translation_draft_min_delta_chars`, and `translation_draft_max_source_chars`. It will also send `translation_readability_context_sentences` and `translation_readability_context_max_chars` for high-accuracy English-to-Chinese sessions. The backend normalizes these values with safe bounds and defaults to disabled for old clients. This avoids behavior changes outside high-accuracy English-to-Chinese mode.

3. Maintain a per-client latest draft scheduler instead of using the existing FIFO translation queue.

   The existing queue is for final translation of completed ASR segments. Drafts can update many times per utterance, so the scheduler keeps only the newest draft per `utterance_id`, enforces interval and delta thresholds, and allows at most one draft inference per client at a time.

4. Reuse the existing translation path without mutating final state.

   Draft inference should reuse language resolution, glossary/term protection, model output guards, and the shared translator lock, but it must not update final dedup state, translation merge buffers, `last_translated_source_text`, warning state, or formal statistics. Draft emission uses a dedicated WebSocket-only path.

5. Build bounded readability context from successful final units only.

   The translation client maintains a small per-client history of recent successful final English source units. The configured default is the latest 2 units, with the combined context source capped at 220 characters. If the context exceeds the cap, trim from the oldest side first and keep the current source sentence intact. Drafts, failed translations, placeholders such as `翻译暂不可用`, warning outputs, and source-echo fallbacks do not update this history.

6. Use protected boundaries and current-utterance extraction.

   Draft and final English-to-Chinese translation may call NLLB with an input shaped as recent context, a protected boundary marker, and the current source sentence. The output shown to users and persisted for final translations must contain only the current sentence translation. If the boundary marker is missing, extraction is ambiguous, or output validation fails for the extracted current sentence, retry the current source sentence alone. If draft fallback fails, drop the draft silently; if final fallback fails, keep the existing final translation error behavior.

7. Use stable draft identity and revisions.

   Each draft segment carries `translation_id` and `revision`. The frontend uses `translation_id` first, ignores lower revisions, and clears draft entries when a final segment shares any `source_utterance_ids`. The backend rechecks revision, finalization state, client exit, and current English-to-Chinese direction before emitting a draft result.

8. Keep logs and summaries final-only.

   Draft segments are `completed: false` and must not call `meeting_logs.append_segments`. Admin status and summary generation continue to operate only on final translations.

## Readability Context Algorithm

1. Eligibility

   Contextual translation is enabled only when the same high-accuracy English-to-Chinese draft feature is explicitly enabled and the client config has a positive `translation_readability_context_sentences` value. Other modes, Chinese-to-English output, old clients, and transcription-only sessions continue to translate the current source without readability context.

2. History source

   The translation client records successful final translation units after final translation completes and passes output validation. A unit stores the final English `source_text` and the final Chinese `text` for diagnostics, but only the English source units are used as NLLB input context. The history length is capped by `translation_readability_context_sentences`, defaulting to 2.

3. Context trimming

   Build context from oldest to newest among the retained units, then clamp the combined context source to `translation_readability_context_max_chars`, defaulting to 220. Trimming removes older content before newer content and should avoid cutting through the middle of an English word when possible. The current source sentence has its own source limit and is never shortened to make room for history.

4. Contextual input and extraction

   The contextual source is constructed as `context + boundary marker + current source`. The marker must be unlikely to appear naturally in speech text and must survive enough NLLB outputs to support extraction. After translation, extract only the text after the marker's translated or protected counterpart. The extracted text then goes through the same output guard and cleanup checks as a normal current-sentence translation.

5. Fallbacks

   When marker extraction is unreliable, contextual output appears to include history, or the extracted current sentence fails validation, retry with current source only. Draft retry failure drops the draft without a user-visible error. Final retry failure follows existing final translation fallback and warning handling.

## Risks / Trade-offs

- Increased translation calls → Bounded by interval, delta, source length, one in-flight draft per client, and final-translation priority.
- Longer translation input → Bounded by at most 2 history units and 220 history characters, with the current sentence capped separately. This can improve readability but may add bounded GPU time.
- Draft text may change substantially → This is accepted; drafts are explicitly provisional and are replaced by later revisions or the final translation.
- Contextual output may leak previous context → Protected boundaries and extraction are required; ambiguous extraction falls back to current-only translation.
- Draft inference may complete after final translation → Revision and finalization checks drop stale draft results before emission.
- Draft inference can add latency to final translation if it holds the shared lock → Final translation gets priority over pending drafts, but already-running draft inference is not interrupted. This preserves model safety while bounding additional delay through draft interval and source length limits.
- Frontend may leave duplicate draft/final rows → Final merge logic must delete drafts whose `source_utterance_ids` intersect final segments, including final translations that merge multiple source utterances.
- Container lacks Node in some environments → Implementation verification should attempt the configured container `node --check` once and report the environment failure if `node` is unavailable.

## Migration Plan

- Add backend defaults with draft translation disabled, so existing clients behave unchanged.
- Add frontend draft and readability context config only for high-accuracy mode; non-high-accuracy modes send `false` or omit these fields.
- Deploy backend and frontend together. If issues occur, disable the frontend flag or keep backend default disabled; final translation behavior remains intact.

## Open Questions

- Whether the backend should expose a runtime CLI flag for allowing draft translation globally in addition to the client opt-in flag. The proposed default is disabled unless both backend normalization and client config allow it.
- Which boundary marker survives NLLB most reliably while remaining easy to extract and unlikely to collide with user speech.
