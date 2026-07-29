## ADDED Requirements

### Requirement: Draft translation enablement
The system SHALL emit draft Chinese translations only when high-accuracy interpretation is active, the ASR draft source language is English, the resolved target language is Chinese, and the client explicitly enables translation drafts in its connection configuration.

#### Scenario: High-accuracy English-to-Chinese draft is enabled
- **WHEN** a client connects with `service_mode` set to `accurate`, `translation_draft_enabled` set to true, an English ASR draft segment, and resolved translation target language `zh`
- **THEN** the system may schedule a Chinese draft translation for that ASR draft utterance

#### Scenario: Non-high-accuracy modes do not draft
- **WHEN** a client uses standard interpretation, conversation translation, or transcription mode
- **THEN** the system MUST NOT schedule or emit draft translation segments even if draft fields are present in the client configuration

#### Scenario: Non-English source does not draft
- **WHEN** the current ASR draft source language is not English
- **THEN** the system MUST NOT schedule or emit a Chinese draft translation for that draft

#### Scenario: Target is not Chinese
- **WHEN** the resolved translation target language is not `zh`
- **THEN** the system MUST NOT schedule or emit draft translation segments

#### Scenario: Old client omits draft flag
- **WHEN** a client does not send `translation_draft_enabled`
- **THEN** the system MUST keep existing final-translation-only behavior

### Requirement: Draft translation configuration bounds
The system SHALL normalize draft translation configuration values before use and SHALL apply safe defaults and lower or upper bounds.

#### Scenario: Frontend sends high-accuracy draft configuration
- **WHEN** the frontend starts a high-accuracy connection
- **THEN** it SHALL send `translation_draft_enabled: true`, `translation_draft_interval_seconds: 1.2`, `translation_draft_min_delta_chars: 8`, and `translation_draft_max_source_chars: 220`

#### Scenario: Frontend starts non-high-accuracy connection
- **WHEN** the frontend starts a non-high-accuracy connection
- **THEN** it SHALL send `translation_draft_enabled: false` or omit the draft translation fields

#### Scenario: Backend receives invalid draft interval
- **WHEN** the backend receives a missing, non-numeric, or too-small draft interval
- **THEN** it SHALL use a normalized interval of at least 0.5 seconds

#### Scenario: Backend receives invalid delta threshold
- **WHEN** the backend receives a missing, non-numeric, or too-small draft minimum delta
- **THEN** it SHALL use a normalized minimum delta of at least 1 character

#### Scenario: Backend receives invalid source length limit
- **WHEN** the backend receives a missing, non-numeric, too-small, or too-large draft source length limit
- **THEN** it SHALL clamp the value to a bounded range and default to 220 characters when needed

### Requirement: Latest-slot draft scheduling
The system SHALL maintain a latest draft slot per client and source utterance and SHALL coalesce frequent ASR draft updates before invoking translation.

#### Scenario: New ASR draft replaces pending draft
- **WHEN** a newer English ASR draft arrives for the same `utterance_id` before a previous draft starts inference
- **THEN** the system SHALL replace the pending draft text with the newest draft text and keep only one pending draft for that utterance

#### Scenario: Draft interval has not elapsed
- **WHEN** a draft update arrives less than the configured interval after the previous draft inference for the client
- **THEN** the system SHALL NOT start another draft inference immediately

#### Scenario: Draft delta is too small
- **WHEN** the latest draft differs from the last translated draft by fewer than the configured minimum delta characters
- **THEN** the system SHALL NOT start another draft inference for that update

#### Scenario: Draft text is unsuitable
- **WHEN** the draft text is empty, pure punctuation, or not an English ASR draft
- **THEN** the system SHALL skip draft translation for that update

#### Scenario: Draft source is too long
- **WHEN** the latest English draft exceeds the configured draft source length limit
- **THEN** the system SHALL translate only a bounded suffix of the current utterance and MUST NOT cut through the middle of an English word when trimming is possible

#### Scenario: One draft inference per client
- **WHEN** a client already has a draft inference in progress
- **THEN** the system SHALL NOT start a second draft inference for that client until the in-progress draft finishes or is discarded

### Requirement: Draft revision and finalization safety
The system SHALL use per-utterance monotonic revisions and finalization checks so stale draft results cannot overwrite newer drafts or final translations.

#### Scenario: Draft revision is current
- **WHEN** a draft inference completes and its revision still matches the latest revision for the source utterance
- **THEN** the system may emit the draft if the utterance is not finalized, the client is active, and the direction is still English-to-Chinese

#### Scenario: Draft revision is stale
- **WHEN** a draft inference completes with a revision older than the latest known revision for the source utterance
- **THEN** the system SHALL silently drop the draft result

#### Scenario: Final segment arrives with pending draft
- **WHEN** the final ASR segment arrives for an utterance that has a pending draft
- **THEN** the system SHALL invalidate that pending draft and prioritize final translation

#### Scenario: In-flight draft completes after final translation
- **WHEN** an already-running draft inference completes after the source utterance has been finalized
- **THEN** the system SHALL silently drop the draft result and MUST NOT emit it to the client

#### Scenario: Direction changes before draft result
- **WHEN** a draft inference completes after the client direction is no longer English-to-Chinese
- **THEN** the system SHALL drop the draft result

### Requirement: Draft WebSocket translation segments
The system SHALL send draft translations only to the owning WebSocket client using `translated_segments` entries marked as incomplete and carrying stable draft identity metadata.

#### Scenario: Draft translation is emitted
- **WHEN** a draft Chinese translation result is valid and current
- **THEN** the system SHALL send a `translated_segments` message containing one segment with `completed: false`, the source `utterance_id`, `source_utterance_ids: [utterance_id]`, stable `translation_id`, current `revision`, `source_text`, `source_language: en`, `target_language: zh`, `translation_model`, and draft `text`

#### Scenario: Final translation is emitted
- **WHEN** final translation completes for the source utterance
- **THEN** the system SHALL continue to emit a `completed: true` translated segment using existing final translation fields and source utterance binding

#### Scenario: Draft result fails validation
- **WHEN** draft translation output is empty, invalid, or classified as failed by output guards
- **THEN** the system SHALL NOT emit a user-visible failed draft segment and SHALL leave final translation behavior unchanged

### Requirement: Frontend draft replacement
The browser SHALL display translation drafts as incomplete translation rows and replace or remove them according to `translation_id`, `revision`, and final source bindings.

#### Scenario: Higher draft revision arrives
- **WHEN** a `completed: false` translated segment arrives with the same `translation_id` and a higher `revision`
- **THEN** the browser SHALL replace the previous draft with the higher revision draft

#### Scenario: Lower draft revision arrives late
- **WHEN** a `completed: false` translated segment arrives with the same `translation_id` and a lower or equal `revision` than the stored draft
- **THEN** the browser SHALL ignore the stale draft

#### Scenario: Final translation arrives for draft source
- **WHEN** a `completed: true` translated segment arrives whose `source_utterance_ids` intersects an existing draft segment
- **THEN** the browser SHALL remove the matching draft before storing and rendering the final translation

#### Scenario: Final translation merges multiple source utterances
- **WHEN** a final translated segment references multiple `source_utterance_ids`
- **THEN** the browser SHALL remove all incomplete draft translations whose source IDs intersect the final segment source IDs

#### Scenario: Layout changes during draft display
- **WHEN** the user switches among split, stacked, interleaved, and single-language layouts while a draft is visible
- **THEN** all layouts SHALL render the same draft/final translation state without replaying or duplicating draft content

#### Scenario: Draft row is rendered
- **WHEN** a draft translation is displayed in any layout
- **THEN** the browser SHALL mark it as incomplete using the existing incomplete presentation style

### Requirement: Draft isolation from logs and Admin state
Draft translation segments SHALL NOT affect meeting logs, summary inputs, final translation buffers, final translation deduplication, or formal Admin translation statistics.

#### Scenario: Draft segment is sent to client
- **WHEN** a `completed: false` translated segment is emitted over WebSocket
- **THEN** the system MUST NOT call `meeting_logs.append_segments` for that draft segment

#### Scenario: Meeting is exported after drafts
- **WHEN** a meeting containing draft translation activity is exported as JSON, Markdown, or DOCX
- **THEN** the exported meeting log SHALL include final translation segments only and SHALL NOT include draft translation text

#### Scenario: Summary is generated after drafts
- **WHEN** an AI meeting summary is generated after draft translation activity
- **THEN** the summary input SHALL be based on final meeting log content and SHALL NOT include draft translation text

#### Scenario: Admin status is updated
- **WHEN** draft translation segments are emitted
- **THEN** formal Admin translation message counts, final translation text status, and persisted translation statistics SHALL NOT treat drafts as final translation output

### Requirement: Draft translation resource constraints
The implementation SHALL reuse existing translation model resources and SHALL keep draft inference serial with the existing model lock.

#### Scenario: Draft translation runs
- **WHEN** the system performs draft translation inference
- **THEN** it SHALL reuse the existing loaded NLLB model, tokenizer, translation device, model path, precision, and shared inference lock

#### Scenario: Draft feature is enabled
- **WHEN** draft translation is enabled for a client
- **THEN** the system MUST NOT create or copy an additional translation model instance, increase batch size, enable default parallel inference, add a persistent GPU cache, change `translation_device`, or modify vLLM summary service behavior

#### Scenario: Final translation waits behind in-flight draft
- **WHEN** final translation becomes ready while a draft inference is already inside the shared model lock
- **THEN** the system may wait for that in-flight draft to finish, but it SHALL prioritize final translation over not-yet-started drafts

### Requirement: Draft observability
The system SHALL emit bounded diagnostic logs for draft scheduling and stale drops without logging complete long draft text.

#### Scenario: Draft is scheduled or coalesced
- **WHEN** the draft scheduler schedules or coalesces a draft update
- **THEN** it SHALL log a bounded diagnostic event containing client uid, utterance id, revision, source character count, and scheduling reason without full long text

#### Scenario: Draft is emitted or dropped
- **WHEN** a draft result is emitted or dropped as stale
- **THEN** it SHALL log a bounded diagnostic event containing client uid, utterance id, revision, character counts, and elapsed time without full long text
