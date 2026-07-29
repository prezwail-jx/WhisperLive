## ADDED Requirements

### Requirement: Faster-whisper finalization completes tail ASR before session finish
The system SHALL request a final faster-whisper ASR pass when a user-initiated `END_OF_AUDIO` is received before finalizing the meeting session.

#### Scenario: Remaining audio is shorter than realtime chunk threshold
- **WHEN** faster-whisper receives `END_OF_AUDIO` and has remaining audio shorter than the normal realtime minimum chunk duration
- **THEN** the existing ASR thread performs one final transcription pass for that remaining audio before reporting ASR finalization complete

#### Scenario: Final ASR tail produces last segment
- **WHEN** the final ASR pass produces a last segment that passes existing silence, hallucination, mixed-noise, RMS, and deduplication checks
- **THEN** the system marks that last segment as completed and enqueues it for final translation

#### Scenario: ASR finalization times out
- **WHEN** faster-whisper ASR finalization cannot complete within the remaining backend finalization budget
- **THEN** the system records `ASR_FINALIZE_TIMEOUT` and continues translation drain and session finalization without hanging indefinitely

### Requirement: Completed source segments reach a translation terminal state
The system SHALL resolve every completed source segment that enters final translation processing to either a successful completed translation segment or a completed failed translation segment.

#### Scenario: Final translation succeeds
- **WHEN** a completed source segment is processed by the translation backend and model output passes validation
- **THEN** the system emits a completed translated segment that references the source segment by utterance ID and time range, and clears any pending draft state for that source segment

#### Scenario: Final translation fails
- **WHEN** a completed source segment cannot be translated because inference fails, model output is rejected, or processing raises a recoverable per-segment exception
- **THEN** the system emits a completed translated segment containing `翻译暂不可用`, retaining the original recognized text in `source_text`, referencing the source segment by utterance ID and time range, and including a stable `translation_warning`

#### Scenario: Merged final translation covers multiple sources
- **WHEN** multiple completed source segments are emitted as one merged final translation
- **THEN** the translated segment references all covered source utterance IDs and the covered time range, and each covered source segment is considered terminal

### Requirement: Draft translations cannot remain stale after source finalization
The browser SHALL stop displaying a draft translation as pending after its source segment has become final and no final translation terminal state arrives within 12 seconds.

#### Scenario: Final translation replaces draft
- **WHEN** a draft translation is visible for a source segment and a completed translated segment referencing or covering that source arrives
- **THEN** the browser removes the draft translation and displays the completed translated segment without pending styling

#### Scenario: Terminal failure replaces draft
- **WHEN** a draft translation is visible for a source segment and a completed failed translated segment referencing or covering that source arrives
- **THEN** the browser removes the draft translation and displays the failed placeholder without pending styling

#### Scenario: Final source times out without terminal translation
- **WHEN** a source segment is completed and no completed translated segment references or covers it within 12 seconds
- **THEN** the browser removes any draft translation for that source and displays a non-pending in-memory `frontend_timeout` fallback row for that source

#### Scenario: Late terminal translation follows frontend fallback
- **WHEN** the browser has displayed a timeout fallback for a source segment and a completed translated segment referencing or covering that source later arrives
- **THEN** the browser replaces the fallback row for that source instead of creating a duplicate translation row

### Requirement: Session finalization drains pending final translations
The system SHALL drain queued final translation work before marking a meeting session finished and before the browser closes the WebSocket for a user-initiated stop, using a 15-second backend finalization budget across ASR finalization and translation drain.

#### Scenario: User stops a meeting with pending final work
- **WHEN** the browser sends `END_OF_AUDIO` while faster-whisper final audio, final source segments, or translation buffer items remain pending
- **THEN** the server completes ASR finalization, flushes translation buffers, processes queued final translations, emits resulting translated segments, finalizes the meeting log, and sends a `SESSION_FINALIZED` message before the browser closes the socket

#### Scenario: Translation drain completes successfully
- **WHEN** all pending final translation work completes within the bounded drain timeout
- **THEN** the meeting log is marked finished after the final translated segments have been appended

#### Scenario: Translation drain times out
- **WHEN** pending final translation work cannot complete within the 15-second backend finalization budget
- **THEN** the system records bounded diagnostics, emits `translation_drain_timeout` placeholders for known unresolved source segments when possible, suppresses duplicate late terminal output for those timed-out source segments, and still completes resource cleanup without hanging indefinitely

#### Scenario: Server finalization message is not received by browser
- **WHEN** the browser has sent `END_OF_AUDIO` and does not receive `SESSION_FINALIZED` within 20 seconds
- **THEN** the browser closes the socket with a visible finish-timeout status, converts remaining pending translation rows to in-memory frontend placeholders, and does not pretend that backend meeting finalization has succeeded

### Requirement: Chinese-to-English context risk triggers direct verification
The system SHALL keep Chinese-to-English readability context enabled, but SHALL run one additional no-context direct translation only when contextual output is high risk.

#### Scenario: Short Chinese source is high risk
- **WHEN** a completed Chinese source segment has at most 24 effective Chinese characters and the contextual English output is available
- **THEN** the system runs one no-context direct translation and emits the direct result only if it passes existing output and glossary validation

#### Scenario: Contextual output is abnormally expanded
- **WHEN** a completed Chinese source segment has contextual English output that is at least 160 characters and more than four times the source character count
- **THEN** the system runs one no-context direct translation and emits the direct result only if it passes existing output and glossary validation

#### Scenario: Contextual output repeats previous translation
- **WHEN** a contextual Chinese-to-English output contains a continuous run of at least four English words from the previous final translation
- **THEN** the system runs one no-context direct translation and emits the direct result only if it passes existing output and glossary validation

#### Scenario: Context extraction or glossary restoration fails
- **WHEN** contextual boundary extraction fails, existing output validation fails, or glossary placeholders cannot be restored safely
- **THEN** the system runs one no-context direct translation and emits a failed placeholder if the direct translation also fails validation

#### Scenario: Pathologically short output remains rejected
- **WHEN** a completed Chinese source segment produces empty, malformed, repetitive, source-echoing, or otherwise structurally invalid English output
- **THEN** the system emits a completed failed translation segment with `翻译暂不可用` and a stable `translation_warning`

#### Scenario: Non-risk contextual output remains single pass
- **WHEN** Chinese-to-English contextual output does not meet any high-risk condition and passes existing validation
- **THEN** the system emits the contextual translation without running a second direct inference

#### Scenario: Context fallback is observable
- **WHEN** the system rejects contextual Chinese-to-English output and falls back to single-segment translation
- **THEN** the system logs a bounded context-fallback event containing the reason, language direction, and source/translated lengths

#### Scenario: Existing abnormal expansion guard remains active
- **WHEN** Chinese-to-English output is at least 240 characters and more than six times the source character count
- **THEN** the system rejects the output according to the existing hard abnormal-expansion guard

### Requirement: Chinese-to-English glossary terms are preserved when recognized
The system SHALL preserve configured translation glossary terms in final Chinese-to-English output when the recognized source text contains the glossary source term.

#### Scenario: Recognized source contains glossary term
- **WHEN** a completed Chinese source segment contains a glossary source term provided by the client hotword or translation glossary payload
- **THEN** the final English translation preserves the configured target term or marks the output for retry/failure if the term cannot be restored safely

#### Scenario: Recognized source does not contain glossary term
- **WHEN** the source ASR text does not contain a configured glossary source term
- **THEN** the translation layer does not inject that glossary target term into the output

#### Scenario: Model emits literal HTML entity
- **WHEN** the model emits literal `&amp;` in a final translation such as `R&amp;D`
- **THEN** the system normalizes the final displayed and persisted translation to `&`
