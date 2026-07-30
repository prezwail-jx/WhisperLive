# Translation Output Reliability Specification

## Purpose

Define reliable translation output handling, including model-output validation, bounded recovery, user-safe failure presentation, observability, and failed-segment merge isolation.

## Requirements

### Requirement: Unified model-output validation
The system SHALL validate every model-generated translation result using one output policy before presenting or persisting it as a successful translation.

#### Scenario: Model returns the normalized source text
- **WHEN** a cross-language model result is equivalent to the source text after Unicode, surrounding whitespace, and common punctuation are normalized and the text is not an exempt bounded proper term
- **THEN** the system classifies the result as `source_echo` and does not present it as a successful translation

#### Scenario: Short proper term remains unchanged
- **WHEN** cross-language model output preserves a configured translation term or a bounded short ASCII proper term composed only of acronym, mixed-case, or numeric tokens
- **THEN** the system does not classify that identity output as `source_echo`

#### Scenario: Model returns empty or structurally invalid output
- **WHEN** a model result is empty, abnormally long, excessively repetitive, or contains a configured malformed-character run
- **THEN** the system classifies the result with a stable output-failure reason

#### Scenario: Chinese-to-English output retains substantial Chinese text
- **WHEN** an NLLB Chinese-to-English result crosses the configured residual-CJK character and ratio thresholds
- **THEN** the system classifies the result as `residual_cjk`

#### Scenario: Trusted translation rule produces output
- **WHEN** a fixed short-phrase or exact glossary rule resolves the translation without model generation
- **THEN** the system returns the trusted target without applying model-output retry policy

### Requirement: Bidirectional completeness validation
The system SHALL validate high-accuracy NLLB final translations for bidirectional completeness before treating a model result as fully successful.

#### Scenario: English-to-Chinese output omits numeric or unit anchors
- **WHEN** a high-accuracy NLLB English-to-Chinese result omits a required numeric, percentage, currency, unit, acronym, or configured glossary fact anchor from the English source
- **THEN** the system SHALL classify the result as incomplete and enter the bounded recovery pipeline

#### Scenario: Chinese-to-English output omits numeric or unit anchors
- **WHEN** a high-accuracy NLLB Chinese-to-English result omits a required numeric, percentage, currency, unit, acronym, or configured glossary fact anchor from the Chinese source
- **THEN** the system SHALL classify the result as incomplete and enter the bounded recovery pipeline

#### Scenario: Short source contains fact anchors
- **WHEN** a short source segment contains required fact anchors
- **THEN** fact-anchor completeness validation SHALL apply even if the segment is shorter than the normal length-based undertranslation threshold

#### Scenario: Output preserves equivalent units
- **WHEN** a translation preserves a fact anchor using an accepted equivalent such as `tons` to `吨`, `RMB` to `人民币`, or percent wording to `%` or `百分之`
- **THEN** the system SHALL treat that fact anchor as covered

### Requirement: Incomplete English fragment handling
The system SHALL keep final English incomplete-fragment waiting bounded while recognizing common dangling connector phrases.

#### Scenario: Dangling connector waits for context
- **WHEN** an English translation buffer ends with a configured dangling connector phrase such as `and they`, `but they`, `and we`, `assuming you`, or `we need to`
- **THEN** the system SHALL wait for more completed source context until the configured incomplete wait budget expires

#### Scenario: Incomplete wait budget expires
- **WHEN** the incomplete wait budget expires for an English buffer
- **THEN** the system SHALL flush the current best translation candidate without setting `translation_warning` solely because of `incomplete_timeout`

#### Scenario: Timeout candidate remains safe but incomplete
- **WHEN** an incomplete-timeout translation candidate is safe to display but remains incomplete after recovery attempts
- **THEN** the system SHALL emit the candidate with `translation_confidence: "low"` and without `translation_warning`

### Requirement: Chinese-to-English final sentence buffering
The system SHALL buffer completed Chinese source segments before final Chinese-to-English translation when the buffered source appears semantically incomplete, and SHALL flush that buffer using bounded timing and boundary conditions.

#### Scenario: Incomplete Chinese fragment waits for continuation
- **WHEN** a completed Chinese source segment enters final translation processing for an English target and the buffered text ends with a configured incomplete Chinese connective or lead-in phrase
- **THEN** the system keeps the segment in the final translation buffer instead of invoking fixed translation, glossary translation, or model translation immediately

#### Scenario: Punctuated incomplete Chinese fragment still waits
- **WHEN** a completed Chinese source segment for an English target ends with sentence punctuation but the text before trailing punctuation ends with a configured incomplete Chinese phrase
- **THEN** the system treats the buffered source as incomplete and keeps waiting within the bounded buffering limits

#### Scenario: Following Chinese segment completes the buffered source
- **WHEN** a following completed Chinese segment arrives before any boundary or timeout flushes the buffer
- **THEN** the system joins the buffered Chinese source text and translates the joined text as one final translation unit

#### Scenario: Complete Chinese sentence flushes immediately
- **WHEN** the buffered Chinese source for an English target ends with a complete sentence ending and does not match incomplete-Chinese heuristics
- **THEN** the system flushes the final translation buffer without waiting for the idle timeout

#### Scenario: Idle timeout flushes incomplete Chinese buffer
- **WHEN** a Chinese-to-English final translation buffer remains incomplete and no new source segment arrives within the configured Chinese-to-English idle timeout
- **THEN** the system flushes the buffer for final translation rather than waiting indefinitely

#### Scenario: Hard bounds flush Chinese buffer
- **WHEN** a Chinese-to-English final translation buffer reaches the configured maximum accumulated audio duration or maximum source character limit
- **THEN** the system flushes the buffer for final translation even if the incomplete-Chinese heuristic still matches

#### Scenario: Boundary prevents unrelated merge
- **WHEN** a buffered Chinese-to-English source segment is followed by a segment with a different source language, a different speaker, or a source time gap greater than the configured maximum merge gap
- **THEN** the system flushes the existing buffer before adding the new segment

#### Scenario: Merged Chinese translation preserves source references
- **WHEN** multiple completed Chinese source segments are translated as one final Chinese-to-English segment
- **THEN** the translated segment references all covered source utterance IDs and the covered start/end time range so each completed source segment reaches a terminal translation state

#### Scenario: English-to-Chinese remains on existing path
- **WHEN** an English source segment enters final translation processing for a Chinese target
- **THEN** the system uses the existing English incomplete-sentence buffering behavior and does not apply Chinese incomplete-sentence heuristics

### Requirement: Bounded translation recovery
The system SHALL use staged bounded recovery for invalid, incomplete, or explicitly transient translation failures and SHALL avoid retrying known non-recoverable failures.

#### Scenario: Invalid first output recovers
- **WHEN** the first model result is classified as invalid or incomplete and a later staged recovery candidate produces a safe and complete result
- **THEN** the system returns the recovery result as a successful translation without failure warning metadata

#### Scenario: Invalid output persists after recovery
- **WHEN** all staged recovery candidates are unsafe or unavailable
- **THEN** the system stops retrying and produces a failed translation segment with the final diagnostic reason

#### Scenario: Batch timeout falls back to direct inference
- **WHEN** NLLB batch translation times out and direct inference is used as fallback
- **THEN** the direct inference SHALL count as a recovery stage and the system SHALL NOT retry indefinitely

#### Scenario: Non-recoverable inference failure occurs
- **WHEN** translation cannot proceed because of CUDA out-of-memory, an unavailable model, client exit, unsupported configuration, or another classified non-recoverable failure
- **THEN** the system produces a failed translation result without repeating inference

#### Scenario: Context retry is available
- **WHEN** a translation result is unsafe or incomplete and bounded readability context is available for the same language direction
- **THEN** the system may retry translation with the existing bounded context and SHALL reject the context candidate if boundary extraction, context leak checks, or output safety checks fail

#### Scenario: Chunked retry is available
- **WHEN** strict, context, and relaxed candidates remain incomplete for a high-accuracy NLLB segment and the source can be split into bounded chunks
- **THEN** the system may translate up to three chunks and use the chunked candidate only if it passes safety checks and improves completeness evidence

### Requirement: Conservative relaxed generation
The system SHALL support a conservative relaxed NLLB generation profile for recovery candidates without increasing persistent model resources.

#### Scenario: Relaxed recovery is attempted
- **WHEN** strict and context translation candidates remain unsafe or incomplete for high-accuracy NLLB
- **THEN** the system may generate a relaxed candidate using `num_beams=3`, `max_new_tokens=320`, and `length_penalty=1.1`

#### Scenario: Relaxed candidate is unsafe
- **WHEN** a relaxed translation candidate fails safety validation
- **THEN** the system SHALL NOT present that candidate as successful or low-confidence output

#### Scenario: Relaxed profile runs
- **WHEN** the relaxed generation profile is used
- **THEN** the system MUST reuse the existing loaded NLLB model, tokenizer, device, and inference lock without creating another persistent model instance

### Requirement: User-safe translation failure presentation
The system SHALL present a target-language unavailable placeholder only when no safe translation candidate can be displayed.

#### Scenario: Safe candidate remains incomplete
- **WHEN** recovery attempts produce at least one safe candidate but no complete candidate
- **THEN** the translated segment contains the best safe candidate in `text`, retains the original recognized content in `source_text`, includes `translation_confidence: "low"`, and does not include `translation_warning`

#### Scenario: Translation attempts are exhausted without safe candidate
- **WHEN** a translation remains unsafe or unavailable after its allowed recovery stages
- **THEN** the translated segment contains the target-language unavailable placeholder in `text`, retains the original recognized content in `source_text`, and includes a stable `translation_warning`

#### Scenario: Chinese target unavailable placeholder is emitted
- **WHEN** a final failed translation resolves target language to `zh`
- **THEN** the translated segment text SHALL be `翻译暂不可用`

#### Scenario: English target unavailable placeholder is emitted
- **WHEN** a final failed translation resolves target language to `en`
- **THEN** the translated segment text SHALL be `Translation unavailable`

#### Scenario: Historical error-suffix segment is displayed
- **WHEN** the browser loads an existing translation segment whose text ends with `（翻译出错）`
- **THEN** the browser continues to apply its existing compatibility display and warning indicator behavior

### Requirement: Low-confidence translation metadata
The system SHALL persist low-confidence metadata for safe best-effort translations without using warning metadata intended for visible failure indicators.

#### Scenario: Best safe candidate is incomplete
- **WHEN** all recovery candidates are safe but the best candidate still has completeness failures
- **THEN** the emitted final translation segment SHALL contain `translation_confidence: "low"` and SHALL NOT contain `translation_warning`

#### Scenario: Low-confidence segment is persisted
- **WHEN** a final translation segment contains `translation_confidence: "low"`
- **THEN** meeting JSON export SHALL preserve the field with that segment

#### Scenario: Low-confidence segment is merged
- **WHEN** a merged final translation contains one or more low-confidence source translation segments
- **THEN** the merged translation segment SHALL preserve `translation_confidence: "low"`

#### Scenario: Low-confidence segment is displayed
- **WHEN** the browser receives a final translation segment that contains `translation_confidence: "low"` and no `translation_warning`
- **THEN** it SHALL display the translation text without the warning marker used for `translation_warning`

### Requirement: Translation failure observability
The system SHALL record bounded operational diagnostics for staged translation recovery, low-confidence output, and final failures and SHALL persist the final reason with failed translation segments.

#### Scenario: Invalid output triggers retry
- **WHEN** model output is classified as unsafe or incomplete and a retry is initiated
- **THEN** the system logs a retry event containing the client, model, language pair, stage, failure reason, timing context, and bounded text previews

#### Scenario: Low-confidence output is emitted
- **WHEN** recovery ends with a safe but incomplete best candidate
- **THEN** the system logs a low-confidence event containing the client, model, language pair, selected stage, completeness reason, and bounded coverage metrics

#### Scenario: Translation ends in failure
- **WHEN** recovery is exhausted or a non-recoverable failure occurs without any safe candidate
- **THEN** the system logs a terminal failure event and stores the stable reason in the segment's `translation_warning` field

#### Scenario: Translation succeeds normally
- **WHEN** the initial or retry model result passes validation as safe and complete
- **THEN** the system does not attach failure warning metadata or low-confidence metadata to the successful segment

### Requirement: Failed-segment merge isolation
The system SHALL preserve a failed translation as an independent segment and SHALL not merge its placeholder or warning metadata with adjacent successful translations.

#### Scenario: Failure follows buffered successful translation
- **WHEN** a failed translation arrives while successful translation text is waiting in the merge buffer
- **THEN** the system flushes the successful buffer before emitting the failed segment independently

#### Scenario: Successful translation follows failure
- **WHEN** a successful translation arrives after an independently emitted failed segment
- **THEN** the successful translation starts a new merge buffer without inheriting the failure warning

#### Scenario: Browser displays multiple translations for one source
- **WHEN** a failed translation and a successful translation reference the same source utterance
- **THEN** the browser displays the failed placeholder as a separate translation row and marks repeated source rows as the same source segment instead of concatenating the texts

#### Scenario: Connection closes with buffered success
- **WHEN** a successful translation is waiting only for merge delay and the client is cleaned up
- **THEN** the system force-flushes the merge buffer before exit cleanup so the translated segment is emitted

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
