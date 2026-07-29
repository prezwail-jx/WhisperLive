## ADDED Requirements

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

## MODIFIED Requirements

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
