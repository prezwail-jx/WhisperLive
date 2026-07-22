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

### Requirement: Bounded translation recovery
The system SHALL make at most one recovery attempt for invalid model output or an explicitly transient inference failure and SHALL avoid retrying known non-recoverable failures.

#### Scenario: Invalid first output recovers
- **WHEN** the first model result is classified as invalid and the single retry produces a valid result
- **THEN** the system returns the retry result as a successful translation without failure warning metadata

#### Scenario: Invalid output persists after retry
- **WHEN** the first model result and the single retry are both classified as invalid
- **THEN** the system stops retrying and produces a failed translation segment with the final diagnostic reason

#### Scenario: Batch timeout falls back to direct inference
- **WHEN** NLLB batch translation times out and direct inference is used as fallback
- **THEN** the direct inference counts as the single recovery attempt and the system does not perform a third inference call

#### Scenario: Non-recoverable inference failure occurs
- **WHEN** translation cannot proceed because of CUDA out-of-memory, an unavailable model, client exit, unsupported configuration, or another classified non-recoverable failure
- **THEN** the system produces a failed translation result without repeating inference

### Requirement: User-safe translation failure presentation
The system SHALL present `翻译暂不可用` for a final failed translation instead of presenting the source text or rejected model output as translated content.

#### Scenario: Translation attempts are exhausted
- **WHEN** a translation remains invalid after its allowed recovery attempt
- **THEN** the translated segment contains `翻译暂不可用` in `text`, retains the original recognized content in `source_text`, and includes a stable `translation_warning`

#### Scenario: Historical error-suffix segment is displayed
- **WHEN** the browser loads an existing translation segment whose text ends with `（翻译出错）`
- **THEN** the browser continues to apply its existing compatibility display and warning indicator behavior

### Requirement: Translation failure observability
The system SHALL record bounded operational diagnostics for translation retries and final failures and SHALL persist the final reason with the translation segment.

#### Scenario: Invalid output triggers retry
- **WHEN** model output is classified as invalid and a retry is initiated
- **THEN** the system logs a retry event containing the client, model, language pair, failure reason, timing context, and bounded text previews

#### Scenario: Translation ends in failure
- **WHEN** recovery is exhausted or a non-recoverable failure occurs
- **THEN** the system logs a terminal failure event and stores the stable reason in the segment's `translation_warning` field

#### Scenario: Translation succeeds normally
- **WHEN** the initial or retry model result passes validation
- **THEN** the system does not attach failure warning metadata to the successful segment

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
