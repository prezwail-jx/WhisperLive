## ADDED Requirements

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
