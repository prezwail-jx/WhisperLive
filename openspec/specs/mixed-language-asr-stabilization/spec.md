# Mixed-Language ASR Stabilization Specification

## Purpose

Define conditional mixed-language ASR retry behavior to stabilize faster-whisper language switching during high-accuracy bidirectional interpretation, ensuring consistent source-language metadata propagation through the transcription and translation pipeline.

## Requirements

### Requirement: Conditional mixed-language ASR retry
The system SHALL perform at most one conditional same-audio ASR retry with the previous stable language when a high-accuracy mixed-interpretation faster-whisper chunk proposes a suspicious isolated switch between Chinese and English.

#### Scenario: Suspicious switch triggers previous-language retry
- **WHEN** a high-accuracy mixed-interpretation faster-whisper client has a previous stable `zh` or `en` language and the next automatic decode proposes the other language without strong acceptance evidence
- **THEN** the system SHALL decode the same audio once more using the previous stable language as the forced ASR language

#### Scenario: Non-mixed modes do not retry
- **WHEN** a client is not using both `service_mode: accurate` and `translation_mode: mixed_interpretation`
- **THEN** the system SHALL NOT perform the conditional mixed-language ASR retry

#### Scenario: Non-faster-whisper backends do not retry
- **WHEN** the active ASR backend is not faster-whisper
- **THEN** the system SHALL NOT perform the conditional mixed-language ASR retry

#### Scenario: Retry is bounded per audio chunk
- **WHEN** a suspicious switch has already caused one previous-language retry for the current audio chunk
- **THEN** the system MUST NOT perform another language retry for that same audio chunk

### Requirement: Mixed-language ASR candidate selection
The system SHALL choose between the automatic ASR candidate and the previous-language retry candidate using bounded evidence before emitting source segments or sending completed segments to translation.

#### Scenario: Retry candidate is safer
- **WHEN** the retry candidate has better language-text consistency and is not worse by the configured average log-probability margin
- **THEN** the system SHALL select the retry candidate and keep the previous stable language

#### Scenario: Automatic candidate is clearly stronger
- **WHEN** the automatic candidate has strong language probability or a clearly better average log probability and passes existing hallucination and noise guards
- **THEN** the system SHALL accept the automatic candidate as a real language switch and update the stable language

#### Scenario: Candidate is rejected by safety guard
- **WHEN** an ASR candidate is dominated by existing hard hallucination, mixed-interpretation noise, or repeated hotword rules
- **THEN** the system SHALL NOT prefer that candidate over a safe alternative

### Requirement: Consistent source language metadata
The system SHALL propagate the selected mixed-language ASR language consistently to the WebSocket source segment, translation queue segment, Admin source status, and meeting log source segment.

#### Scenario: ASR selected language is present
- **WHEN** a completed source segment already contains selected language metadata of `zh` or `en`
- **THEN** downstream server and translation processing SHALL preserve that language instead of overriding it using text-only character heuristics

#### Scenario: ASR selected language is missing
- **WHEN** a source segment does not contain selected `zh` or `en` language metadata
- **THEN** downstream processing may infer the source language from text using the existing character heuristic

### Requirement: Mixed-language ASR retry observability
The system SHALL emit bounded diagnostic logs for mixed-language ASR retry decisions without logging full long recognized text.

#### Scenario: Suspicious switch is evaluated
- **WHEN** the system evaluates an automatic mixed-language switch candidate
- **THEN** it SHALL log the client uid, previous language, automatic language, bounded probabilities, candidate scores, and bounded text preview

#### Scenario: Retry result is selected or rejected
- **WHEN** the retry candidate is compared with the automatic candidate
- **THEN** the system SHALL log which candidate was selected and the bounded score evidence used for selection
