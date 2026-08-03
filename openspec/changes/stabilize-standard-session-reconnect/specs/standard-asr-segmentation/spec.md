## ADDED Requirements

### Requirement: Non-accurate ASR uses selectable conservative segmentation profiles
The system SHALL provide `legacy`, `v2`, and opt-in `v3` segmentation profiles for standard, conversation, and transcription-only faster-whisper sessions without changing accurate-mode behavior.

#### Scenario: V2 keeps current chunk cadence
- **WHEN** a non-accurate faster-whisper client starts with the V2 profile
- **THEN** the client configuration keeps `min_transcription_chunk_seconds` at `2.5` seconds and does not add a new-audio inference interval

#### Scenario: V3 reduces duplicate inference with an explicit interval
- **WHEN** a non-accurate faster-whisper client starts with the V3 profile
- **THEN** it keeps `min_transcription_chunk_seconds` at `2.5` seconds and waits until at least `250ms` of new audio exists beyond the prior inference window before normal repeat inference

#### Scenario: V3 finalization processes the tail immediately
- **WHEN** a V3 client finalizes with buffered audio that has not met the `250ms` new-audio interval
- **THEN** the backend processes that tail without waiting for additional audio

#### Scenario: Accurate mode remains unchanged
- **WHEN** a client starts accurate mode
- **THEN** the existing accurate-mode segmentation, hotword, translation draft, and GPU translation behavior remains unchanged by this capability

### Requirement: Ordinary low-energy filtering remains unchanged
The system SHALL preserve the existing ordinary low-energy threshold for non-accurate modes while retaining existing protections against known silence hallucinations.

#### Scenario: Existing ordinary threshold is retained
- **WHEN** a non-accurate source segment is evaluated for low energy
- **THEN** the system uses the existing ordinary RMS threshold rather than a V2-specific lower threshold

#### Scenario: Silence hallucination protection remains strict
- **WHEN** a recognized segment matches a known silence hallucination phrase or weak-evidence hotword hallucination pattern
- **THEN** the stricter hallucination filters can still drop it using the existing low-energy policy

### Requirement: Repeated and duration-limit completion are less eager in non-accurate modes
The system SHALL reduce premature source completion in non-accurate modes by requiring more stable repeats and a longer incomplete-duration limit than the prior standard profile.

#### Scenario: Stable text does not complete too early
- **WHEN** an incomplete non-accurate source text repeats fewer than the tuned repeat threshold
- **THEN** the backend keeps it as an incomplete row instead of forcing a completed segment only because of repetition

#### Scenario: Long incomplete text still has a bounded fallback
- **WHEN** a non-accurate incomplete source segment exceeds the tuned maximum incomplete duration
- **THEN** the backend completes it with a duration-limit diagnostic rather than allowing it to grow indefinitely

### Requirement: Sentence completion requires punctuation, stability, duration, and trailing silence
The system SHALL complete non-accurate incomplete source text at a sentence boundary only when text and audio evidence both support the boundary.

#### Scenario: Stable punctuated utterance completes after a pause
- **WHEN** non-accurate incomplete text ends with strong sentence punctuation, remains stable for the required observations, lasts at least the minimum utterance duration, and has enough low-energy trailing audio
- **THEN** the backend completes that source segment with a sentence-boundary diagnostic

#### Scenario: Punctuation alone does not complete the sentence
- **WHEN** non-accurate incomplete text has sentence punctuation but lacks stability, minimum duration, or trailing low-energy audio
- **THEN** the backend keeps the segment incomplete unless another completion rule applies

### Requirement: Compatible short fragments are coalesced with bounded delay
The system SHALL reduce short fragmented source rows in non-accurate modes by briefly holding very short completed fragments and merging only safe compatible continuations.

#### Scenario: Short compatible continuation is merged
- **WHEN** a short non-accurate completed source fragment is followed within the bounded hold window by a compatible fragment with matching language and speaker context and a small timing gap
- **THEN** the backend emits one merged completed source segment to the browser, translation queue, and meeting log

#### Scenario: Strong terminal boundary prevents merge
- **WHEN** a short fragment ends with a strong terminal boundary such as a question or exclamation, or the next fragment changes speaker or language incompatibly
- **THEN** the backend emits the original fragment without merging it into the next fragment

#### Scenario: Hold window expires
- **WHEN** no compatible continuation arrives before the bounded hold window expires
- **THEN** the backend emits the original completed fragment unchanged

#### Scenario: V3 uses a longer opt-in hold window
- **WHEN** a non-accurate faster-whisper client starts with the V3 profile
- **THEN** it holds a short completed fragment for up to `2.5s` before releasing it unchanged or merging a compatible continuation

### Requirement: Segmentation diagnostics are bounded and do not store audio
The system SHALL log bounded segmentation diagnostics for tuning and troubleshooting without persisting raw audio.

#### Scenario: Completion and merge reasons are logged
- **WHEN** a non-accurate source segment completes, is held, is merged, or is released without merge
- **THEN** the backend logs the reason, relevant timing and threshold metadata, and a bounded text preview

#### Scenario: Audio is not persisted for diagnostics
- **WHEN** segmentation diagnostics are emitted
- **THEN** the system does not write raw audio samples or derived audio files to project logs or meeting-log storage
