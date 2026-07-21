## ADDED Requirements

### Requirement: Safe hotword normalization
The system SHALL normalize ASR hotwords into an ordered, deduplicated list and SHALL enforce bounded per-term, term-count, and total-prompt limits before initializing a recognition backend.

#### Scenario: Duplicate and malformed terms are normalized
- **WHEN** a hotword source contains blank entries, repeated terms, comments, or invalid oversized terms
- **THEN** the system excludes invalid entries, retains the first occurrence of each valid term, and preserves the order of accepted terms

#### Scenario: Prompt safety limit is reached
- **WHEN** valid hotwords exceed the configured count or total-prompt limit
- **THEN** the system uses only the ordered prefix that fits the limits and records that truncation occurred

### Requirement: Hotword source precedence
The system SHALL preserve explicit locked or client hotwords over meeting hotwords, and meeting hotwords over default hotwords, while producing one canonical ASR prompt for the selected source.

#### Scenario: Meeting hotwords override defaults
- **WHEN** a connection selects a meeting with valid hotwords and does not provide locked hotwords
- **THEN** the system conditions ASR with the normalized meeting hotwords and does not append default hotwords

#### Scenario: Locked client hotwords override meeting selection
- **WHEN** a connection supplies locked valid hotwords and also selects a meeting
- **THEN** the system conditions ASR with only the normalized locked hotwords

### Requirement: Translation glossary isolation
The system SHALL exclude translation-only glossary rules from ASR conditioning while preserving those rules for translation processing.

#### Scenario: Mixed meeting configuration is loaded
- **WHEN** a meeting hotword file contains both plain ASR terms and `source => target` translation rules
- **THEN** the ASR prompt contains only normalized plain terms and the translation glossary retains the valid translation rules

### Requirement: Consistent backend conditioning
The system SHALL pass the canonical bounded hotword prompt through supported recognition paths without compounding hotword bias across multiple recognition stages.

#### Scenario: Faster-whisper standard and batch paths are used
- **WHEN** equivalent connections are processed by standard faster-whisper inference and batch inference
- **THEN** both paths construct decoding input from the same canonical hotword prompt

#### Scenario: Backend rejects hotword input
- **WHEN** a supported backend model raises an argument compatibility error for hotword conditioning
- **THEN** the system retries recognition without hotwords and emits a warning without terminating the connection

### Requirement: Evidence-based hotword hallucination suppression
The system SHALL suppress hotword-dominated segment output only when recognition or audio evidence also indicates silence or weak speech.

#### Scenario: Hotword output occurs during weak audio
- **WHEN** a segment is wholly or predominantly composed of configured hotwords and its no-speech probability or audio-energy evidence crosses the hallucination threshold
- **THEN** the system excludes the segment from partial and completed transcript output and records the drop reason

#### Scenario: Spoken hotword has sufficient evidence
- **WHEN** a segment contains a configured hotword and speech evidence remains above the acceptance threshold
- **THEN** the system retains the segment and processes it through the normal transcript path

#### Scenario: Weak non-hotword speech is processed
- **WHEN** a weak segment is not wholly or predominantly composed of configured hotwords
- **THEN** the new hotword-specific guard does not reject it and existing general filtering rules remain authoritative

### Requirement: Hotword safety observability
The system SHALL expose bounded operational metadata for hotword normalization and hallucination decisions without logging complete hotword files.

#### Scenario: Connection activates hotwords
- **WHEN** a connection resolves and normalizes a hotword source
- **THEN** the system logs the source, original and accepted counts, truncation state, and a bounded preview

#### Scenario: Hotword hallucination is rejected
- **WHEN** the hotword-specific guard rejects a segment
- **THEN** the system logs the weak-evidence reason and a bounded output preview associated with the client
