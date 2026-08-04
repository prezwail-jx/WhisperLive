## ADDED Requirements

### Requirement: Batch inference admission is bounded
The system SHALL bound pending shared Faster-Whisper batch requests and return a recoverable failure instead of allowing unbounded queue growth.

#### Scenario: Batch queue is saturated
- **WHEN** a client submits a request while the configured batch queue capacity is full
- **THEN** submission fails within a bounded timeout with a diagnostic error and does not retain the request audio indefinitely

#### Scenario: Worker stops with queued requests
- **WHEN** the batch worker is stopped while requests are pending
- **THEN** every pending request is signaled with a stopped error and no waiting client remains blocked indefinitely

### Requirement: Timed-out and disconnected batch requests are cancelled
The system SHALL mark a batch request cancelled when its waiting ASR client times out or disconnects and SHALL skip cancelled work before processing.

#### Scenario: ASR wait times out
- **WHEN** an ASR client does not receive a batch result within its configured wait budget
- **THEN** the client marks its request cancelled and receives a recoverable transcription failure

#### Scenario: Worker receives cancelled request
- **WHEN** the worker collects a cancelled request before GPU preprocessing or decoding
- **THEN** it skips the request, signals completion, and does not execute model inference for that request

#### Scenario: Request is already executing
- **WHEN** cancellation occurs after a request has entered a GPU batch
- **THEN** the worker completes the in-flight batch safely and suppresses delivery of the cancelled request result

### Requirement: Batch decoding rejects low-quality fixed-temperature output
The system SHALL apply per-request quality checks and temperature fallback to prompt-compatible multi-request batches to prevent decoder runaway on short or silent audio.

#### Scenario: Initial batch decode is acceptable
- **WHEN** a request's initial decode satisfies configured compression, confidence, and no-speech quality checks
- **THEN** the system accepts that result without retrying the request

#### Scenario: Initial batch decode fails quality checks
- **WHEN** a request's initial decode fails configured compression or confidence quality checks and is not classified as silence
- **THEN** the system retries only that request with the configured temperature fallback sequence

#### Scenario: Low-confidence no-speech audio is detected
- **WHEN** a request has high no-speech probability and low confidence
- **THEN** the system accepts an empty result rather than producing a decoder-runaway transcript

### Requirement: Batch worker shutdown releases shared runtime state
The system SHALL stop the shared batch worker during server shutdown and clear its shared reference after the bounded worker join.

#### Scenario: Server shuts down
- **WHEN** the transcription server exits
- **THEN** it requests batch worker stop, waits for the configured bounded join, and clears the shared worker reference before process teardown
