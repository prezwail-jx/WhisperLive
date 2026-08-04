## ADDED Requirements

### Requirement: Connection initialization releases partial resources
The system SHALL release every ASR, translation, queue, thread, manager, and socket resource created for a connection when initialization fails before the connection becomes ready.

#### Scenario: ASR initialization fails after translation setup
- **WHEN** translation resources have been created and ASR client initialization fails
- **THEN** the system stops and joins the translation worker within its bounded cleanup period, releases the queue, does not register an active client, and closes the connection

#### Scenario: Meeting-log initialization fails after client construction
- **WHEN** ASR and optional translation clients exist but the meeting-log session cannot start or resume
- **THEN** the system releases the created clients and workers, does not send ready, and does not consume a client-capacity slot

#### Scenario: Ready delivery fails
- **WHEN** a connection is registered and sending ready metadata fails
- **THEN** the system initiates the same idempotent cleanup path used for an unexpected connection failure

### Requirement: Idle and stopped ASR workers have bounded lifecycle
The system SHALL avoid polling indefinitely before a client sends its first audio frame and SHALL make ASR worker shutdown observable with a bounded join.

#### Scenario: Idle client waits for first audio
- **WHEN** an ASR client has started but has not received any audio frame
- **THEN** its transcription worker waits on a synchronization signal instead of repeatedly polling the empty audio buffer

#### Scenario: Cleanup wakes an idle worker
- **WHEN** cleanup is requested for an ASR client waiting for its first audio frame
- **THEN** the worker is signaled, exits without transcription, and the bounded join reports completion

#### Scenario: In-flight worker exceeds join budget
- **WHEN** an ASR worker is still executing after the configured join budget
- **THEN** the system logs a bounded timeout diagnostic and continues connection cleanup without retaining the client as active

### Requirement: Disconnected Admin status is bounded and socket-free
The system SHALL retain a lightweight disconnected-client diagnostic snapshot for 600 seconds without retaining the closed WebSocket or runtime client objects.

#### Scenario: Client disconnects
- **WHEN** a client connection closes
- **THEN** the system removes the WebSocket-keyed active status and retains a socket-free disconnected snapshot with its disconnect time

#### Scenario: Disconnected status expires
- **WHEN** a disconnected snapshot is older than 600 seconds
- **THEN** the system removes it before returning the next Admin status snapshot

#### Scenario: Disconnect churn exceeds snapshot capacity
- **WHEN** disconnected snapshots exceed the configured maximum count before their TTL expires
- **THEN** the system evicts the oldest snapshots and keeps the retained count at or below the configured bound

### Requirement: Completed output survives browser delivery failure
The system SHALL persist completed source and translation output through server-side callbacks before attempting best-effort WebSocket delivery.

#### Scenario: Source delivery fails
- **WHEN** a completed source segment cannot be sent because the browser WebSocket is closed
- **THEN** the completed segment remains available to the meeting-log callback and the connection enters coordinated cleanup

#### Scenario: Translation delivery fails
- **WHEN** a completed translation segment cannot be sent because the browser WebSocket is closed
- **THEN** the completed segment remains available to the meeting-log callback and the translation worker stops through coordinated cleanup

### Requirement: Per-session failures do not mutate server configuration
The system MUST keep process-wide backend and default VAD configuration unchanged when one client requests options or encounters backend construction failure.

#### Scenario: Backend construction fails for one connection
- **WHEN** construction of a requested backend client fails
- **THEN** the system rejects only that connection and later connections retain the process startup backend configuration

#### Scenario: Session VAD option differs from default
- **WHEN** a client supplies a session-specific VAD option
- **THEN** the option applies only to that client and does not change the default used by subsequent clients
