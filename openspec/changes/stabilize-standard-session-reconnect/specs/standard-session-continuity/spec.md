## ADDED Requirements

### Requirement: Standard sessions use stable GPU affinity
The system SHALL route the initial and every resumed `/ws-standard` WebSocket connection for one `session_id` to the same GPU backend.

#### Scenario: Transient disconnect reconnects to the assigned GPU
- **WHEN** a standard session disconnects and reconnects with its existing `session_id`
- **THEN** the WebSocket is routed to the same GPU backend that accepted the initial connection

#### Scenario: Different sessions remain distributable
- **WHEN** standard sessions are created with different random session IDs
- **THEN** the routing layer can distribute those sessions across GPU0 and GPU1 while preserving affinity within each session

### Requirement: Standard sessions do not silently fail over
The system MUST NOT route a resumed standard session to the other GPU backend when its assigned backend is unavailable.

#### Scenario: Assigned GPU is unavailable
- **WHEN** a standard session reconnects while its assigned GPU backend cannot accept the connection
- **THEN** the reconnect fails visibly and no connection for that session is opened on the other GPU backend

### Requirement: The browser continues a resumed session in place
The browser SHALL retain the current session identity and completed transcript rows during an unexpected disconnect and SHALL append resumed output on the same page after recovery.

#### Scenario: Successful in-page resume
- **WHEN** the browser receives a successful resume response for the current session
- **THEN** existing captions remain visible and new captions are appended without creating a new session or clearing completed rows

#### Scenario: Audio gap is visible in session metadata
- **WHEN** a disconnected session resumes successfully
- **THEN** the meeting log records the disconnected interval as an audio gap and increments the session connection count

### Requirement: Resumed output uses a continuous timeline
The server SHALL restore the persisted session timeline offset before emitting resumed source or translation segments.

#### Scenario: New captions follow pre-disconnect captions
- **WHEN** a session resumes after previously persisting completed segments
- **THEN** newly emitted segment timestamps are greater than or continuous with the persisted timeline and the browser displays them after earlier rows

### Requirement: Resume success is explicit and truthful
The server MUST send a ready response for a resumed connection only after the meeting-log session has been successfully loaded and resumed.

#### Scenario: Persisted session is resumed
- **WHEN** the requested session exists, is resumable, and belongs to the same browser instance
- **THEN** the server sends ready metadata containing the session ID, resumed status, connection count, and timeline offset

#### Scenario: Session resume fails
- **WHEN** the requested session is missing, finished, unreadable, or belongs to another browser instance
- **THEN** the server sends a machine-readable resume error, cleans up the failed connection, and does not claim that the session is ready

### Requirement: Stale connections cannot interrupt a newer resume
The system SHALL associate session cleanup with the connection generation that initiated it and SHALL ignore stale interruption writes from an older generation.

#### Scenario: Old cleanup completes after new resume
- **WHEN** a newer connection resumes a session before cleanup of the previous WebSocket completes
- **THEN** cleanup of the previous connection does not change the newer connection's active session status to interrupted

### Requirement: Production WebSockets have an eight-hour limit
Both production GPU backends SHALL use a per-WebSocket maximum connection time of `28800` seconds for standard and accurate routes.

#### Scenario: Meeting remains connected below eight hours
- **WHEN** a healthy WebSocket has been connected for less than `28800` seconds
- **THEN** the backend does not disconnect it for exceeding the configured connection duration

#### Scenario: Connection reaches the limit
- **WHEN** a standard WebSocket reaches `28800` seconds
- **THEN** the existing reconnect flow reuses the same session ID and routes the reconnect to the assigned GPU
