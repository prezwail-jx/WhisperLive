## ADDED Requirements

### Requirement: Both GPU backends use shared meeting-log storage
GPU0 and GPU1 SHALL read and write meeting logs through the same mounted meeting-log directory while retaining session-affinity ownership for active writes.

#### Scenario: GPU1 log is visible through the unified Admin route
- **WHEN** GPU1 finishes or interrupts a session and persists its log
- **THEN** the Admin node discovers that session from shared storage without requiring an ASR service restart

#### Scenario: GPU0 log is visible through the unified Admin route
- **WHEN** GPU0 finishes or interrupts a session and persists its log
- **THEN** the same unified meeting-log API can list and download that session

### Requirement: User log selection includes persisted interrupted sessions
The browser SHALL list finished and interrupted meeting-log sessions, SHALL label their status, and MUST exclude active sessions from the user log selector.

#### Scenario: Interrupted session is listed
- **WHEN** the meeting-log API returns a persisted session with status `interrupted`
- **THEN** the browser includes it in the meeting selector with an interrupted status label

#### Scenario: Active session is hidden
- **WHEN** the meeting-log API returns a session with status `active`
- **THEN** the browser does not include it in the meeting selector used for download, editing, or summary generation

### Requirement: Persisted interrupted logs are downloadable
The browser SHALL allow users to download the content already persisted for an interrupted session without representing missing audio or unwritten segments as recovered.

#### Scenario: Download interrupted Markdown log
- **WHEN** a user selects an interrupted session and requests its Markdown log
- **THEN** the server returns the persisted log content and the browser preserves the interrupted status indication

#### Scenario: Download interrupted DOCX log
- **WHEN** a user selects an interrupted session and requests an available DOCX layout
- **THEN** the server generates or returns the document from persisted source and translation segments

### Requirement: Interrupted sessions require explicit finish before post-processing
The system SHALL permit an interrupted session to be explicitly marked finished, and SHALL require finished status before transcript editing or summary generation.

#### Scenario: User finishes an interrupted session
- **WHEN** a user explicitly confirms finishing the selected interrupted session
- **THEN** the system changes its status to `finished` and enables the normal editing and summary workflow

#### Scenario: Interrupted session remains unfinalized
- **WHEN** an interrupted session has not been explicitly finished
- **THEN** log download remains available while transcript editing and summary generation remain disabled

### Requirement: Historical logs migrate without silent overwrite
The deployment migration MUST back up both source directories, inventory records by `session_id`, copy non-conflicting records into shared storage, and stop automatic overwrite for conflicting session IDs.

#### Scenario: Session exists in only one source directory
- **WHEN** a valid session ID and its companion files occur in only one backend's source log directory
- **THEN** migration copies the complete session file set into the shared directory

#### Scenario: Duplicate session ID differs between sources
- **WHEN** the same session ID has different payloads or companion files in both source directories
- **THEN** migration reports the conflict and leaves both source copies backed up for manual reconciliation without silently choosing one

#### Scenario: Migrated logs are validated
- **WHEN** unique sessions have been copied to shared storage
- **THEN** deployment verification compares inventory counts and confirms representative list and download operations before using the shared directory in production
