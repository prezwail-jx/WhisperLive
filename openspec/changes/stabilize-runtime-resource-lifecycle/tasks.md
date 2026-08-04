## 1. Connection Lifecycle Ownership

- [x] 1.1 Map per-connection ASR, translation, queue, meeting-log, manager, and socket ownership into one idempotent server cleanup coordinator.
- [x] 1.2 Route ASR construction, translation construction, meeting-log start/resume, manager registration, and ready-message failures through the coordinator without sending a false ready response.
- [x] 1.3 Ensure cleanup removes partially registered manager entries and preserves existing meeting-log generation protection when interrupting a failed connection.
- [ ] 1.4 Add focused server tests for ASR setup failure after translation setup, meeting-log initialization failure, and ready-message delivery failure.

## 2. ASR Worker Lifecycle

- [x] 2.1 Add first-frame event waiting to `ServeClientBase` and signal it from successful frame arrival and cleanup.
- [x] 2.2 Add a common bounded ASR stop/join contract that wakes idle workers and reports join timeout diagnostics without blocking other cleanup.
- [x] 2.3 Adapt Faster-Whisper, FunASR, TensorRT, OpenVINO, and MLX startup paths to the common lifecycle contract without changing inference configuration.
- [ ] 2.4 Add backend tests proving an idle worker does not poll, cleanup wakes it without transcription, and an in-flight join timeout leaves no active client slot.

## 3. Admin Status Retention

- [x] 3.1 Split live WebSocket-keyed client status from socket-free recent-disconnected snapshots in `ClientManager`.
- [x] 3.2 Implement 600-second expiry and a bounded recent-snapshot capacity, cleaning expired entries on registration, status reads, and explicit deletion.
- [x] 3.3 Preserve existing Admin status fields needed for short-term diagnostics while excluding WebSocket, client, queue, and audio references.
- [x] 3.4 Add tests for disconnect snapshot creation, WebSocket reference removal, TTL expiry, capacity eviction, and same-instance replacement.

## 4. Batch Request Backpressure And Shutdown

- [x] 4.1 Add bounded BatchInferenceWorker queue capacity derived from configured concurrency and return a recoverable submission error when saturated.
- [x] 4.2 Add request cancellation/completion state; make Faster-Whisper check batch wait timeout and cancel requests on timeout or client cleanup.
- [x] 4.3 Skip cancelled requests during queue collection and preprocessing, and suppress result delivery for requests cancelled after entering a batch.
- [x] 4.4 Make worker shutdown signal all queued waiters, perform bounded join, and clear the shared Faster-Whisper worker reference during server shutdown.
- [ ] 4.5 Add batch tests for queue saturation, wait timeout, cancelled-request skipping, in-flight cancellation safety, stopped worker draining, and shared-reference reset.

## 5. Batch Decode Quality Recovery

- [x] 5.1 Adapt upstream per-request compression, confidence, and no-speech quality gates while retaining local prompt and hotword compatibility grouping.
- [x] 5.2 Implement temperature fallback that retries only failed non-silence requests and uses lower-cost decoding for nonzero temperatures.
- [ ] 5.3 Add batch tests for accepted initial output, fallback selection, silence acceptance, and preservation of valid sibling results.
- [ ] 5.4 Validate multi-client batch behavior in the deployment container without adding model instances or increasing configured batch capacity.

## 6. Delivery Failure And Configuration Isolation

- [x] 6.1 Order source and translation completed-segment callbacks so meeting-log/Admin persistence occurs before best-effort browser WebSocket delivery.
- [x] 6.2 Detect source and translation delivery failure, schedule the connection cleanup coordinator once, and prevent continued unbounded work for the closed socket.
- [x] 6.3 Remove per-connection mutations of process-wide backend and default VAD configuration; scope fallback decisions to the failing connection.
- [ ] 6.4 Add tests proving send failure preserves completed log callbacks, triggers one cleanup transition, and does not alter subsequent client backend/VAD behavior.

## 7. Runtime Diagnostics And Deployment Verification

- [x] 7.1 Add bounded diagnostics or metrics for initialization rollback, cleanup duration, ASR join timeout, active/recent status counts, batch rejection/cancellation/timeout, and delivery failure.
- [ ] 7.2 In the deployment container, run Python syntax checks and the directly affected base backend, batch inference, Faster-Whisper backend, server extended, and translation backend test modules.
- [ ] 7.3 Establish deployment baseline for RSS, Python threads, active/recent status counts, batch queue depth, translation queue depth, and `REALTIME_DROP` before validation.
- [ ] 7.4 Run repeated connection churn, idle connection, normal finish, browser offline/reconnect, initialization failure, ready-send failure, and batch timeout scenarios; verify resources return near baseline after cleanup and TTL expiry.
- [ ] 7.5 Run sustained multi-client validation for 30-60 minutes; compare latency, GPU behavior, queue depth, RSS, and thread count against baseline, then record the chosen batch capacity and join timeout.
- [x] 7.6 Run `openspec validate stabilize-runtime-resource-lifecycle`, `git diff --check`, `git status --short`, and review the complete related diff before delivery.
