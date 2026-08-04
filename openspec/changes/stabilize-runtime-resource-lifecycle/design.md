## Context

WhisperLive has evolved from a single ASR loop into a per-WebSocket runtime containing an ASR client and thread, optional translation client and thread, bounded translation queue, meeting-log writer, Admin status entry, and optional shared batch-inference request. The browser reconnect path and file-backed meeting logs deliberately preserve a session across transient disconnects, but that does not make partially initialized resources or closed WebSocket references safe to retain.

The current runtime has several independent cleanup paths. Some handle a failed meeting-log start, while ASR construction, ready-message delivery, and manager registration failures can bypass equivalent cleanup. ASR workers poll while idle, normal cleanup does not wait for ASR threads, disconnected status retains WebSocket keys indefinitely, and a timed-out batch request can still consume GPU work. The upstream project provides proven patterns for first-frame event waiting, propagated client send failures, and batch decoding quality fallback, but its implementation must be adapted to local translation, hotword, reconnect, and batch prompt-group behavior.

Constraints:

- Preserve session-affinity, resume generation protection, meeting-log schema, model instances, GPU placement, and translation device choices.
- Do not add a model, database, external queue, or background service.
- Keep audio buffers and translation queues bounded; do not persist raw audio for diagnostics.
- Release runtime resources in bounded time without allowing one stuck inference to block cleanup of unrelated clients.
- Run Python tests and service validation only in the existing deployment container.

## Goals / Non-Goals

**Goals:**

- Give each accepted connection one explicit, idempotent lifecycle owner.
- Release partially initialized ASR/translation resources on every failure path.
- Eliminate idle ASR polling and make stopped ASR threads observable with bounded joins.
- Retain disconnected Admin diagnostics for 10 minutes without retaining closed WebSocket objects or growing unboundedly.
- Bound batch request admission and ensure callers can cancel or time out without stale GPU work accumulating.
- Preserve completed server-side segments when browser delivery fails, then converge on one cleanup path.
- Prevent per-session errors from changing process-wide backend or VAD behavior.
- Emit bounded diagnostics for resource cleanup and backpressure decisions.

**Non-Goals:**

- Do not redesign `MeetingLogStore` caching, add an event journal, or change JSON/Markdown persistence; that is a separate second-stage change.
- Do not guarantee forceful cancellation of an inference already executing inside CTranslate2 or a translation model.
- Do not alter accurate-mode ASR segmentation, translation quality policy, or browser reconnect timing.
- Do not expose new user-facing controls or change meeting-log download/edit semantics.

## Decisions

### 1. Use a connection-owned runtime cleanup coordinator

`initialize_client()` will track every resource it creates and transfer ownership only after the client manager registration succeeds. A single idempotent cleanup coordinator will stop translation, request ASR stop, cancel outstanding batch work, remove manager entries, finalize or interrupt the meeting log according to the existing connection generation, and close the socket when appropriate.

The coordinator will be used for initialization failure, ready-message send failure, receive-loop exit, Admin-initiated disconnect, and WebSocket send failure. Cleanup will be safe if invoked concurrently or twice; the first caller performs actions and later callers observe completion.

Creating translation before ASR remains acceptable only when the same coordinator owns both from the first allocation. Reordering all construction was rejected because ASR and translation setup each depend on different session options and failures must be handled regardless of order.

### 2. Stop ASR cooperatively and join only for a bounded period

`ServeClientBase` will use a first-frame event. The ASR thread waits on this event while no audio has arrived and cleanup signals it together with the exit flag. Each backend retains its current client thread model but exposes a common stop/join contract. Server cleanup joins an ASR thread only for a finite timeout and logs a diagnostic if it remains alive.

Threads will not be made daemon-only as a replacement for cleanup: daemon threads hide leaks and can abandon finalization. A forceful Python thread kill was rejected because it is unsafe around GPU/native code. A stuck in-flight inference may outlive the join timeout, but it must no longer retain an active manager slot or block other cleanup.

### 3. Separate active Admin status from recent disconnected snapshots

`ClientManager` will store active status with the live WebSocket only while connected. On disconnect, it will copy allowed diagnostic fields into a lightweight recent-disconnected record keyed by stable client identity, remove the WebSocket-keyed entry, and retain that snapshot for 600 seconds. Cleanup runs on registration, status snapshot reads, and explicit deletion; a fixed maximum record count prevents churn from exceeding memory even before TTL expiry.

The recent snapshot deliberately excludes WebSocket, ASR client, translation client, queues, and audio. Finished and interrupted meeting history remains the responsibility of `MeetingLogStore`, not Admin status.

### 4. Make BatchInferenceWorker admission and cancellation explicit

Batch requests will have a cancellation state and an explicit completion result. Submission uses a bounded queue and timeout. The waiting ASR caller checks the `Event.wait()` result; a timeout cancels its request and returns a recoverable transcription error. Worker collection and preprocessing discard cancelled requests and signal stopped or rejected requests so no caller waits forever. Worker shutdown drains pending requests with a stopped error, joins the daemon worker, and clears the shared worker reference.

The queue capacity will be derived from configured concurrent-client capacity with a small bounded multiplier rather than an arbitrary unbounded queue. Cancellation cannot interrupt an already executing GPU call, so the worker checks cancellation before CPU preprocessing and before each GPU batch. This minimizes stale work without claiming impossible immediate GPU cancellation.

### 5. Adapt upstream quality fallback inside local prompt-compatible batches

The multi-request batch decoder will retain local grouping by initial prompt and hotwords. Within each compatible group, it will evaluate output quality per request using compression ratio, average log probability, and no-speech probability, retry only failed items with the upstream temperature progression, and accept empty output for high-no-speech low-confidence audio. Retry decoding uses a lower-cost beam configuration for nonzero temperatures.

Directly replacing the local batch decoder was rejected because it would lose prompt compatibility grouping and local language/hotword behavior. Retrying every item together was rejected because one low-quality request should not recompute valid requests.

### 6. Persist completed output before best-effort browser delivery

For both source and translation completed segments, server-side post-processing, meeting-log/Admin callbacks, and deduplication occur before browser WebSocket delivery. A delivery failure marks the connection failed and schedules the coordinator exactly once. The system does not retry arbitrary completed messages after a closed WebSocket; reconnect continues from persisted state under existing resume rules.

Persisting after browser delivery was rejected because a dead browser must not decide whether a completed server result reaches the meeting log. Synchronous socket-close cleanup from an ASR thread was rejected to avoid recursive cleanup and lock contention; the coordinator will serialize the transition.

### 7. Keep server configuration immutable per process

Backend selection and default VAD policy remain process configuration set at startup. Per-client options are passed to the client constructor or local variables. Backend construction failure rejects only that connection and never changes `self.backend` or `self.use_vad` for future sessions.

Fallback by mutating global backend state was rejected because it causes unrelated sessions to run on a different backend after one failed connection.

### 8. Instrument lifecycle boundaries without audio payloads

Logs and metrics will record initialization rollback reason, cleanup cause and duration, ASR join timeout, active/recent status counts, batch queue rejection/cancellation/timeout, and delivery failure. Text previews remain subject to existing bounded segmentation diagnostics; no raw samples, audio files, or unbounded exception payloads are recorded.

## Risks / Trade-offs

- [Risk] A 10-minute status snapshot can omit older Admin debugging context. -> Meeting logs remain durable; cap snapshots and document the diagnostic retention window.
- [Risk] ASR join timeout can leave a native inference temporarily alive. -> Detach it from active capacity and log the condition; validate that repeated churn does not cause sustained thread growth.
- [Risk] Bounded batch admission can reject work during overload. -> Return a recoverable error, retain existing pending-audio bounds, and expose queue diagnostics rather than silently growing memory.
- [Risk] Temperature fallback adds GPU work for low-quality batch items. -> Retry only failing items and use lower-cost nonzero-temperature decoding.
- [Risk] Persist-before-send can write a segment the disconnected browser never displayed. -> This is correct server truth; a resumed session can retrieve it from persisted state.
- [Risk] Coordinated cleanup can race old and resumed sockets. -> Continue using meeting-log connection generation checks and make coordinator state connection-local.

## Migration Plan

1. Add focused unit tests for each lifecycle state before enabling deployment validation.
2. Deploy to one existing GPU service during a quiet period with current meeting-log mount and model settings unchanged.
3. Record baseline RSS, Python thread count, active/recent Admin counts, batch queue depth, and translation queue depth.
4. Exercise normal finish, browser offline/reconnect, ready-send failure, ASR initialization failure, translation initialization failure, and batch timeout paths.
5. Run repeated connection churn and sustained multi-client tests in the deployment container; compare metrics to baseline after cleanup windows expire.
6. Roll back by restoring the prior application revision. Existing meeting logs remain compatible because this change does not modify their schema.

## Open Questions

- The exact batch queue multiplier and ASR join timeout must be tuned from deployment-container measurements while preserving current latency and `REALTIME_DROP` behavior.
- The deployment validation must determine whether batch temperature fallback improves observed hallucination cases without unacceptable GPU latency under the configured batch size.
