## Why

The server creates ASR clients, translation clients, threads, queues, WebSocket status entries, and batch requests for every live connection, but not every error or disconnect path releases them. Over a long-running deployment, failed initialization, idle sockets, batch timeouts, and closed browsers can leave resources resident, increase CPU/GPU work, or permanently consume client capacity.

This must be addressed before further production rollout of long meetings and reconnect support so that connection churn has bounded memory, queue, and thread costs.

## What Changes

- Make ASR/translation session initialization transactional so every partially created resource is released when a later step fails.
- Add explicit ASR thread stop and bounded join behavior, including event-based waiting for an idle client’s first audio frame.
- Retain disconnected Admin client status as a lightweight snapshot for 10 minutes, with a bounded capacity, without retaining closed WebSocket objects.
- Add bounded BatchInferenceWorker submission, request cancellation, timeout handling, cancelled-request skipping, and orderly worker shutdown.
- Adapt upstream batch decoding quality fallback to prevent fixed-temperature decoder runaway in multi-request batches.
- Make source and translation delivery record completed segments before best-effort browser delivery, then trigger a single cleanup path when WebSocket sending fails.
- Prevent per-connection configuration or backend construction failure from mutating process-wide server behavior.
- Add non-audio runtime diagnostics and focused tests for cleanup, cancellation, status expiry, and failure recovery.

## Capabilities

### New Capabilities
- `runtime-resource-lifecycle`: Bounded ownership, cleanup, status retention, and failure recovery for live ASR and translation sessions.
- `batch-inference-resilience`: Bounded batch request lifecycle, cancellation, shutdown, and decoder quality fallback for shared Faster-Whisper inference.

### Modified Capabilities
- None.

## Impact

- Affected modules: `whisper_live/server.py`, `whisper_live/backend/base.py`, Faster-Whisper and translation backends, `whisper_live/batch_inference.py`, and Admin client-status handling.
- Affected tests: base backend, batch inference, Faster-Whisper backend, server extended, and translation backend tests.
- Affected runtime behavior: disconnected Admin status becomes temporary, saturated batch submission becomes a recoverable error, and WebSocket send failure actively initiates cleanup.
- No new model instance, GPU placement change, external dependency, database, or meeting-log storage format change is introduced.
