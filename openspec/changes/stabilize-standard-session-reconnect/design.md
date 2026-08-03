## Context

The browser creates one `session_id` when capture starts and reuses it when reconnecting, but `/ws-standard` currently load-balances each WebSocket independently. Accurate mode appears stable because `/ws-accurate` is pinned to GPU0, while a standard session can reconnect from GPU0 to GPU1.

Each backend currently owns an independent in-memory `MeetingLogStore` and may use a container-local `logs` directory. A resume on the other backend therefore raises `meeting log session not found`. `initialize_client()` catches that exception, still sends `SERVER_READY`, and assigns the missing session ID to the client. ASR can continue with a timeline starting at zero, completed log appends are discarded because no record exists, and the browser's time-sorted last rows can hide the new zero-based segments behind older rows.

The browser log selector also filters out every status except `finished`. An interrupted JSON log can exist and be downloadable through the API while remaining invisible in the user interface. Production routing and mounts are environment-specific under the Git-ignored `deploy/` directory, so deployment instructions and acceptance checks are part of the change.

Non-accurate modes also share one faster-whisper segmentation profile. The browser currently sends `min_segment_rms=0.028`, `same_output_threshold=7`, `max_incomplete_segment_seconds=10.0`, `sentence_completion_min_seconds=0.0`, `min_transcription_chunk_seconds=2.5`, and VAD silence duration `900ms`. This disables punctuation-based completion and relies on Whisper natural multi-segment output or repeat/time-limit completion, which can fragment nearfield speech.

Constraints and confirmed product decisions:

- A standard session must reconnect only to its originally selected GPU backend.
- If that GPU is unavailable, the session must report interruption rather than fail over to the other GPU.
- Finished and interrupted logs must be visible; active sessions remain excluded from the user log selector.
- Both backends use one shared meeting-log directory for unified discovery and download.
- The production per-WebSocket connection limit is eight hours (`28800` seconds).
- Non-accurate modes include standard interpretation, conversation translation, and transcription-only mode; accurate mode remains out of scope for segmentation tuning.
- V2 short-fragment coalescing may add at most a bounded sub-second delay. An explicitly selected V3 profile may hold short fragments for up to 2.5 seconds and throttle repeat inference until 250ms of new audio arrives.
- Low-confidence mixed-language retry is not enabled in this change.
- The change must not alter models, GPU placement, translation devices, batches, or persistent log schema compatibility. V3 may reduce repeat ASR inference frequency through its explicit 250ms new-audio interval.

## Goals / Non-Goals

**Goals:**

- Deterministically bind every `/ws-standard` connection for one session to one GPU backend.
- Preserve the current page, session identity, and rendered transcript during transient reconnects.
- Resume with a persisted timeline offset so new captions and log segments continue after earlier content.
- Prevent stale connection cleanup from changing the state of a newer resumed connection.
- Fail visibly when resume cannot be completed instead of presenting a false ready state.
- Make persisted logs from both backends discoverable and downloadable through the unified Admin route.
- Expose interrupted logs and provide an explicit transition to finished before editing or summarization.
- Migrate existing logs without overwriting conflicting sessions.
- Improve non-accurate ASR readability by reducing short fragmented rows, low-volume word drops, and unnatural punctuation boundaries.
- Add diagnostics that explain why a source segment completed, merged, or was dropped without saving audio.

**Non-Goals:**

- Do not provide cross-GPU session failover when the assigned backend is unavailable.
- Do not recover audio that was never delivered during a disconnect or transcript segments that were never persisted.
- Do not persist live browser transcript state for restoration after a full page reload or browser process loss.
- Do not change accurate-mode segmentation behavior.
- Do not add routine second-pass ASR inference for mixed-language uncertainty.
- Do not change translation, diarization, model loading, GPU memory usage, or batch behavior.
- Do not introduce a database, distributed lock service, or new dependency.

## Decisions

1. Use `session_id` as the standard-route affinity key.

   The browser already creates `currentSessionId` before opening the WebSocket. It will append that value as a query parameter to `/ws-standard`; reconnects reuse the same value. Nginx will consistently hash that key across GPU0 and GPU1. A session UUID distributes new sessions without coupling routing to IP addresses, cookies, or the per-connection `uid`.

   IP hashing was rejected because users behind one NAT would be concentrated on one GPU. A cookie was rejected because the existing session identifier already has the required lifecycle and avoids another browser state mechanism.

2. Disable standard-route cross-backend retry.

   The standard upstream will disable proxy retry and passive remapping for an assigned peer so an unavailable GPU produces a connection failure instead of silently opening the same session on the other GPU. The Nginx configuration will define a safe fallback affinity key for legacy requests without `session_id`, but the maintained browser client will always send it. `/ws-accurate` remains pinned to GPU0.

   Automatic failover was rejected because the user explicitly requires same-GPU continuation and the current file-backed store has no distributed writer coordination suitable for seamless active-active takeover.

3. Share meeting-log storage for discovery, not active cross-GPU writing.

   Both backends will mount the same host directory at the same container path and pass that path through `--meeting_logs_dir`. `MeetingLogStore.refresh_sessions()` already discovers externally written JSON records, allowing the Admin node to list and download logs created by either GPU. Session affinity ensures only the assigned ASR backend appends to an active session.

   Keeping separate directories and aggregating HTTP responses was rejected because download, finish, transcript editing, and summary APIs would still require per-session request routing and duplicate Admin orchestration.

4. Treat log resume as a readiness barrier.

   A reconnect cannot send `SERVER_READY` until `resume_session()` succeeds and returns the persisted status, connection count, and timeline offset. A missing, finished, mismatched, or unreadable session returns a machine-readable resume error and closes that socket after cleaning up any partially initialized client resources. The browser keeps its existing transcript, reports the failure, and follows the existing bounded reconnect/interruption flow.

   Continuing after a failed resume was rejected because it creates a false healthy state, zero-based caption ordering, and unpersisted segments.

5. Guard session state with a connection generation.

   Each successful start or resume has a monotonically increasing connection generation derived from `connection_count` and attached to the server-side client. Unexpected cleanup may mark a session interrupted only when its generation still matches the active generation in the log record. This prevents delayed cleanup from an older WebSocket from overwriting a newer resumed session.

   Relying only on operation order was rejected because socket close and reconnect run concurrently and cannot guarantee that old cleanup completes before new initialization.

6. Preserve in-page transcript state and continue on the persisted timeline.

   Reconnect does not clear completed source or translation stores. On successful resume, the server-provided timeline offset remains the authority for segment timestamps, causing new segments to sort after pre-disconnect content. The UI reports reconnect attempts and notes that audio during the gap was not recorded. A new manual Start remains the only action that creates a new session and clears transcript state.

7. Include interrupted sessions in the meeting-log workflow.

   The selector will include `finished` and `interrupted` records, label each status, and keep `active` records hidden. Both statuses can download their persisted Markdown or DOCX content. An interrupted selection can call the existing finish endpoint after explicit user action; only after it becomes `finished` can transcript editing and summary generation proceed.

8. Standardize the production connection limit at eight hours.

   GPU0 and GPU1 startup commands will both pass `--max_connection_time 28800`. The value remains a per-WebSocket limit. If it is reached, the existing reconnect path uses the same session affinity key and resumes on the assigned GPU. Nginx WebSocket timeouts must be at least as long as the intended connection duration.

9. Migrate logs through a validated staging directory.

   Deployment first inventories both source directories by `session_id`, validates JSON records, and creates backups. Sessions present in only one source are copied to staging with their meeting subdirectory and companion Markdown/summary files. Duplicate `session_id` values are reported and not overwritten automatically; they require segment/status comparison before one canonical record is selected or merged. The completed staging tree becomes the shared directory only after inventory counts and sample downloads pass.

10. Tune non-accurate completion thresholds conservatively.

   Non-accurate clients will keep `min_segment_rms=0.028`, `min_transcription_chunk_seconds=2.5`, and VAD silence duration `900ms` to avoid changing low-energy filtering, routine GPU inference frequency, or too many timing variables at once. Repeated-output completion becomes less eager, moving from `7` toward `9`, and duration-limit completion moves from `10s` toward `12s` to reduce premature fixed-window cuts.

   Lowering chunk size was rejected for this change because it would increase the number of ASR requests per active session. Lowering VAD silence duration was rejected for the first pass because fragmentation is already a concern and VAD changes can create new split points.

11. Add conservative sentence completion based on stability and trailing silence.

   Non-accurate modes will not complete merely because Whisper produced a sentence-ending punctuation mark. Completion requires strong punctuation, at least two stable incomplete observations, a minimum utterance duration of about `3s`, and roughly `600ms` of low-energy trailing audio from the original buffered chunk. This uses existing audio already in memory and does not save audio to disk.

   Enabling the existing punctuation-only completion was rejected because Whisper can insert punctuation early or unnaturally, especially in mixed Chinese/English speech. Stability plus trailing silence makes the boundary closer to a real pause.

12. Coalesce compatible short source fragments with bounded latency.

   Before sending a very short completed source segment to the client, translation queue, and meeting log, the backend may hold it for up to about `700ms`. If the next completed fragment is compatible by language, speaker, and timing gap, and neither fragment has a strong terminal boundary such as a question or exclamation, the system merges them into one source segment and cleans unnatural intermediate punctuation. If no compatible continuation appears before the delay, the original fragment is emitted unchanged.

   Frontend-only coalescing was rejected because meeting logs and translation should receive the same stable source units users see. Indefinite buffering was rejected because live captions and translation latency must remain bounded.

13. Add segmentation diagnostics before and after tuning.

   The backend will log bounded, non-audio diagnostic events for low-energy drops, completion reasons, trailing-silence sentence completion, short-fragment hold/merge/release, boundary dedupe, and realtime audio drops. These logs are used to tune thresholds during deployment validation and to explain remaining fragmentation or missing words.

14. Add an explicit V3 coalescing and inference-cadence profile.

    `--standard_segmentation_profile v3` retains V2's completion thresholds and high-accuracy exclusion, but increases the short completed-fragment hold from 700ms to 2.5s. It also requires at least 250ms of newly buffered audio after the prior inference window before another normal inference call. Finalization bypasses the interval so a disconnect or explicit stop cannot leave tail audio unprocessed. `legacy` and `v2` retain their existing behavior for rollback and comparison.

    Applying the longer hold to V2 was rejected because it changes the already deployed tuning profile. A wall-clock throttle was rejected because inference latency varies; counting newly buffered audio directly prevents duplicate inference without delaying when actual audio accumulates.

## Risks / Trade-offs

- [Risk] Strict affinity reduces availability when one GPU backend is down. -> Keep the session on the assigned GPU by design, show a visible interruption, and allow manual continuation after that backend returns.
- [Risk] Nginx passive failure handling could remap a consistent hash to another peer. -> Disable upstream retry/passive peer suppression for this route and verify the actual production configuration with repeated reconnect tests.
- [Risk] Old and new sockets can overlap briefly. -> Use connection-generation checks so stale cleanup cannot interrupt the resumed generation.
- [Risk] A shared filesystem does not provide cross-process transactional locking. -> Keep active writes on one affinity-selected backend and limit the other backend/Admin process to refreshed reads or explicit post-session operations.
- [Risk] Historical directories can contain duplicate or malformed session files. -> Back up first, validate by `session_id`, quarantine invalid records, and never overwrite conflicts automatically.
- [Risk] Interrupted downloads can contain only the segments persisted before failure. -> Label the status and document that missing audio or never-written segments cannot be reconstructed.
- [Risk] An eight-hour socket can retain server resources for a long period. -> Preserve `max_clients`, client cleanup, and Admin visibility; do not remove the limit entirely.
- [Risk] Session IDs appear in proxy URLs and access logs. -> Treat the random UUID as a routing identifier rather than an authorization token and avoid logging query strings where production policy requires it.
- [Risk] Short-fragment coalescing adds latency. -> Bound the hold to about `700ms`, release immediately on strong terminal boundaries, and verify user-visible delay remains acceptable.
- [Risk] V3 can delay a short completed source row by up to `2.5s`. -> Make V3 opt-in, retain V2 and legacy for rollback, and verify translation latency and `REALTIME_DROP` during deployment.
- [Risk] Merging fragments can combine text that should remain separate. -> Require compatible language/speaker/timing, avoid merges across strong terminal punctuation, and preserve the original unmerged behavior when no safe continuation appears.
- [Risk] Sentence completion can still be fooled by model punctuation. -> Require text stability, minimum duration, and trailing low-energy audio rather than punctuation alone.

## Migration Plan

1. Inspect both containers' mounts, running `run_server.py` arguments, Nginx upstream behavior, active client counts, and source log directories.
2. Wait for zero active meetings. Back up both log directories and inventory valid JSON records by `session_id`, status, update time, and companion files.
3. Build and verify a shared staging directory. Copy unique sessions, quarantine malformed files, and stop for manual review if duplicate session IDs differ.
4. Deploy backend and frontend changes, including a frontend asset version update so maintained clients send the routing key.
5. Mount the shared directory into both GPU containers and start both services with the identical `--meeting_logs_dir` and `--max_connection_time 28800` values.
6. Update the actual production Nginx configuration for strict `/ws-standard` session affinity, retain fixed `/ws-accurate` routing, and run `nginx -t` before reload.
7. Verify one new session initially assigned to GPU0 and one to GPU1. For each, interrupt the WebSocket, confirm reconnect remains on the same GPU, captions continue with increasing timestamps, and one log records the gap and incremented connection count.
8. Confirm finished and interrupted logs from both GPUs appear through the unified user page and can be downloaded. Confirm interrupted logs require finishing before editing or summarization.
9. Validate non-accurate segmentation with nearfield Chinese, English, and mixed-language speech. Compare fragmentation, low-energy drops, completion reasons, perceived punctuation quality, translation latency, and `REALTIME_DROP` before and after tuning.
10. To roll back application or Nginx behavior, restore the prior code/configuration while retaining the shared directory as the source of truth. Do not copy newer shared logs back over old per-GPU backups.

## Open Questions

- No product decisions remain open. The concrete host path, container mount path, source directories, and active Nginx file must be discovered during deployment preflight rather than assumed in repository code.
