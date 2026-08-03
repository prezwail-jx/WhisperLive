## 1. Session State Safety

- [x] 1.1 Extend `MeetingLogStore` session start/resume results with a connection generation based on `connection_count` and attach it to each initialized server client.
- [x] 1.2 Make unexpected-disconnect interruption conditional on the cleanup generation still matching the session's active generation so stale WebSockets cannot interrupt a newer resume.
- [x] 1.3 Add `tests.test_meeting_logs` coverage for matching-generation interruption, stale-generation cleanup, and uninterrupted resumed state.

## 2. Truthful Resume Protocol

- [x] 2.1 Refactor WebSocket initialization so meeting-log start or resume must succeed before the server sends `SERVER_READY`.
- [x] 2.2 Return a stable machine-readable resume error for missing, finished, unreadable, or browser-mismatched sessions and clean up partially initialized ASR/translation resources.
- [x] 2.3 Preserve successful resume metadata including `session_id`, `connection_count`, `timeline_offset_seconds`, and audio-gap recording.
- [x] 2.4 Add `tests.test_server_extended` coverage proving failed resumes never send ready and successful resumes retain timeline and generation metadata.

## 3. Standard Session Affinity And Browser Continuity

- [x] 3.1 Add the current `session_id` as a routing query parameter when opening `/ws-standard`, reusing it for every automatic or manual continuation attempt.
- [x] 3.2 Handle machine-readable resume errors in the browser without clearing completed transcript rows or pretending the connection is healthy.
- [x] 3.3 Preserve current reconnect behavior so a successful resume appends offset source and translation segments on the existing page and a failed retry sequence exposes the interrupted-session actions.
- [x] 3.4 Update the frontend asset version in `web/index.html` so deployed browsers receive the affinity-aware client.

## 4. Interrupted Log Visibility And Completion

- [x] 4.1 Change the meeting-log selector to include `finished` and `interrupted` sessions, exclude `active` sessions, and display a clear status label for each option.
- [x] 4.2 Keep Markdown and DOCX downloads enabled for interrupted sessions while disabling transcript editing and summary generation until they are finished.
- [x] 4.3 Add an explicit, confirmed action that calls the existing finish endpoint for the selected interrupted session and refreshes its state before enabling editing and summaries.
- [x] 4.4 Ensure download and finish failures identify the selected session and remain visible in the summary drawer status.

## 5. Non-Accurate ASR Segmentation

- [x] 5.1 Add bounded segmentation diagnostics for completion reasons, ordinary low-energy drops, hallucination drops, short-fragment hold/merge/release, boundary dedupe, and realtime audio drops without writing raw audio.
- [x] 5.2 Preserve ordinary `min_segment_rms` at `0.028`, raise `same_output_threshold` toward `9`, and raise `max_incomplete_segment_seconds` toward `12.0` while keeping accurate mode unchanged.
- [x] 5.3 Preserve `min_transcription_chunk_seconds=2.5` and VAD `min_silence_duration_ms=900` for non-accurate modes so routine ASR inference frequency and VAD split behavior do not change in this pass.
- [x] 5.4 Preserve existing silence-hallucination and weak-evidence hotword filters alongside the unchanged ordinary RMS sensitivity.
- [x] 5.5 Implement non-accurate sentence-boundary completion only when strong punctuation, repeated text stability, minimum utterance duration, and trailing low-energy audio are all present.
- [x] 5.6 Implement bounded short-fragment coalescing before client/log/translation emission, merging only compatible language/speaker/timing continuations and releasing after the hold window expires.
- [x] 5.7 Add focused backend tests for unchanged ordinary low-energy filtering, silence hallucination rejection, repeat threshold behavior, duration-limit fallback, punctuation-without-silence not completing, stability-plus-silence completion, compatible short-fragment merge, terminal-boundary no-merge, and hold-window release.
- [x] 5.8 Apply the conservative segmentation defaults to all non-accurate Faster-Whisper sessions: hold short fragments for up to 2.5 seconds and require 250ms of newly buffered audio before repeat inference while bypassing the interval during finalization.
- [x] 5.9 Add focused tests for conservative hold/merge behavior, high-accuracy exclusion, new-audio throttling, and finalization bypass.

## 6. Historical Log Migration

- [x] 6.1 Add a dependency-free migration utility that inventories two meeting-log trees by `session_id`, validates JSON records, and supports a dry-run report.
- [x] 6.2 Make the migration utility copy complete non-conflicting session file sets into a staging/shared directory without overwriting duplicate or malformed records.
- [x] 6.3 Add focused tests for unique-session copying, identical duplicates, differing duplicate conflicts, malformed JSON quarantine/reporting, and companion Markdown/summary preservation.
- [x] 6.4 Document backup, dry-run, conflict review, inventory comparison, representative download checks, and the limitation that never-persisted post-disconnect content cannot be recovered.

## 7. Production Routing And Runtime Configuration

- [ ] 7.1 Document and apply an Nginx `/ws-standard` upstream keyed by `session_id`, with a legacy fallback key, strict no-cross-GPU retry behavior, and unchanged fixed `/ws-accurate` routing.
- [ ] 7.2 Update maintained startup scripts and production examples so both GPU backends use `--max_connection_time 28800` and the same mounted `--meeting_logs_dir`.
- [ ] 7.3 During deployment preflight, inspect the actual ignored Nginx file, container mounts, process arguments, log source paths, and active client counts instead of assuming repository example paths.
- [ ] 7.4 With zero active meetings and explicit restart authorization, back up both source directories, run the migration into shared storage, update both mounts, run `nginx -t`, and restart/reload only the required existing services.

## 8. Verification And Delivery

- [ ] 8.1 In the deployment container, run `py_compile` for each changed Python module and at most the directly affected meeting-log, server, and ASR segmentation test modules.
- [ ] 8.2 In the deployment container, run `node --check web/app.js` and validate the actual Nginx configuration with `nginx -t`.
- [ ] 8.3 Verify a GPU0 standard session and a GPU1 standard session each reconnect to their original GPU, keep prior captions, append increasing timestamps, and record one audio gap plus an incremented connection count.
- [ ] 8.4 Verify an unavailable assigned GPU does not route that session to the other GPU and the browser reports interruption instead of a false ready state.
- [ ] 8.5 Verify unified listing and Markdown/DOCX download for representative finished and interrupted logs from both GPUs, then finish one interrupted log and confirm editing/summary controls become available.
- [ ] 8.6 Verify non-accurate Chinese, English, and mixed-language nearfield sessions produce fewer short fragments without increased `REALTIME_DROP`, without routine extra ASR passes, and without a noticeable increase in live caption latency beyond the bounded coalescing delay.
- [ ] 8.7 Compare segmentation diagnostics before and after tuning for low-energy drops, completion reasons, short-fragment merges, and duration-limit completions.
- [ ] 8.8 Run `openspec validate stabilize-standard-session-reconnect`, `git diff --check`, `git status --short`, and review the complete related diff before delivery.
