## Why

Standard-mode WebSocket reconnects can be routed to a different GPU backend while the two backends use separate meeting-log directories. The new backend then cannot resume the session, yet the browser may still appear connected while no new captions are visible and the session log cannot be found from the unified Admin endpoint. In the same non-accurate modes, current segmentation can also produce short fragmented rows, unnatural punctuation boundaries, and low-volume word drops, making normal recognition harder to follow even when the connection is healthy.

## What Changes

- Route every standard-mode reconnect for a session back to the same GPU backend selected for its initial connection, with no silent cross-GPU fallback when that backend is unavailable.
- Keep the current browser session and rendered captions across transient disconnects, then continue appending captions on the same page after a successful resume.
- Reject failed session resumes explicitly instead of sending a misleading ready state.
- Configure both GPU backends to use one shared meeting-log directory and migrate existing non-conflicting logs from both backend-local directories.
- Show both finished and interrupted sessions in the browser meeting-log selector, label their status, and allow persisted interrupted logs to be downloaded.
- Set both production backends to an eight-hour per-WebSocket connection limit (`28800` seconds).
- Improve non-accurate ASR segmentation by reducing premature repeated-output completion, avoiding ten-second hard cuts, and adding conservative stable-text-plus-silence sentence completion while preserving the existing ordinary low-energy threshold.
- Coalesce very short completed source fragments with the following compatible fragment after a bounded delay so standard, conversation, and transcription-only modes produce fewer unnatural rows.
- Preserve existing GPU model placement, translation devices, batch settings, model count, inference frequency, and session log formats.

## Capabilities

### New Capabilities

- `standard-session-continuity`: Defines stable same-GPU routing, in-page reconnect behavior, explicit resume failure, and the production connection-duration policy for standard sessions.
- `shared-meeting-log-access`: Defines shared dual-backend log discovery, historical log migration, interrupted-log visibility, and download behavior.
- `standard-asr-segmentation`: Defines non-accurate ASR segmentation diagnostics, safer low-energy handling, conservative sentence completion, and short-fragment coalescing.

### Modified Capabilities

- None.

## Impact

- Browser WebSocket connection and reconnect handling in `web/app.js` and related frontend helpers.
- WebSocket session initialization and resume error handling in `whisper_live/server.py`.
- Meeting-log listing and download presentation in `web/app.js` and `web/index.html`.
- Production Nginx `/ws-standard` upstream routing and retry policy under the environment-specific `deploy/` configuration.
- GPU0/GPU1 container mounts, `--meeting_logs_dir`, and `--max_connection_time` startup arguments.
- Historical meeting-log migration and deployment acceptance procedures in `whisperlive-ops-guide.md`.
- Faster-whisper segment completion, low-energy filtering, diagnostics, and short-fragment coalescing in `whisper_live/backend/base.py` and `whisper_live/backend/faster_whisper_backend.py`.
- Frontend non-accurate recognition parameters in `web/app.js`.
- Targeted server, meeting-log, ASR segmentation, frontend syntax, Nginx configuration, and deployment integration checks; no new dependencies, model instances, persistent GPU memory usage, or additional routine ASR inference passes.
