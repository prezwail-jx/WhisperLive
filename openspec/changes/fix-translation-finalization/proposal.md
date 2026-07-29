## Why

In faster-whisper accurate mode, stopping a meeting currently risks ending the session before the final ASR tail and queued translations are fully resolved. Completed source rows can remain stuck as gray draft or "translating" rows, and recent Chinese-to-English testing shows short high-risk segments can be polluted by readability context from previous turns.

## What Changes

- Add faster-whisper finalization so `END_OF_AUDIO` requests one final ASR pass for remaining audio, even when it is shorter than the normal realtime chunk threshold.
- Add an explicit terminal translation lifecycle for final source segments: every completed source segment must resolve to a successful final translation or a visible failed translation placeholder.
- Replace stale draft translations with a non-draft frontend timeout placeholder after 12 seconds when no terminal translation arrives.
- Change meeting stop behavior from immediate close to finalize-then-close: the client sends `END_OF_AUDIO`, waits up to 20 seconds for `SESSION_FINALIZED`, then closes the WebSocket.
- Drain queued final translation work before marking the meeting log finished, using one 15-second backend budget across ASR finalization and translation drain.
- Keep Chinese-to-English readability context, but add one no-context direct-translation verification pass only for high-risk segments.
- Preserve existing hard abnormal expansion guards, HTML entity normalization, translation model behavior, model count, device placement, and GPU usage.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `translation-output-reliability`: Require faster-whisper session finalization to complete ASR tail processing and translation drain before meeting completion, require completed source segments to reach an explicit final translation terminal state, and add Chinese-to-English risk verification that prevents context-polluted final output.

## Impact

- Backend translation lifecycle and queue shutdown behavior in `whisper_live/backend/translation_backend.py`.
- Faster-whisper final ASR tail processing in `whisper_live/backend/base.py` and server orchestration in `whisper_live/server.py`.
- Chinese-to-English context-risk validation and no-context direct verification in `whisper_live/backend/translation_backend.py`.
- WebSocket stop/finalization flow in `whisper_live/server.py` and `web/app.js`.
- Browser transcript merge and draft cleanup behavior in `web/app.js`.
- Meeting log consistency for final translation placeholders in `whisper_live/meeting/` through existing translated segment persistence.
- Targeted Python and frontend syntax checks; no new dependencies and no additional model instances.
