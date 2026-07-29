## Why

High-accuracy interpretation currently waits for an ASR utterance to finish before producing a Chinese translation, so English-to-Chinese users see source text update like ASR but translated text still arrives only as a completed block. This change adds ASR-like draft Chinese translations for high-accuracy English-to-Chinese sessions so users can start reading evolving translation text before the source utterance is finalized.

## What Changes

- Add an opt-in translation draft mode for high-accuracy English-to-Chinese interpretation.
- Allow the frontend to request draft translation behavior with bounded draft scheduling parameters.
- Maintain per-client, per-utterance draft state so new ASR draft text coalesces into the latest draft slot instead of flooding the existing translation FIFO.
- Emit WebSocket `translated_segments` with `completed: false` for draft Chinese translations, then replace them with the existing `completed: true` final translation.
- Ensure draft translations never enter meeting logs, summary input, persistent translation state, or formal Admin translation statistics.
- Preserve all existing behavior for standard interpretation, Chinese-to-English, conversation translation, transcription-only mode, and clients that do not enable the draft flag.

## Capabilities

### New Capabilities

- `accurate-translation-drafts`: ASR-like draft Chinese translation output for high-accuracy English-to-Chinese interpretation, including enablement rules, draft scheduling, WebSocket semantics, frontend replacement behavior, and log/Admin isolation.

### Modified Capabilities

None.

## Impact

- Affected frontend: `web/app.js` connection config, translation segment keying, draft/final replacement, and rendering state cleanup.
- Affected backend: `whisper_live/backend/translation_backend.py`, `whisper_live/backend/base.py` if needed for draft ASR segment handoff, `whisper_live/server.py`, and `run_server.py` for parameter parsing and propagation.
- Affected tests: translation backend tests, server configuration tests, and frontend syntax checks.
- No new model instances, model paths, GPU defaults, WebSocket endpoint paths, persistent file formats, Admin APIs, or meeting log formats are introduced.
