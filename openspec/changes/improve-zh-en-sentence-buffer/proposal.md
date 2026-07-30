## Why

Chinese-to-English translation quality suffers when faster-whisper emits semantically incomplete Chinese fragments and the translation layer sends each fragment to NLLB independently. Buffering Chinese fragments until a likely complete sentence is available gives the translation model enough source context to preserve the speaker's intended claim without changing ASR text or adding model cost.

## What Changes

- Enhance the existing final translation buffer for Chinese-to-English source text so incomplete Chinese fragments can wait for following completed source segments before model translation.
- Apply this behavior to standard Chinese-to-English sessions and to Chinese source segments in bidirectional interpretation.
- Keep English-to-Chinese behavior on the existing English incomplete-sentence path.
- Preserve the current ordering: ASR text correction first, then sentence buffering, then fixed glossary/translation rules, then NLLB or the selected translation model.
- Bound latency with idle, segment-gap, audio-duration, and character-count limits, and force-flush on language or speaker boundary.
- Do not add translation models, ASR models, or persistent GPU memory use.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `translation-output-reliability`: Add bounded Chinese-to-English sentence buffering before final translation inference, while preserving terminal-state guarantees for all completed source segments.

## Impact

- Affected backend code: `whisper_live/backend/translation_backend.py`.
- Affected connection configuration: existing translation runtime config from `web/app.js` and server normalization in `whisper_live/server.py` may pass bounded Chinese-to-English buffering parameters.
- Affected tests: translation buffer tests in `tests/test_translation_backend.py`.
- No new model dependencies, service processes, GPU residency, or meeting log schema changes.
