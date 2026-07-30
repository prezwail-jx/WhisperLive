## Why

High-accuracy mixed interpretation currently allows isolated language misclassification, incomplete cross-segment English fragments, and NLLB omissions to reach users as confusing translations or overly coarse failure placeholders. Recent meeting logs show that the system needs bounded recovery for both ASR language selection and translation completeness without increasing resident model memory or relaxing the 4-second realtime latency budget.

## What Changes

- Add high-accuracy mixed-interpretation ASR language stabilization that can perform one conditional second decode with the previous stable language when an isolated language switch looks suspicious.
- Keep English incomplete-fragment waiting capped at 4 seconds while expanding narrow incomplete-ending detection for common dangling connector phrases.
- Extend translation completeness checks for high-accuracy NLLB in both `en -> zh` and `zh -> en`, including numeric, unit, acronym, and glossary fact-anchor coverage.
- Replace immediate placeholder failure behavior with staged recovery: current strict translation, context retry, conservative relaxed generation, chunked retry, best safe low-confidence output, and only then a target-language unavailable placeholder.
- Persist `translation_confidence: "low"` in meeting JSON for safe but incomplete best-effort translations without setting `translation_warning` or showing a frontend `!`.
- Localize unavailable placeholders by target language: `翻译暂不可用` for Chinese targets and `Translation unavailable` for English targets.
- Preserve resource constraints: no new model instance, no larger persistent batch size, no persistent GPU cache, and no default device change.

## Capabilities

### New Capabilities
- `mixed-language-asr-stabilization`: Stabilizes high-accuracy mixed-interpretation ASR language selection with bounded conditional re-decode and consistent language metadata propagation.

### Modified Capabilities
- `translation-output-reliability`: Changes translation recovery, failure presentation, completeness validation, low-confidence metadata, unavailable placeholders, and observability requirements.

## Impact

- Affected backend code: `whisper_live/backend/faster_whisper_backend.py`, `whisper_live/backend/translation_backend.py`, and `whisper_live/server.py`.
- Affected frontend code: `web/app.js` and `web/index.html` for localized frontend timeout placeholders and cache busting.
- Affected tests: `tests/test_faster_whisper_backend.py`, `tests/test_translation_backend.py`, `tests/test_server_extended.py`, and `tests/test_meeting_logs.py`.
- Affected persisted data: new optional `translation_confidence` field on final translation segments in meeting JSON.
- No API-breaking change is required; existing clients that ignore `translation_confidence` continue to render translation text normally.
