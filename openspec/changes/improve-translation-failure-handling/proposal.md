## Why

Abnormal translation output is currently replaced with the source text for several guard and exception paths, which makes untranslated Chinese appear as if it were an English translation and leaves meeting exports without a reliable failure diagnosis. Translation output needs consistent validation, one bounded recovery attempt, and a user-safe failure result that remains observable in runtime logs and meeting JSON.

## What Changes

- Classify empty output, normalized source echo, residual source-language text, abnormal length, repetitive output, and malformed-character runs through one translation-output validation policy.
- Retry every invalid model-generated output at most once, and retry only explicitly transient inference exceptions; do not retry model-unavailable, client-exit, CUDA out-of-memory, or other non-recoverable failures.
- Replace final failed translation text with `翻译暂不可用` while retaining the original content in `source_text`.
- Record stable failure reasons in `translation_warning` and emit bounded structured retry and final-failure logs.
- Keep failed translation segments independent from adjacent successful translations so merge buffering and browser grouping cannot hide warning metadata or combine placeholder text with valid output.
- Preserve display compatibility for historical segments that use the existing `（翻译出错）` suffix.
- Add focused regression coverage for validation, retry outcomes, exception classification, failure presentation, diagnostics, and merge isolation.

## Capabilities

### New Capabilities

- `translation-output-reliability`: Defines translation-output validation, bounded retry behavior, user-safe failure presentation, diagnostic metadata, and failed-segment merge isolation.

### Modified Capabilities

None.

## Impact

- Primarily affects translation inference, output guarding, warning propagation, merge buffering, and cleanup flushing in `whisper_live/backend/translation_backend.py`.
- Reuses the existing `translation_warning` field and browser warning indicator, with browser grouping updated so same-source failed segments remain visible as independent rows.
- Extends focused backend and frontend coverage; no model, dependency, device-selection, batch-size, ASR, hotword, or resident GPU-memory changes are introduced.
