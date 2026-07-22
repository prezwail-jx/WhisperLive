## 1. Unified Output Classification

- [x] 1.1 Add one translation-output failure classifier in `whisper_live/backend/translation_backend.py` that covers empty output, normalized source echo with bounded short-proper-term exemptions, residual CJK, abnormal length, low word diversity, repeated n-grams, and malformed-character runs with stable reason codes.
- [x] 1.2 Route both direct and NLLB batch model results through the shared classifier while leaving fixed short-phrase and exact glossary translations trusted and unchanged.

## 2. Bounded Retry And Failure Results

- [x] 2.1 Replace the residual-CJK-only retry branches with one-attempt recovery for every classified model-output failure in both direct and batch paths, counting batch-to-direct timeout fallback as the single recovery attempt.
- [x] 2.2 Classify retryable transient exceptions separately from model-unavailable, client-exit, CUDA out-of-memory, unsupported configuration, and other non-recoverable failures so only eligible failures repeat inference.
- [x] 2.3 Return `翻译暂不可用` after recovery is exhausted, retain the original text in `source_text`, and attach the final stable reason through `translation_warning` without generating new source-text or suffix fallbacks.
- [x] 2.4 Emit bounded structured retry and terminal-failure logs containing client, model, batch state, language pair, first/final reasons, timing context, lengths, and truncated previews.

## 3. Failed-Segment Isolation

- [x] 3.1 Update translation merge buffering so a warning segment flushes preceding successful output, is emitted immediately as an independent segment, causes following successful output to begin a new merge buffer, and cleanup force-flushes any remaining merge buffer.
- [x] 3.2 Preserve `translation_warning`, `source_text`, source utterance IDs, and the failed segment's time range through client payloads, browser interleaved display, and meeting JSON while retaining historical browser compatibility for `（翻译出错）` records.

## 4. Regression Coverage And Verification

- [x] 4.1 Extend `tests/test_translation_backend.py` for every output-failure reason, normalized source echo, valid first output, retry recovery, retry exhaustion, batch timeout fallback, transient exception retry, and non-recoverable failures.
- [x] 4.2 Add focused merge tests proving failed placeholders remain independent between successful translations and warning metadata survives serialization.
- [x] 4.3 Run container-based `py_compile` for the changed Python module and `tests.test_translation_backend`, then run `git diff --check` and inspect the final diff and status.
