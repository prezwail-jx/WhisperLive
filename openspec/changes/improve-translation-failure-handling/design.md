## Context

The translation client currently has two partially independent safety paths. NLLB Chinese-to-English output with substantial residual CJK text is retried once, while the general output guard detects malformed, excessively long, or repetitive output only after inference and replaces it with the source text without retrying or setting `translation_warning`. Model-unavailable and direct inference exception paths also return the source text. This makes failure output visually indistinguishable from a translation and prevents meeting JSON from explaining why the fallback occurred.

Translation buffering adds another diagnostic gap: `translation_warning` is attached before merge buffering, but merged translation segments do not preserve the field. A failed segment can therefore be combined with successful text and lose its failure metadata.

The change must preserve low translation latency, existing glossary and fixed-phrase behavior, batch fallback semantics, and historical meeting display. It must not load another model instance, increase batch size, change translation devices, or add resident GPU memory.

## Goals / Non-Goals

**Goals:**

- Apply one testable validation policy to every model-generated translation output in batch and direct inference paths.
- Give invalid model output one bounded recovery attempt without retrying known non-recoverable failures.
- Show a concise failure placeholder instead of source text in the translation pane while retaining `source_text` for transcript fidelity.
- Preserve stable diagnostic reasons in runtime logs and meeting JSON.
- Keep failed segments independent from adjacent successful merge output.
- Preserve successful translation, glossary, fixed-phrase, and historical error-display behavior.

**Non-Goals:**

- Changing translation models, model paths, devices, batching configuration, or GPU allocation.
- Changing ASR, hotword conditioning, audio filtering, or source transcript segmentation.
- Retrying indefinitely or guaranteeing recovery from every invalid output.
- Validating trusted fixed short-phrase or exact glossary targets as model-generated output.
- Migrating existing meeting JSON files.

## Decisions

### Classify model output before presenting it

Introduce one output-failure classifier that receives source text, translated text, source language, and target language. It will retain the existing structural guard checks and add empty-output and normalized source-echo checks. NLLB Chinese-to-English residual-CJK detection will become another reason returned by the same policy rather than a separate presentation path.

Stable reasons will include `empty_output`, `source_echo`, `residual_cjk`, `underscore_run`, `length_ratio`, `low_unique_word_ratio`, and `repeated_ngram`. Source-echo comparison will normalize Unicode, surrounding whitespace, and common punctuation so cosmetic differences cannot bypass the guard. Short ASCII proper terms such as `OpenAI`, `NICE T`, and `GPT-4` remain valid identity output when they are configured translation terms or consist only of a small number of acronym, mixed-case, or numeric tokens; ordinary untranslated sentences do not receive this exemption.

Alternative considered: treat any target-language script mismatch as invalid. This was rejected because names, acronyms, glossary targets, and mixed-language technical speech legitimately retain source-script content. The existing bounded residual-CJK rule is safer for the observed Chinese-to-English failure.

### Retry invalid model output once

Both batch and direct inference paths will validate the first generated result. Any classified output failure will emit a bounded retry diagnostic and invoke the same inference path once more. A valid second result is returned normally with no warning. A second invalid result becomes a final failed translation.

Content failures from batch inference will retry through batch so normal throughput and device scheduling are preserved. The current NLLB batch timeout fallback to direct inference counts as the recovery attempt; direct inference receives one attempt and must not cause an additional third inference. If a second batch attempt times out after an invalid first output, translation fails without another direct fallback.

Only explicit `TimeoutError` failures are transient. Model-unavailable, client-exit, CUDA out-of-memory, unsupported-model, and unknown inference exceptions will not be retried. If model-backed glossary protection exhausts recovery and produces a failed result, glossary marker fallback will stop instead of starting another plain whole-sentence translation; a normal marker-loss result without warning retains the existing plain fallback.

Alternative considered: retry every exception. This was rejected because repeating OOM or model initialization failures increases latency and resource pressure without a credible recovery path.

### Return a user-safe placeholder and preserve the source separately

After the bounded attempt is exhausted, the translated segment will use `翻译暂不可用` as `text`, retain the recognized content in `source_text`, and carry a stable `translation_warning` reason. Runtime logs will include bounded source and rejected-output previews, but full text will not be logged.

New failures will no longer append `（翻译出错）` to model output or return the source as translated text. The browser's existing support for historical suffix-bearing segments remains in place so persisted meetings need no migration.

Alternative considered: omit failed translation segments. This was rejected because users could mistake missing translation for lost source audio and the timeline would no longer reveal where translation failed.

### Isolate failed segments from merge buffering

A segment carrying `translation_warning` will force-flush any preceding successful merge buffer, be emitted immediately as its own segment, and leave the next successful translation to start a new buffer. This preserves its precise time range and prevents placeholder text from being concatenated with valid translation. Browser interleaved display will also avoid re-grouping warning segments with adjacent translations that share the same source utterance; repeated same-source rows will show a compact same-source marker instead of hiding the later failed segment.

Cleanup will force-flush both the pending translation buffer and the successful-translation merge buffer before setting the client exit flag, so a final successful segment that is only waiting for merge delay is not lost when the connection closes.

Alternative considered: propagate a combined warning through the existing merge output. This was rejected because it would not identify which source interval failed and could produce visually confusing mixed success/failure text.

### Use bounded structured diagnostics

Retry and terminal failure logs will use stable event names and include client UID, model, batch state, language pair, first and final reason, lengths, elapsed time, and bounded previews. Meeting JSON will rely on the existing `translation_warning` field rather than introducing a parallel error object.

## Risks / Trade-offs

- [A retry increases latency for invalid output] -> Limit recovery to one attempt and perform no retry for known non-recoverable states.
- [Source-echo detection rejects an intentional identity translation] -> Apply it only to model-generated cross-language translation, exempt bounded proper terms, and exclude trusted fixed/glossary exact outputs.
- [Transient-exception classification misses a recoverable error] -> Start with explicit timeout handling and structured diagnostics before broadening the policy.
- [Placeholder text enters exported meeting content] -> Preserve the full source in `source_text` and mark the segment with a machine-readable warning.
- [Changing merge boundaries increases translation segment count] -> Split only failed segments and same-source failed display rows; successful segments retain the existing merge behavior.

## Migration Plan

1. Add the shared classifier and focused unit tests without changing successful translation behavior.
2. Route batch and direct model output through the bounded retry policy.
3. Replace terminal fallback text and attach stable warning reasons.
4. Isolate warning segments in merge buffering and browser grouping, and verify warning persistence in serialized payloads.
5. Deploy without data migration; historical suffix-based records remain display-compatible.
6. Roll back by reverting the translation backend changes; no persisted configuration or model changes are required.

## Open Questions

None. Failure presentation, retry scope, merge isolation, same-source display, Markdown presentation, and cleanup flushing were confirmed before implementation.
