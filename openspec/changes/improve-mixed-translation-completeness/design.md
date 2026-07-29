## Context

High-accuracy mixed interpretation uses faster-whisper with per-chunk language auto-detection and NLLB for bidirectional Chinese/English translation. Recent logs show three classes of failures: isolated language switches produce wrong-language ASR text, completed English fragments are translated before enough context arrives, and NLLB drops numbers, units, acronyms, or tail clauses while sometimes falling back too quickly to a placeholder.

The current pipeline also makes language decisions in multiple places. Faster-whisper sets `current_language`, `server.py` can infer language again from text, and the translation backend can infer language again before buffering. This makes language metadata inconsistent between source transcript, translation queue, and meeting logs. Translation recovery currently retries invalid output in a bounded way, but it treats safety failures and completeness failures similarly and can discard usable but incomplete candidates.

The design must preserve realtime constraints: no new model instance, no persistent GPU cache, no default device change, no larger default batch size, and English incomplete-fragment waiting remains capped at 4 seconds.

## Goals / Non-Goals

**Goals:**

- Stabilize mixed-interpretation ASR language selection for suspicious isolated language switches using at most one conditional second decode.
- Ensure source transcript, translation queue, and meeting logs use the same selected language metadata.
- Expand incomplete English ending detection without increasing the 4-second incomplete wait budget.
- Detect bidirectional NLLB completeness failures using length, numeric/unit/acronym/glossary fact anchors, and tail-clause risk.
- Add staged translation recovery: strict current translation, context retry, conservative relaxed generation, chunked retry, safe low-confidence output, and target-language unavailable placeholder only as the last resort.
- Persist `translation_confidence: "low"` for safe best-effort translations without setting `translation_warning`.
- Localize unavailable placeholders by resolved target language.

**Non-Goals:**

- Do not replace ASR or translation models.
- Do not add or duplicate model instances.
- Do not increase persistent GPU memory usage, persistent cache size, default batch size, or default translation device.
- Do not extend English incomplete-fragment final wait beyond 4 seconds.
- Do not use ASR draft segments as final translation input.
- Do not solve ASR audio gaps, domain hotword quality, or mining-domain terminology in this change.
- Do not migrate or rewrite historical meeting JSON.

## Decisions

### Decision: Use conditional ASR second decode instead of language-label smoothing only

Suspicious language switches often already contain wrong-language ASR text. Smoothing only the `language` field would send incorrect text to translation and logs. The faster-whisper client will therefore compare the automatic decode against one forced decode using the previous stable `zh` or `en` language when a high-accuracy mixed-interpretation chunk proposes a language switch.

Alternatives considered:

- Only smooth the language label. Rejected because it cannot recover wrong ASR text.
- Always decode both languages. Rejected because it doubles ASR work for normal chunks.
- Lock the session language. Rejected because mixed interpretation must support true language switching.

### Decision: Gate ASR retry to high-accuracy mixed interpretation

The retry is enabled only when `service_mode == "accurate"`, `translation_mode == "mixed_interpretation"`, and the backend is faster-whisper. Standard modes keep their current behavior.

This contains GPU cost and avoids changing non-mixed transcription behavior.

### Decision: Select ASR candidate before transcript and translation enqueue

The selected candidate language and text must be finalized before `format_segment()` output is appended to transcript and queued for translation. `server.py` and the translation backend should only infer language when a segment does not already carry a trusted `zh` or `en` language. This keeps WebSocket source output, Admin state, meeting logs, and translation input aligned.

### Decision: Keep incomplete wait strict but improve ending detection

The English incomplete wait remains bounded by `translation_incomplete_max_wait_seconds`, which high-accuracy frontend currently sends as 4 seconds. Detection expands only for narrow dangling connector phrases such as `and they`, `but they`, `and we`, `assuming you`, and `we need to`. Broad verb-ending heuristics are not added because they would delay complete short sentences too often.

### Decision: Split translation safety failures from completeness failures

Safety failures such as source echo, residual CJK, hard hallucination phrases, repeated n-grams, empty output, and abnormal expansion cannot be shown to users. Completeness failures such as undertranslation, missing fact anchors, and tail omission can be shown if they are the best safe candidate and marked with `translation_confidence: "low"`.

This allows the system to avoid unnecessary placeholders while still protecting users from unsafe model output.

### Decision: Add conservative NLLB relaxed profile

The relaxed profile uses `num_beams=3`, `max_new_tokens=320`, and `length_penalty=1.1`. Beam count remains unchanged to avoid a notable transient memory increase. The relaxed profile is only used for high-accuracy NLLB recovery candidates after strict and context attempts fail or remain incomplete.

### Decision: Use fact anchors for bidirectional completeness

Length ratios alone miss cases where a translation is long enough but drops important facts. The translation backend will extract anchors for digits, percentages, currency, units, acronyms, configured glossary terms, and selected quantity words. Missing anchors trigger recovery and may produce a low-confidence output if no complete safe candidate is available.

### Decision: Low-confidence output is metadata-only for UI warning

Safe but incomplete translations include `translation_confidence: "low"` and do not include `translation_warning`. The existing frontend warning marker remains tied to `translation_warning`, so low-confidence translations do not display `!` while meeting JSON preserves diagnostic metadata.

### Decision: Last-resort placeholder is target-language localized

The backend and frontend timeout fallback choose `翻译暂不可用` for Chinese targets and `Translation unavailable` for English targets. These final unavailable segments still carry `translation_warning` and remain independently emitted so failure metadata does not merge with successful translations.

## Risks / Trade-offs

- Conditional ASR retry increases GPU work for suspicious language switches -> Mitigate by enabling only high-accuracy mixed interpretation and allowing at most one retry per audio chunk.
- True language switches may be delayed or rejected -> Mitigate by accepting automatic candidates that have strong language probability, better text-language consistency, or clearly better average log probability.
- Relaxed generation may produce longer hallucinations -> Mitigate by running the same safety validation on every relaxed candidate before selection.
- Fact-anchor detection can produce false positives for paraphrased units or quantities -> Mitigate with conservative anchor classes and explicit equivalence rules for common units and currencies.
- Chunked translation may be less fluent than whole-segment translation -> Mitigate by preferring whole-segment candidates when fact coverage is equivalent.
- Strict 4-second incomplete wait cannot catch future ASR segments that arrive later -> Mitigate by documenting this latency boundary and using low-confidence metadata instead of warning for safe incomplete outputs.
- New `translation_confidence` metadata may be dropped by merge or logs -> Mitigate with merge and meeting-log tests that preserve `low` confidence.
