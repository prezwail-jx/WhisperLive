## Context

The translation backend already uses `translation_buffer` before final model inference. It can merge completed ASR segments, preserve multiple `source_utterance_ids`, flush on language switch, and bound translation with wait, context, and character thresholds. English-to-Chinese already has incomplete-ending detection, but Chinese-to-English only has a very short CJK buffer and generic sentence-ending logic. This means faster-whisper fragments such as "我认为应用为导向" can be translated before the following segment supplies the object or conclusion.

The existing ASR correction layer runs before completed segments enter the translation queue. This change should preserve that ordering and only adjust when Chinese source text is flushed to fixed translation rules and NLLB.

## Goals / Non-Goals

**Goals:**

- Improve Chinese-to-English final translation quality by holding likely incomplete Chinese fragments until a following completed source segment arrives or bounded timing requires flush.
- Apply to standard Chinese-to-English sessions and Chinese source segments in bidirectional interpretation.
- Keep latency bounded with idle, segment-gap, accumulated-audio, character-count, language-switch, speaker-switch, cleanup, and drain boundaries.
- Preserve source segment terminal-state tracking by keeping merged translations bound to all covered source utterance IDs and time ranges.
- Keep ASR source text, meeting source logs, model choices, translation devices, and GPU memory behavior unchanged.

**Non-Goals:**

- Do not add a grammar parser, LLM, or separate sentence segmentation model.
- Do not change English-to-Chinese draft translation behavior.
- Do not change FunASR behavior.
- Do not alter meeting log schema or frontend rendering semantics.
- Do not guarantee linguistically perfect sentence completion; this is a bounded heuristic.

## Decisions

1. Extend the existing translation buffer instead of adding a second buffer.

   The current buffer already feeds fixed phrase translation, glossary protection, NLLB translation, final segment emission, terminal-state resolution, and cleanup drain. Extending its flush decision avoids duplicate state and preserves existing `source_utterance_ids` semantics. A parallel buffer was rejected because it would need to reimplement language switching, pending final tracking, drain behavior, and failure placeholders.

2. Enable only for Chinese source with resolved English target.

   The new logic should run when `source_language == "zh"` and `_resolved_target_language(source_language) == "en"`. This covers standard Chinese-to-English and Chinese turns in bidirectional interpretation while leaving English-to-Chinese on its existing incomplete-English path.

3. Use conservative incomplete-Chinese heuristics.

   The backend will strip trailing whitespace and common Chinese/English punctuation before checking incomplete endings. It will hold text ending in connective or lead-in phrases such as `那么`, `以及`, `从而`, `我们认为`, `我认为`, `具体来说`, `主要包括`, `如果`, `因为`, `虽然`, `为了`, `对于`, `关于`, `一方面`, `另一方面`, `就是`, `在于`, `意味着`, and `的话`. It will also treat comma-like endings as incomplete. A full predicate/object detector was rejected because Chinese speech often omits subjects and predicates naturally, making false merges likely.

4. Add idle-based flush alongside existing max wait.

   The current buffer starts timing when the first segment arrives. For Chinese fragment buffering, quality depends on whether a following segment arrives shortly after the latest segment. The implementation should track `translation_buffer_last_added_at` and flush Chinese-to-English incomplete buffers when no new segment arrives for about 1.2 seconds. This gives the next completed source segment a chance to arrive without waiting for the full max wait.

5. Bound continuous buffering with existing and new limits.

   Chinese-to-English buffering will still flush on max source characters and cleanup/drain. It should also flush when accumulated buffered audio reaches about 8 seconds. Segment gaps over about 1.0 second and speaker changes should flush the previous buffer before adding the new segment, preventing unrelated utterances from merging.

6. Keep fixed translation and glossary after buffering.

   The merged Chinese source text will continue through `translate_fixed_short_phrase`, `translate_with_glossary`, `translate_standalone_interjection`, and then `translate_text`. This preserves the existing trusted rule precedence and NLLB validation path.

## Risks / Trade-offs

- Increased final translation latency → Bound with a short idle timeout, segment-gap flush, max audio seconds, max chars, and force flush on drain/cleanup.
- False merge across speakers → Flush before adding a segment when `speaker` changes between buffered Chinese segments.
- False merge across language turns in bidirectional interpretation → Preserve existing language-switch flush before adding the new language segment.
- Whisper may add punctuation to semantically incomplete text → Strip trailing punctuation before incomplete-ending detection so phrases such as `主要包括。` still wait.
- Holding too long can delay terminal translation rows → Keep finalization drain force-flush behavior and ensure pending final segments covered by a merged translation are resolved together.
- Heuristic rules may miss some incomplete Chinese fragments → Keep the rule set small and test-driven; future rule expansion can be additive without changing the architecture.
