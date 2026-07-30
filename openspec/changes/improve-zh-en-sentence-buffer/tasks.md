## 1. Translation Buffer Heuristics

- [x] 1.1 Add bounded Chinese-to-English sentence-buffer configuration fields to `ServeClientTranslation.__init__` with safe defaults for enabled state, idle seconds, max audio seconds, and max segment gap seconds.
- [x] 1.2 Track `translation_buffer_last_added_at` and reset it whenever the translation buffer is flushed or cleared.
- [x] 1.3 Add `chinese_text_ends_incomplete()` with conservative connective and lead-in phrase rules that strip trailing punctuation before matching.
- [x] 1.4 Add a helper that determines whether Chinese-to-English sentence buffering applies for the current buffer direction.

## 2. Flush And Boundary Behavior

- [x] 2.1 Update `translation_buffer_flush_reason()` so incomplete Chinese-to-English buffers wait for continuation until idle timeout, max audio duration, max chars, force, or complete sentence conditions apply.
- [x] 2.2 Preserve immediate flush for complete Chinese sentence endings that do not match incomplete-Chinese heuristics.
- [x] 2.3 Flush the existing buffer before adding a new segment when source language changes, speaker changes, or source time gap exceeds the configured max gap.
- [x] 2.4 Ensure cleanup, exit signal, and translation drain still force-flush any buffered Chinese-to-English source segments.
- [x] 2.5 Keep fixed phrase translation, glossary protection, and model translation after the final joined source text is selected.

## 3. Runtime Configuration

- [x] 3.1 Pass Chinese-to-English sentence-buffer options from `TranscriptionServer.initialize_client()` into `ServeClientTranslation`.
- [x] 3.2 Add frontend runtime config values for accurate and standard modes without changing translation device, model, or ASR configuration.
- [x] 3.3 Keep the feature active only for standard Chinese-to-English and Chinese turns in bidirectional interpretation; English-to-Chinese must continue using the existing English incomplete-sentence path.

## 4. Tests

- [x] 4.1 Add unit tests for Chinese incomplete-ending detection, including punctuated incomplete fragments such as `主要包括。`.
- [x] 4.2 Add translation buffer tests proving multiple Chinese fragments merge into one Chinese-to-English final translation with all source utterance IDs preserved.
- [x] 4.3 Add tests for complete Chinese sentence immediate flush and idle-timeout flush.
- [x] 4.4 Add tests for max audio duration, max character count, language switch, speaker switch, and segment gap flush boundaries.
- [x] 4.5 Add regression tests proving English-to-Chinese buffering behavior is unchanged.

## 5. Verification

- [ ] 5.1 Run container-side `py_compile` for changed Python files in the existing deployment container.
- [ ] 5.2 Run the directly related translation backend test module in the existing deployment container.
- [x] 5.3 Run `git diff --check` and inspect `git status --short` plus relevant diffs before delivery.
