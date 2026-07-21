## Why

Enabling meeting or default hotwords can over-condition ASR decoding, causing terms from the hotword list to appear during silence, weak speech, or unrelated speech. The hotword path needs explicit safety limits and hallucination guards so terminology assistance improves recognition without reducing transcript reliability.

## What Changes

- Normalize and deduplicate ASR hotwords before they are passed to recognition backends.
- Bound the number and encoded prompt size of active ASR hotwords, with observable logging when entries are ignored or truncated.
- Keep translation-only glossary rules out of ASR conditioning while preserving their translation behavior.
- Apply backend-aware hotword conditioning so faster-whisper standard and batch paths consume the same bounded prompt without amplifying bias.
- Reject low-evidence recognition output that is dominated by configured hotwords, while retaining matching terms when supported by speech.
- Add regression coverage for parsing, prompt construction, backend forwarding, and low-energy hotword hallucinations.

## Capabilities

### New Capabilities
- `hotword-conditioning`: Defines safe hotword normalization, bounded backend conditioning, and evidence-based hallucination suppression.

### Modified Capabilities

None.

## Impact

- Affects meeting hotword parsing and storage in `whisper_live/meeting/hotwords.py`.
- Affects connection-time hotword selection and observability in `whisper_live/server.py`.
- Affects faster-whisper batch inference and shared segment filtering under `whisper_live/backend/` and `whisper_live/batch_inference.py`. FunASR hotword forwarding is left unchanged at this stage.
- Extends focused hotword and backend tests without adding models, changing device selection, or increasing resident GPU memory.
