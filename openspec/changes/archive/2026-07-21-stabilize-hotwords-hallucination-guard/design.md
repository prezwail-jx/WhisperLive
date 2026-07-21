## Context

WhisperLive currently accepts client, meeting, and default hotwords as an unstructured string. Meeting files are parsed into one space-separated prompt, then the same value is forwarded through server initialization to faster-whisper or FunASR. The faster-whisper batch path rebuilds prompts itself, while FunASR can apply the value during streaming recognition, non-streaming recognition, and final refinement. Existing no-speech, VAD, low-energy, and phrase filters are generic and do not account for output induced by the configured hotword list.

The change crosses meeting configuration, server orchestration, backend adapters, batch inference, and segment filtering. It must preserve low latency, avoid additional model instances or GPU memory use, and keep translation glossary behavior independent from ASR conditioning.

## Goals / Non-Goals

**Goals:**

- Produce one normalized, deduplicated, bounded ASR hotword representation per connection.
- Keep default, meeting, and client hotwords consistent across supported recognition paths.

- Suppress hotword-dominated output only when audio or model evidence indicates silence or weak speech.
- Make safety decisions observable through concise structured logs and focused tests.

**Non-Goals:**

- Changing ASR models, decoding devices, batch sizes, or GPU allocation.
- Guaranteeing recognition of every configured term or eliminating all general ASR hallucinations.
- Reimplementing meeting hotword processing in the browser.
- Changing translation glossary syntax or applying translation-only rules to ASR.
- Adding per-hotword weights before backend support and operational evidence justify them.

## Decisions

### Normalize once before backend initialization

Extend the meeting hotword module with a canonical ASR hotword representation containing the accepted terms, prompt text, original count, accepted count, and truncation metadata. Normalize whitespace, remove exact duplicates while retaining source order, reject empty or oversized entries, and cap both accepted term count and total prompt size.

The server will apply this normalization after resolving client, meeting, or default precedence and before constructing a backend client. Central normalization is preferred over independent backend cleanup because it guarantees matching behavior, logging, status metadata, and batch prompt signatures.

Alternative considered: rely on each ASR library to truncate prompts. This was rejected because backend limits and failure modes differ and are not visible to administrators.

### Preserve source precedence and glossary separation

Keep the existing precedence of explicit locked/client hotwords, selected meeting hotwords, then default hotwords. Translation rules containing `=>` remain in the translation glossary and never become ASR terms. Normalized ASR terms and translation terms remain separate fields even when sourced from the same meeting file.

Alternative considered: merge translation source terms into ASR hotwords automatically. This was rejected because translation terminology can be much larger and can create recognition bias the user did not request.

### Use one backend-specific conditioning policy

Faster-whisper standard and batch paths will consume the same bounded prompt. Batch grouping will continue to include that prompt in its compatibility signature.

FunASR backend hotword forwarding is left unchanged in this change to avoid compounding risk; its existing retry-without-hotwords fallback remains active.

### Combine hotword dominance with existing speech evidence

Add a backend-independent segment guard in the shared output path. It will identify whether normalized output is wholly or predominantly composed of configured hotwords, but it will discard the segment only when paired with existing weak-evidence signals such as low RMS or excessive no-speech probability. A hotword match by itself is never sufficient for rejection.

The guard belongs beside existing low-energy and no-speech checks so completed and incomplete segments use the same policy. Backends that do not provide meaningful no-speech probability can still use RMS/VAD evidence.

Alternative considered: maintain a blacklist of observed hallucinated hotwords. This was rejected because meeting vocabularies are dynamic and legitimate terms would be suppressed globally.

### Log decisions without recording excessive content

Connection logs will report source, original/accepted counts, truncation reason, and a limited preview. Segment-drop logs will include the evidence type and a bounded text preview. Full meeting files and unbounded recognized text will not be copied into logs.

## Risks / Trade-offs

- [Aggressive limits reduce recall for large terminology lists] -> Preserve source order, expose accepted counts, and choose conservative limits covered by tests.
- [Low-volume speakers may be mistaken for silence] -> Require both hotword dominance and weak evidence; do not reject ordinary low-energy text solely because it contains a hotword.
- [Normalization changes batch compatibility signatures] -> Normalize before request creation so equivalent lists share the same stable signature.
- [Extra text matching adds latency] -> Precompute normalized terms once per connection and use bounded lists and linear matching in the segment path.

## Migration Plan

1. Introduce normalization and metadata while retaining current hotword source precedence.
2. Route all backend initialization through the normalized prompt and add observability.
3. Apply backend-specific forwarding and the shared hallucination guard.
4. Deploy with existing hotword files; no file migration is required.
5. Roll back by reverting the code change; stored meeting files and client protocol remain compatible.

## Open Questions

- Exact default count, per-term length, and total prompt limits should be selected during implementation using current production hotword files and tokenizer constraints, then fixed in regression tests.
