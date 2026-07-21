## 1. Canonical Hotword Configuration

- [x] 1.1 Add ordered deduplication, per-term validation, term-count limiting, total-prompt limiting, and truncation metadata to `whisper_live/meeting/hotwords.py` while keeping translation rules separate.
- [x] 1.2 Update default, meeting, and locked/client hotword resolution in `whisper_live/server.py` to produce one canonical prompt and preserve the existing source precedence.
- [x] 1.3 Extend connection and client-status metadata with bounded source, accepted-count, original-count, truncation, and preview information without exposing complete hotword files.

## 2. Faster-Whisper Conditioning

- [x] 2.1 Only enable canonical ASR hotword conditioning for `accurate` service mode, preserve translation glossaries in other modes, and ensure standard and batch faster-whisper paths use the same prompt with incompatible prompts grouped separately.

## 3. Hallucination Guard

- [x] 3.1 Precompute bounded normalized hotword matching data for accurate-mode faster-whisper clients and detect output wholly or predominantly composed of active hotwords.
- [x] 3.2 Integrate hotword-dominance detection with no-speech probability and segment/audio RMS evidence for both completed and partial transcript output.
- [x] 3.3 Emit a bounded `HOTWORD_HALLUCINATION_DROP` diagnostic containing the client, evidence type, and text preview whenever the new guard rejects output.

## 4. Regression Coverage And Verification

- [x] 4.1 Extend meeting hotword tests for normalization, duplicate ordering, safety limits, truncation metadata, source precedence, and glossary isolation.
- [x] 4.2 Add focused tests for service-mode hotword gating, translation-glossary preservation, canonical prompt forwarding, and standard/batch prompt isolation.
- [x] 4.3 Add shared segment-filter tests proving weak hotword-dominated output is dropped while supported spoken hotwords and unrelated weak text retain existing behavior.
- [x] 4.4 Run container-based syntax checks for changed Python modules and the directly affected test modules, then run `git diff --check` and inspect the final diff.
