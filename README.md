# PDCA — Profile-Driven Compression Architecture

**Experimental v0.2.0-alpha** — a reversible, profile-based JSON-value codec designed for LLM workflows.

PDCA discovers repeated JSON schemas, constant fields, string prefixes, and strings, saves a reusable **Profile**, and encodes new data without running an LLM. You can optionally ask OpenAI or Gemini to propose reusable string fragments. The deterministic engine **validates every proposed rule** and rejects candidates that do not save tokens or cannot be losslessly decoded.

## Quickstart

```bash
pip install -e '.[tokenizer]'
pdca train products.json -o profile.json
pdca encode products.json --profile profile.json -o products.pdca.json
pdca decode products.pdca.json -o restored.json
pdca optimize products.json -o optimized.pdca.json --profile-out optimized.profile.json
pdca benchmark products.json

# Merge previously generated candidate dictionaries without calling AI again:
pdca optimize products.json -o merged.pdca.json --candidate-profiles luna.pdca.json sol.pdca.json gemini.pdca.json
```

A profile can be reused for new datasets. Unknown keys are stored as extras rather than discarded. To avoid including the profile in every payload, use `--external-profile` when encoding and `--profile profile.json` when decoding.

Optional AI-assisted profile candidate generation (sends a bounded sample of source data to the provider, **only on explicit request**):

```bash
pip install -e '.[tokenizer,openai,gemini]'
export OPENAI_API_KEY=...
pdca optimize products.json -o products.pdca.json --ai-provider openai --ai-model gpt-6
# or export GEMINI_API_KEY=... and:
pdca optimize products.json -o products.pdca.json --ai-provider gemini --ai-model YOUR_AVAILABLE_GEMINI_MODEL
```

**GPT-6 support:** `--ai-model gpt-6` is an optional provider model identifier, not a guaranteed API entitlement. The fixed public tokenizer benchmark is OpenAI `o200k_base`, which is known to cover GPT-5.x. The current open-source `tiktoken` model map does not verify a GPT-6-specific tokenizer. We **do not claim** that the reference token count equals real GPT-6 usage. Future exact GPT-6 API token accounting belongs in a separate measurement.

## What is (and isn't) lossless?

- Encoded JSON values are reconstructed exactly by type and content, including nulls, empty fields, nested objects, unknown properties, and Unicode.
- **Original whitespace and key ordering are not retained.** This is *JSON-value* losslessness, not byte-for-byte file preservation.
- Profiles contain **declarative** schema/defaults/constants/field references/string patterns. Untrusted AI proposals are filtered and never executed.
- `encode` and `decode` run **offline**, no API key required.
- The benchmark measures **compact serialized payload plus embedded profile** against **compact JSON**, not raw pretty-printed text. Profile creation and AI usage costs are separate.
- An external reusable Profile is identified by SHA-256, avoiding silent profile mismatch.
- Only use compatible JSON values, not NaN or Infinity.

## Design / roadmap

1. Deterministic JSON profile training, schema constants, field-reference encoding, encoder/decoder, offline CLI, OpenAI reference tokenizer benchmarking.
2. Optional OpenAI/Gemini candidate suggestions with strict validation.
3. Cross-model comprehension test harness, robust adversarial/fuzz testing, larger datasets, explicit profile-amortization accounting.
4. Exact provider API token usage reporting and further stream/CSV support.

## 日本語

PDCAは、JSONの共通構造や繰り返し文字列をProfileとして保存し、次回以降はAIなしで可逆変換する実験的なOSSです。標準のトークン比較は`o200k_base`で統一し、GPT-6の正確なトークン数であるとは主張しません。OpenAI/Gemini連携は利用者が明示的に実行したときだけ呼び出します。データ送信に注意してください。

License: MIT. This project is not affiliated with OpenAI or Google.

## Reproducibility and release limitations

PDCA is a research prototype, not a general-purpose byte compressor. Token benchmarks always include the embedded profile; they exclude the additional text needed to teach an LLM to interpret the format. The AI-assisted optimizer currently proposes string candidates, **not full schemas**. Report the tokenizer encoding and exact input comparison when publishing results. On the Anker 47-product test dataset, three previously generated candidate profiles were merged and evaluated with `o200k_base`; the benchmark is a single dataset, not a claim of general average savings. We have **not** evaluated LLM question-answering accuracy or the per-call AI cost. Do not use packed PDCA data as a standalone natural-language prompt without decoding or a rigorously specified decoder.

## Measured benchmark (2026-11-14, Anker 47 product JSON records)

All numbers use `tiktoken`/`o200k_base` on the **entire compact packet, including the embedded Profile**:

| Method | Tokens | Reduction vs compact JSON |
| --- | ---: | ---: |
| Compact JSON (baseline) | 39,343 | 0% |
| Initial PDCA v0.1 (no AI) | 30,709 | 21.95% |
| Initial PDCA + Luna suggestions | 30,178 | 23.30% |
| Initial PDCA + Gemini suggestions | 30,057 | 23.60% |
| Initial PDCA + Sol suggestions | 29,440 | 25.17% |
| **PDCA v0.2 (no AI)** | **27,996** | **28.84%** |
| **PDCA v0.2 + merged prior AI suggestions** | **27,310** | **30.58%** |

Each method was run on the same 47-record JSON dataset and reference encoding. Source JSON reconstructed exactly (data values); tests are not a claim about general-purpose effectiveness. *AI model calls were performed earlier; no new API calls were required for the combined experiment.* Runtime and provider API costs were not recorded. The actual tokenizer for proprietary provider inference may differ.

Details: [docs/benchmark-ja.md](docs/benchmark-ja.md).

**Known limitations:** packed data is not inherently readable by an LLM; decoding or explicit format instructions is necessary, which may add tokens. There is no measured QA comprehension test yet. Also, the persisted `pdca: 1` format is experimental: new profiles add optional `constants` and `field_refs` sections, and the current decoder supports previously produced packets.