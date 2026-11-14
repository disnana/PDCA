# PDCA — Profile-Driven Compression Architecture

**Experimental v0.1.0-alpha** — a reversible, profile-based JSON-value codec designed for LLM workflows.

PDCA discovers repeated JSON schemas and strings, saves a reusable **Profile**, and encodes new data without running an LLM. You can optionally ask OpenAI or Gemini to propose reusable string fragments. The deterministic engine **validates every proposed rule** and rejects candidates that do not save tokens or cannot be losslessly decoded.

## Quickstart

```bash
pip install -e '.[tokenizer]'
pdca train products.json -o profile.json
pdca encode products.json --profile profile.json -o products.pdca.json
pdca decode products.pdca.json -o restored.json
pdca optimize products.json -o optimized.pdca.json --profile-out optimized.profile.json
pdca benchmark products.json
```

A profile can be reused for new datasets. Unknown keys are stored as extras rather than discarded. To avoid including the profile in every payload, use `--external-profile` when encoding and `--profile profile.json` when decoding.

Optional AI-assisted profile candidate generation (sends a bounded sample of source data to the provider, **only on explicit request**):

```bash
pip install -e '.[tokenizer,openai,gemini]'
export OPENAI_API_KEY=...
pdca optimize products.json -o products.pdca.json --ai-provider openai --ai-model gpt-6
# or export GEMINI_API_KEY=... and:
pdca optimize products.json -o products.pdca.json --ai-provider gemini --ai-model gemini-3.8-flash
```

The Gemini model ID above is verified against Google's official model documentation. Replace it with another supported model ID only if your API project supports it. Never pass the literal placeholder `GEMINI_MODEL`. See https://ai.google.dev/gemini-api/docs/models.\n\n**GPT-6 support:** `--ai-model gpt-6` is an optional provider model identifier, not a guaranteed API entitlement. The fixed public tokenizer benchmark is OpenAI `o200k_base`, which is known to cover GPT-5.x. The current open-source `tiktoken` model map does not verify a GPT-6-specific tokenizer. We **do not claim** that the reference token count equals real GPT-6 usage. Future exact GPT-6 API token accounting belongs in a separate measurement.

## What is (and isn't) lossless?

- Encoded JSON values are reconstructed exactly by type and content, including nulls, empty fields, nested objects, unknown properties, and Unicode.
- **Original whitespace and key ordering are not retained.** This is *JSON-value* losslessness, not byte-for-byte file preservation.
- Profiles contain **declarative** schema/defaults/string patterns. Untrusted AI proposals are filtered and never executed.
- `encode` and `decode` run **offline**, no API key required.
- The benchmark measures **compact serialized payload plus embedded profile** against **compact JSON**, not raw pretty-printed text. Profile creation and AI usage costs are separate.
- An external reusable Profile is identified by SHA-256, avoiding silent profile mismatch.
- Only use compatible JSON values, not NaN or Infinity.

## Design / roadmap

1. Deterministic JSON profile training, encoder/decoder, offline CLI, OpenAI reference tokenizer benchmarking.
2. Optional OpenAI/Gemini candidate suggestions with strict validation.
3. Cross-model comprehension test harness, robust adversarial/fuzz testing, larger datasets, explicit profile-amortization accounting.
4. Exact provider API token usage reporting and further stream/CSV support.

## 日本語

PDCAは、JSONの共通構造や繰り返し文字列をProfileとして保存し、次回以降はAIなしで可逆変換する実験的なOSSです。標準のトークン比較は`o200k_base`で統一し、GPT-6の正確なトークン数であるとは主張しません。OpenAI/Gemini連携は利用者が明示的に実行したときだけ呼び出します。データ送信に注意してください。

License: MIT. This project is not affiliated with OpenAI or Google.