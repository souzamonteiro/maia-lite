# Maia Lite 0.3.0 Small — Architecture and Corpus Policy (Draft v1)

**Status:** DESIGN PROPOSAL — not frozen; no production preprocessing authorized by this document.
**Date:** 2026-10-27
**Source:** Maia Corpus v8, freeze `maia_corpus_v8_001`; stratified pilot v2.

## 1. Evidence and constraints

- Full audit: 10,101,231 documents; 7,590,554,035 raw tokenizer tokens across 6 sources.
- Pilot v2: 6,000 documents (1,000/source); 5,660 train / 340 validation; 4,045,565 / 226,869 tokens including one EOS per document; 3,950 / 221 complete 1,024-token blocks. Six pilot checks passed.
- Candidate tokenizer fingerprint: `5da873b1198ebe273f4254be9a67a5e12426c796c6026022e8a65c57113a2f07`. Historical identity not conclusively proven; sampled recounts matched.
- Dataset licenses, near-duplicate leakage, benchmark contamination, and quality by language are not yet certified.

## 2. Model architecture — proposed baseline, subject to smoke test

| Parameter | Proposal |
|---|---|
| Task | Causal next-token language modeling |
| Network | Decoder-only Transformer, pre-norm |
| Layers | 20 |
| Hidden width | 1,024 |
| Attention heads | 16; head dim 64 |
| KV heads | 16 (standard MHA initially) |
| FFN | SwiGLU, intermediate width 2,752 (rounding-compatible implementation to be verified) |
| Positional embeddings | RoPE (base 10,000 initially) |
| Context length | 2,048 tokens (provisional) |
| Normalization | RMSNorm |
| Attention backend | PyTorch SDPA when supported; deterministic fallback for tests |
| Embeddings | Tied input/output token embeddings |
| Vocabulary | Exact candidate tokenizer vocabulary and special-token IDs from files, verified at runtime; never assume old model IDs except EOS=2 observed in pilot |
| Initialization | Fresh random initialization, explicitly seeded |
| Training dtype | BF16 on supported GPU; FP32 optimizer states |
| Loss | Causal cross-entropy, ignore padding only where present |

**Do not call this a fixed parameter count:** compute exact parameter total from the implemented model before approval. Do not silently substitute a different tokenizer, RoPE convention, FFN implementation, or context length.

## 3. Language and source policy

Observed raw token distribution (not a proposed target):

| Language | Raw tokens | Observed share |
|---|---:|---:|
| Portuguese | 3,094,473,339 | 40.77% |
| English | 2,587,075,221 | 34.08% |
| Spanish | 1,909,005,475 | 25.15% |
| Total | 7,590,554,035 | 100% |

**Proposed sampling policy for pilot training:** PT 40%, EN 35%, ES 25% by **sampled training tokens** (not documents). Within each language, begin with source-proportional sampling by eligible token count; log actual shares. Do not assert that this is optimal: evaluate held-out loss per language and source before changing weights. Keep a natural-distribution baseline for comparison. No cross-split source reuse of identical normalized texts; near-duplicates require a separate audit.

## 4. Data partition and leakage controls

1. Preserve acquisition files, audit SQLite, and freeze directory as read-only inputs.
2. Produce a **persistent split manifest** before tokenization, keyed by stable content hash and source ID, with train/validation/test assigned deterministically. Recommend initial 98/1/1 for large-scale experiments, but reserve a separate independent benchmark; final fractions require approval. Split before packing; never split packed blocks at random.
3. Detect and quarantine exact normalized-text duplicates across sources globally; near-duplicate detection and contamination screening are independent required gates.
4. Avoid treating document hash separation as proof of semantic non-leakage.
5. Record license/terms review by source before releasing or distributing derivative artifacts.

## 5. Tokenization and packing contract

- Validate tokenizer fingerprint against the freeze and record hashes of each tokenizer file.
- Tokenize without truncation; append exactly one EOS token per accepted document, as in pilot v2.
- Pack separately by split. Never combine train and validation tokens into the same block.
- For a 2,048-token context, emit sequences of **2,049 tokens** if targets are produced by a one-position shift within each sequence, or use 2,048-token sequences with an explicit model-internal shift. Specify one convention in code and test it; do not drop the last label inadvertently.
- Preserve document-boundary metadata. Plain concatenation permits attention across documents; if this is undesirable, implement and test boundary-aware masking rather than assuming EOS masks attention.
- Decide and document tail policy: carry forward within split/shard where safe; otherwise pad with labels masked or discard with an exact counter. Never silently lose tokens.
- Ensure deterministic shard ordering, RNG, and exact restart at shard/record offset; record hashes for each output shard.

## 6. Production pipeline — proposed stages

1. **Inventory and provenance gate:** verify freeze manifest SHA-256, source inventory, tokenizer hashes and file existence.
2. **Global dedup/split gate:** create indexed document manifest with stable IDs and split assignments; checkpoint progress.
3. **Tokenizer workers:** stream gzip JSONL, validate text, tokenize, append EOS; write atomic temporary shards, fsync/rename, checksum.
4. **Packing workers:** deterministic token stream per split/source; produce fixed-length sequences, label-shift tests, tail accounting.
5. **Mixture manifest:** construct a seeded schedule that targets PT/EN/ES token shares and logs realized shares.
6. **QA gate:** reconstruct sampled sequences, verify boundaries, IDs, hashes, split disjointness, token conservation and counts.
7. **Training pilot:** verify forward/backward, BF16 stability, memory peak, throughput, per-language validation loss, exact checkpoint resume.
8. **Release gate:** freeze code commit, model config, dataset manifest, tokenizer, preprocessing parameters and QA report.

## 7. Checkpoint and validation contract

- Save model, optimizer, scheduler, gradient scaler if applicable, RNG states (Python/NumPy/Torch/CUDA), data sampler cursor and global step.
- Validate exact resume by comparing next batches and short loss trajectories against an uninterrupted control run.
- Log gradient norms, learning rate, tokens seen, effective batch tokens, GPU memory, tokens/s, loss by language/source, and held-out perplexity where meaningful.
- Save an independent validation set that is not sampled during optimizer updates. Prevent train/validation mixing in packed shards.

## 8. Approval gates

- [ ] Exact tokenizer identity/config accepted; sampled recount documented.
- [ ] Model implemented and exact parameter count verified.
- [ ] Context 2,048 passes memory/throughput pilot on target GPU.
- [ ] Split policy and held-out benchmark accepted.
- [ ] Near-duplicate and licensing review planned/completed for intended use.
- [ ] Full preprocessing runs with resume and checksum tests on a small subset.
- [ ] Language mixing realized ratios verified and compared with natural baseline.
- [ ] Checkpoint exact-resume test passed.

**Next artifact:** executable Colab smoke-test notebook for model instantiation, parameter counting, synthetic forward/backward, tokenizer compatibility, memory profiling, and checkpoint exact-resume. Only after that should the architecture and data format be frozen.
