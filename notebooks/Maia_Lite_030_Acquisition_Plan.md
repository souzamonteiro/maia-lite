# Maia Lite Chat 0.3.0 — Verified Corpus Acquisition Plan

**Date:** 2026-10-18  
**Status:** Proposed source catalog; **zero datasets automatically approved**.  
**Model:** approximately 354.8M parameters.  
**Training target:** 21,289,390,080 processed tokens (3× a 20-token/parameter Chinchilla baseline).  
**Target language exposure:** Portuguese 40%, English 35%, Spanish 25%.  
**Provisional domain exposure:** general language 30%, dialogue 20%, logic 25%, educational 15%, creative 10%. These domain shares are experimental hypotheses, not validated optimal proportions.

## 1. Executive decision

Prioritize FineWeb2 for Portuguese and Spanish general-language coverage, and a complementary English collection (FineWeb-Edu and, subject to review, Dolma); use Wikimedia Wikipedia for multilingual encyclopedic material; OASST1 for *authentic human* dialogue; keep UltraChat/Smol-SmolTalk in a **separate synthetic SFT candidate pool**, not counted toward authentic dialogue. Use carefully licensed academic/educational sources as enrichment. Reserve benchmark datasets and their near duplicates for evaluation, never casually mix them into training.

**Do not initiate full-scale downloads or training before legal/rights review, sampling, storage estimation, tokenizer pilot and leakage audits.** Dataset-level licenses may not cover every original source or downstream redistribution scenario.

## 2. Verified acquisition catalog

| Priority | Dataset / URL | Languages | Proposed domain | Scale evidence | Declared license / caveat | Decision |
|---|---|---|---|---|---|---|
| P0 | [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | PT, ES; multilingual (EN coverage should be checked at configuration level) | language | Over 1000 languages, Common Crawl snapshots 2013–2024 | ODC-BY 1.0; also subject to Common Crawl terms | Sample after review |
| P0 | [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | PT, EN, ES | language, educational | 61.6M rows across all language configurations; 71.8GB total listed | CC BY-SA/GFDL and page-specific terms; attribution and share-alike review | Sample after review |
| P0 | [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | EN, ES, pt-BR, others | dialogue (human) | ~84.4k train rows overall, not all in target languages | Apache-2.0; review privacy, moderation, provenance and exact language counts | Sample after review |
| P1 | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Primarily EN | educational | 1.3T tokens advertised | Verify current dataset terms and Common Crawl obligations | Sample after review |
| P1 | [peS2o](https://huggingface.co/datasets/allenai/peS2o) | Mainly EN | educational / scientific | ~40M open-access paper records advertised | ODC-BY; original paper terms, metadata and derivative permissions require review | Sample after review |
| P1 | [Dolma](https://huggingface.co/datasets/allenai/dolma) | EN | language / creative / educational | 3T tokens; v1.7 ~4.5TB gzip | ODC-BY dataset with original-source terms; verify gated access and version | Evaluate selected shards only |
| P2 | [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | EN | dialogue, synthetic | ~208k train_sft examples | MIT declared; generated-data upstream terms and contamination review | Separate SFT pool |
| P2 | [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | EN | dialogue, synthetic | ~460k train examples | Apache-2.0 declared; synthetic | Separate SFT pool |
| EVAL | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | EN | math benchmark | ~7.5k train and ~1.3k test in main configuration | MIT | Freeze test set, audit overlap |
| EVAL | [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | EN | reasoning benchmark | Small science QA benchmark | CC BY-SA 4.0 | Freeze test/validation, audit overlap |

**Not counted as an independent family:** language subsets, Common Crawl derivatives and mirrors of the same original pages. FineWeb2 and FineWeb-Edu can overlap. Dolma can overlap Wikipedia, Gutenberg and other web sources. Source-family isolation and cross-collection deduplication are required.

## 3. Training exposure budget (not unique token requirements)

Total: **21,289,390,080 processed tokens**.

| Language | Share | Processed tokens |
|---|---:|---:|
| PT | 40% | 8,515,756,032 |
| EN | 35% | 7,451,286,528 |
| ES | 25% | 5,322,347,520 |

| Domain | Share | Processed tokens |
|---|---:|---:|
| General language | 30% | 6,386,817,024 |
| Dialogue | 20% | 4,257,878,016 |
| Logic | 25% | 5,322,347,520 |
| Educational | 15% | 3,193,408,512 |
| Creative | 10% | 2,128,939,008 |

These are **exposure targets**; no claim is made that equivalent unique-token pools exist. Repetition ratios must be measured per dataset, language and domain, and constrained empirically. A large 25% logic share should be tested against a less-specialized baseline.

## 4. Phase-gated implementation

### Phase A — Source authorization (no large downloads)

1. Snapshot each dataset card, terms, license and version/commit hash; capture exact configuration, split, original source provenance and attribution requirements.
2. Review training, commercial use, redistribution, privacy and personal data issues. Preserve `pending_review` until a human records evidence and decision.
3. Freeze the evaluation suite, including held-out prompts for open-ended instructions in PT/EN/ES, plus test splits for GSM8K/ARC. Maintain hashes and audit against all training sources.

### Phase B — Measured 1–5M-token pilot per dataset/language (after authorization)

1. Stream selected dataset configurations into a local or mounted scratch location with a hard byte and record limit; avoid unbounded Colab Drive writes.
2. Preserve original document IDs, URLs, snapshots and hashes; classify `human`, `synthetic`, or `unknown` conversation provenance.
3. Run quality, language-ID, PII, boilerplate, exact/fuzzy dedup and benchmark contamination checks. Report reject rates and accepted tokens using a candidate tokenizer.
4. Estimate compressed storage, bytes per token, CPU time and accepted yield from **actual samples**, not guesses.

### Phase C — Scale only eligible sources

1. Choose a unique-token pool based on measured yield, diversity and repeat rates; never conflate unique tokens with processed tokens.
2. Stream processing with resumable manifests, per-shard SHA-256, checkpoints and reproducible version pins. Prefer scratch SSD/object storage over writing each intermediate to Google Drive.
3. Deduplicate across source families before splits. Split by document and near-duplicate cluster; reserve independent evaluation families where feasible.
4. Prepare candidate tokenizer on an approved, balanced sample; compare 32k/40k/50k vocabularies for fertility, coverage and downstream benchmarks. Freeze tokenizer before token accounting.

### Phase D — Train with scientific checkpoints

Measure actual tokens/sec, CU/hour and optimizer-state resume. Proposed exposure checkpoints at ~5.32B, 10.64B, 15.97B, 21.29B. Compare per-language validation perplexity, open-ended prompt following, dialogue continuity, logic accuracy and memorization. Evaluate the 25% logic mixture against a lower-logic ablation if resources permit.

## 5. Required acquisition ledger fields

`dataset_id`, `upstream_url`, `dataset_revision`, `configuration`, `split`, `language`, `domain`, `original_source_family`, `human_or_synthetic`, `license_declared`, `license_evidence_url`, `upstream_terms`, `reviewer`, `review_date`, `training_approved`, `redistribution_approved`, `attribution`, `sample_records`, `sample_raw_bytes`, `sample_proxy_tokens`, `sample_final_tokens`, `accepted_tokens`, `exact_duplicate_rate`, `near_duplicate_rate`, `pii_flags`, `benchmark_overlap`, `storage_estimate_bytes`, `download_status`.

**Acceptance gate:** `training_approved` is a human-reviewed field; no dataset card license alone can flip it to true.

## 6. Immediate next actions

1. Review and authorize the **FineWeb2 Portuguese and Spanish configurations** for a bounded sample; verify their exact configuration names and current access requirements.
2. Review and sample **OASST1**, measuring genuine PT-BR, ES and EN conversation tokens and inspecting language tags and moderation.
3. Review and sample **Wikipedia PT/EN/ES** as a multilingual anchor, retaining page identifiers and source URLs.
4. Add **FineWeb-Edu** as a separately audited educational stream; measure overlap with FineWeb2 and other web sources.
5. Establish the frozen benchmark suite **before** scaling corpus acquisition.

## 7. Source documentation consulted

- FineWeb2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/main/README.md
- Wikipedia: https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/README.md
- OpenAssistant: https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/README.md
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- peS2o: https://huggingface.co/datasets/allenai/peS2o
- Dolma: https://huggingface.co/datasets/allenai/dolma
- UltraChat: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- Smol-SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- ARC: https://huggingface.co/datasets/allenai/ai2_arc
