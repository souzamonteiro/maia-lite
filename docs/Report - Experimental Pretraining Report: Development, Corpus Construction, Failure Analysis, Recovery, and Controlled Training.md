Maia Lite

Experimental Pretraining Report: Development, Corpus Construction, Failure Analysis, Recovery, and Controlled Training

Project: Maia Lite
Model: Decoder-only Transformer language model
Model size: 354,823,168 parameters
Languages: English, Brazilian Portuguese, Spanish
Additional domains: Source code, mathematics, scientific literature
Primary training platform: NVIDIA A100-SXM4-80GB / Google Colab
Status: Pretraining in progress
Current controlled target: Equivalent step 216,567
Project: Maia Platform

---

1. Executive Summary

Maia Lite is an experimental compact multilingual large language model developed as part of the Maia Platform ecosystem. The project investigates how far a relatively small decoder-only Transformer model, containing approximately 355 million parameters, can be pushed through controlled pretraining, carefully designed multilingual data composition, deterministic corpus materialization, domain diversification, and subsequent supervised fine-tuning.

The project targets a model sufficiently compact for local, edge, embedded, educational, and offline applications while retaining useful multilingual and domain-specific language capabilities.

The experiment evolved through multiple training phases. Early training demonstrated successful loss reduction but subsequently exposed a reproducible instability associated with the original streaming data pipeline. Rather than treating this event merely as a failed run, the project developed a controlled diagnostic methodology involving checkpoint restoration, frozen-loss evaluation, deterministic data materialization, and reconstruction of the training pipeline.

This investigation led to the creation of Maia Corpus V2, a deterministic 2.9-billion-token multilingual and multidisciplinary training corpus.

The new corpus contains English, Portuguese, Spanish, source-code-related material, mathematical material, and scientific literature.

A separate frozen validation corpus containing approximately 20 million tokens was constructed from non-overlapping material and has remained unchanged throughout subsequent experiments.

Using the corrected deterministic pipeline, the Maia Lite model successfully resumed training from a verified checkpoint and demonstrated substantial improvement on every validation domain.

The global validation loss decreased from:

[
3.86998176
]

at equivalent step 130,000 to:

[
3.60799408
]

at equivalent step 140,000.

This corresponds to a perplexity reduction from:

[
47.941512
]

to:

[
36.891976
]

or approximately 23.05%.

All six validation domains improved independently.

A subsequent continuous training trajectory from equivalent step 140,000 toward equivalent step 216,567 is currently underway. At approximately equivalent step 205,000, training loss had reached approximately 3.50, while maintaining a stable downward trend without reproducing the catastrophic behavior observed in the earlier streaming pipeline.

Equivalent step 216,567 is especially significant because it corresponds almost exactly to approximately 20 training tokens per model parameter:

[
20 \times 354,823,168

7,096,463,360
]

tokens.

The planned cumulative training token count at step 216,567 is approximately:

[
216,567\times32,768

7,096,467,456
]

tokens.

The difference between the theoretical 20-token-per-parameter reference and the actual planned token count is only 4,096 tokens.

This checkpoint will therefore constitute a particularly useful experimental reference point for evaluating the behavior of a compact multilingual model around the classical compute/data scaling regime.

---

2. Research Motivation

Modern language-model development increasingly emphasizes models containing billions or hundreds of billions of parameters. This direction produces remarkable capabilities but imposes significant computational, financial, memory, energy, and deployment requirements.

Maia Lite explores the complementary research question:

«How far can a compact 355M-parameter multilingual language model be pushed through controlled corpus composition, continued pretraining, and domain-aware fine-tuning?»

The project is motivated by several potential deployment scenarios:

- edge AI;
- embedded inference;
- offline assistants;
- CPU-based inference;
- low-memory devices;
- educational applications;
- private local inference;
- specialized scientific assistants;
- retrieval-augmented generation;
- multilingual applications;
- compact domain-specific models.

The objective is therefore not to compete directly with frontier-scale general-purpose models.

Instead, Maia Lite investigates the capability-to-size and capability-to-compute relationship of a carefully trained compact model.

---

3. Model Architecture

Maia Lite uses a GPT-style autoregressive decoder architecture implemented using the Hugging Face "GPT2LMHeadModel" infrastructure.

The model contains:

[
\boxed{354,823,168\text{ parameters}}
]

The tokenizer vocabulary contains:

[
\boxed{50,257\text{ tokens}}
]

The principal tokenizer identifiers used throughout training are:

- vocabulary size: 50,257;
- EOS token ID: 2;
- PAD token ID: 1.

The training context length is:

[
\boxed{1,024\text{ tokens}}
]

The architecture is trained using standard causal language modeling.

No masked-language-model objective is used.

---

4. Original Training Phase

The initial Maia Lite pretraining experiment used multilingual Wikipedia material and a streaming data pipeline.

Training initially behaved normally.

Loss decreased substantially from approximately 9 during the early stages to values around the low 3 range after extended training.

A checkpoint at equivalent step 106,000 became particularly important because it preceded a later reproducible training instability.

The checkpoint:

"checkpoint-106000"

was therefore preserved as a golden recovery checkpoint.

---

5. Discovery of Training Instability

After approximately step 106,000, an anomalous increase in loss was observed.

Representative values included:

Equivalent step| Training loss
106,100| 2.716843
106,400| 2.604624
106,500| 2.541471
106,800| 2.870781
106,900| 3.208600
107,000| 3.402284
107,100| 3.531812
107,200| 3.613284
107,300| 3.653657

The behavior was incompatible with ordinary minibatch variance.

The rapid and persistent loss increase indicated either:

1. parameter degradation;
2. optimizer/scheduler instability;
3. corrupted checkpoint state;
4. data-distribution discontinuity;
5. or a problem associated with the streaming input pipeline.

---

6. Frozen-Loss Diagnostic Experiment

To distinguish model corruption from training-pipeline problems, the 106k checkpoint was evaluated without performing any optimization.

The experiment deliberately used:

- no gradients;
- no optimizer;
- no scheduler;
- no weight updates.

Representative results were:

Equivalent position| Mean loss
100| 2.752875
200| 2.686952
300| 2.512
400| 2.542

These results demonstrated that the model weights stored at checkpoint 106k were not intrinsically corrupted.

The checkpoint remained capable of producing substantially lower loss when evaluated under controlled conditions.

This shifted the investigation toward the training state and data pipeline.

---

7. Controlled Recovery Experiment

A controlled recovery was initiated by loading only the model weights from checkpoint 106k.

The following state was deliberately discarded:

- optimizer;
- learning-rate scheduler;
- Trainer state;
- gradient scaler.

New training state was created.

The recovery experiment used a conservative learning rate of:

[
1\times10^{-5}
]

and initially targeted approximately 2,000 controlled steps.

This strategy successfully restored stable learning.

A longer deterministic continuation subsequently reached an equivalent step of approximately 130,000 without reproducing the catastrophic loss behavior.

This established that the 106k model itself was recoverable.

---

8. Root-Cause Investigation

Repeated experiments indicated that the earlier instability was strongly associated with the streaming training pipeline rather than the Transformer architecture itself.

A particularly important observation was that degradation could recur at a similar offset after restarting training sessions.

The evidence was consistent with problematic interaction among:

- live streaming datasets;
- dataset restart behavior;
- Trainer resume behavior;
- sample skipping;
- and "ignore_data_skip=True".

The resulting pipeline could not guarantee the deterministic continuation required for a long scientific pretraining experiment.

The project therefore adopted a new principle:

«Training data must be physically materialized, auditable, deterministic, and independently reproducible before long-duration pretraining.»

This decision led to Maia Corpus V2.

---

9. Maia Corpus V2

Maia Corpus V2 was designed as a multilingual and multidisciplinary training corpus.

Its intended composition is:

Source| Proportion
FineWeb-Edu English| 45.0%
FineWeb2-HQ Portuguese| 22.5%
FineWeb2-HQ Spanish| 22.5%
OpenCoder FineWeb Code| 5.0%
FineMath-4+| 3.0%
peS2o Scientific| 2.0%

Total:

[
100%
]

The corpus therefore contains approximately:

- 90% general multilingual natural language;
- 5% code-oriented material;
- 3% mathematics;
- 2% scientific literature.

---

10. Corpus V2 Physical Composition

The materialized training corpus contains:

Component| Tokens
FineWeb-Edu EN| 1,304,999,936
FineWeb2-HQ PT| 652,499,968
FineWeb2-HQ ES| 652,499,968
OpenCoder| 144,999,424
FineMath| 86,999,040
peS2o| 58,001,408

Total:

[
\boxed{2,899,999,744\text{ tokens}}
]

The materialized corpus consists of:

[
184\text{ shards}
]

and occupies:

[
5,799,999,488\text{ bytes}
]

using unsigned 16-bit token identifiers.

---

11. Unified Corpus Artifact

The 184 training shards were concatenated into one canonical binary artifact in the following fixed source order:

1. FineWeb-Edu English;
2. FineWeb2-HQ Portuguese;
3. FineWeb2-HQ Spanish;
4. OpenCoder;
5. FineMath;
6. peS2o.

The resulting unified corpus contains:

[
2,899,999,744\text{ tokens}
]

or:

[
2,832,031
]

complete 1,024-token sequences.

The artifact size is:

[
5,799,999,488\text{ bytes}
]

The official SHA-256 digest is:

"57e3a10aac2473dd8af428dba2fd35031410f5e10d77fa3d6933c0cd92503ca2"

Both the local SSD copy and Google Drive copy were independently hashed and found to be identical.

This hash serves as the canonical identity of the training artifact.

---

12. Frozen Validation Corpus V2

A separate validation corpus was created and permanently excluded from training.

The validation corpus contains approximately 20 million raw tokens.

Its composition mirrors the six principal training domains.

Domain| Raw tokens
FineWeb-Edu EN| 8,999,936
FineWeb2-HQ PT| 4,499,456
FineWeb2-HQ ES| 4,499,456
OpenCoder| 999,424
FineMath| 599,040
peS2o| 402,432

Total:

[
\boxed{19,999,744\text{ raw tokens}}
]

The corpus contains:

[
19,531
]

sequences.

The validation material starts after the document offsets consumed during construction of the training corpus, preventing direct training-validation overlap.

The validation corpus has subsequently remained frozen.

---

13. Controlled 130k Baseline

The recovered deterministic training pipeline produced a stable equivalent-130k model.

This model was evaluated against the frozen Validation Corpus V2.

Results were:

Domain| Loss| Perplexity
FineWeb-Edu EN| 3.89186480| 49.002180
FineWeb2-HQ PT| 3.95476403| 52.183379
FineWeb2-HQ ES| 3.70961940| 40.838260
OpenCoder| 4.31978168| 75.172215
FineMath| 3.45431165| 31.636504
peS2o| 3.72731316| 41.567273

Global result:

[
\boxed{L_{130k}=3.86998176}
]

with:

[
\boxed{PPL_{130k}=47.941512}
]

over:

[
19,980,213
]

predicted tokens.

This result became the official pre-Corpus-V2 baseline.

---

14. Official 130k → 140k Corpus V2 Experiment

The first controlled training experiment using the unified Corpus V2 continued the 130k model for 10,000 optimizer steps.

Configuration:

- sequence length: 1,024;
- microbatch: 16;
- gradient accumulation: 2;
- effective batch: 32;
- tokens per optimizer step: 32,768;
- learning rate: 1\times10^{-5};
- warmup: 500 steps;
- cosine learning-rate scheduler;
- weight decay: 0.1;
- maximum gradient norm: 1.0;
- BF16 enabled;
- TF32 enabled;
- gradient checkpointing enabled;
- deterministic map-style memory-mapped dataset;
- seed: 42;
- data seed: 42.

The training processed:

[
10,000\times32,768

327,680,000
]

Corpus V2 tokens.

Runtime was approximately:

[
3\text{ h }40\text{ min}
]

on an NVIDIA A100-SXM4-80GB.

Peak allocated GPU memory was approximately:

[
17.15\text{ GB}
]

and peak reserved memory approximately:

[
19.48\text{ GB}.
]

Training loss remained stable and decreased from approximately 3.968 at the beginning to approximately 3.666 near the end.

---

15. Official 140k Validation

The resulting equivalent-140k model was evaluated against exactly the same frozen validation corpus.

Results:

Domain| 130k loss| 140k loss| Δ loss
English| 3.891865| 3.628043| -0.263822
Portuguese| 3.954764| 3.689490| -0.265274
Spanish| 3.709619| 3.485267| -0.224352
Code| 4.319782| 3.977261| -0.342521
Mathematics| 3.454312| 3.073105| -0.381207
Scientific| 3.727313| 3.499774| -0.227540

Global validation:

[
3.86998176
\rightarrow
3.60799408
]

The corresponding perplexity changed from:

[
47.941512
\rightarrow
36.891976
]

representing approximately:

[
\boxed{23.05%}
]

reduction in global perplexity.

Every evaluated domain improved.

The strongest relative improvements occurred in mathematical and code-oriented material.

This result provided strong evidence that Corpus V2 was improving generalization rather than merely reducing training loss.

---

16. Continuous 140k → 216,567 Experiment

Following the successful 140k validation, a longer controlled trajectory was initiated.

The trajectory begins from the 140k model weights with:

- new optimizer;
- new scheduler;
- new Trainer state.

The scheduler horizon was defined from the beginning as:

[
76,567\text{ optimizer steps}
]

corresponding to:

[
216,567-140,000.
]

The configuration is:

Parameter| Value
Microbatch| 16
Gradient accumulation| 2
Effective batch| 32
Context| 1,024
Tokens/step| 32,768
Maximum LR| 1\times10^{-5}
Warmup| 250 steps
Scheduler| Cosine
Weight decay| 0.1
Max gradient norm| 1.0
BF16| Enabled
TF32| Enabled
Gradient checkpointing| Enabled

Total tokens scheduled during this trajectory:

[
76,567\times32,768

\boxed{2,508,947,456}
]

tokens.

---

17. Training-Loss Evolution During the Long Trajectory

The trajectory has exhibited stable improvement.

Representative observations include:

Trajectory step| Equivalent step| Approx. training loss
1| 140,001| 3.6688
5,000| 145,000| 3.6286
10,000| 150,000| 3.5822
20,000| 160,000| 3.5675
25,000| 165,000| 3.5446
26,000| 166,000| 3.5266
65,100| 205,100| 3.4969

Although individual minibatches fluctuate, the overall envelope exhibits a persistent downward trend.

Crucially, the catastrophic increase previously observed with the original streaming pipeline has not reappeared.

This provides additional evidence that the earlier instability originated primarily from the data/training pipeline rather than from an intrinsic optimization instability in the model.

---

18. Successful Full-State Resume

The long trajectory was interrupted by the execution environment and subsequently resumed from:

"checkpoint-65000"

corresponding to:

[
140,000+65,000=
205,000
]

equivalent Maia steps.

The resume successfully restored:

- model parameters;
- optimizer;
- learning-rate scheduler;
- Trainer state.

The remaining training length was:

[
11,567\text{ optimizer steps}
]

or:

[
379,027,456\text{ tokens}.
]

Immediately following resume, a representative training loss of:

[
3.496942
]

was observed at trajectory step 65,100.

No discontinuity or catastrophic degradation was observed following resume.

This provides evidence that the deterministic materialized-corpus pipeline supports reliable interruption and continuation.

---

19. Compute Reference Point at 216,567

The model contains:

[
N=354,823,168
]

parameters.

Using the heuristic reference of approximately 20 training tokens per parameter gives:

[
20N=
7,096,463,360.
]

At equivalent step 216,567, with 32,768 processed tokens per optimizer step, the approximate cumulative accounting is:

[
216,567\times32,768

7,096,467,456.
]

Thus:

[
\frac{D}{N}\approx20.00001.
]

The difference from exactly 20N is only:

[
4,096\text{ tokens}.
]

The equivalent-216,567 checkpoint therefore provides a convenient experimental reference at approximately 20 tokens per parameter.

It should not, however, be interpreted automatically as the optimal stopping point.

Actual stopping decisions will be based on empirical validation behavior.

---

20. Current Compute Economics

The current A100 environment consumes approximately:

[
6.77\text{ compute units/hour}.
]

At one recorded point, the project retained:

[
378.71\text{ compute units}.
]

This corresponds theoretically to approximately:

[
55.94\text{ A100 hours}
]

before accounting for operational overhead.

Observed training throughput during the current trajectory has been approximately:

[
0.76-0.77\text{ optimizer steps/s}.
]

This compute reserve is sufficient to complete the current experiment and potentially support further pretraining and supervised fine-tuning.

Additional compute may be acquired if experimental evidence demonstrates that continued training remains beneficial.

---

21. Potential Extended Training to 300k

If validation at 216,567 demonstrates continued substantial improvement, an extended trajectory may be performed.

The candidate endpoint is:

[
300,000\text{ equivalent steps}.
]

Additional optimizer steps required:

[
300,000-216,567

83,433.
]

Additional processed tokens:

[
83,433\times32,768

2,734,? \text{ billion tokens approximately}.
]

The cumulative token count at 300k would be:

[
300,000\times32,768

9,830,400,000
]

tokens.

Thus:

[
\frac{9,830,400,000}
{354,823,168}
\approx
27.7
]

tokens per parameter.

This would provide an experimentally useful comparison between approximately:

- 20 tokens/parameter;
- 27.7 tokens/parameter.

The extension will only be initiated after evaluation of the 216,567 checkpoint.

---

22. Planned Evaluation at 216,567

The 216,567 checkpoint will be permanently preserved.

The first evaluation will use the frozen Validation Corpus V2.

The same metrics used for the 130k and 140k checkpoints will be calculated:

- global validation loss;
- global perplexity;
- English loss/perplexity;
- Portuguese loss/perplexity;
- Spanish loss/perplexity;
- code loss/perplexity;
- mathematics loss/perplexity;
- scientific-text loss/perplexity.

This will produce the first three-point controlled validation trajectory:

[
130k
\rightarrow
140k
\rightarrow
216,567.
]

Further checkpoints may subsequently include:

[
300k.
]

---

23. Base-Model Generation Evaluation

Loss alone will not determine model usefulness.

Before supervised fine-tuning, the base model will also be evaluated qualitatively through natural continuation tasks.

These tests should avoid chat-style instructions because the base model has not yet been instruction-tuned.

Tests will include natural text continuations in:

- English;
- Portuguese;
- Spanish;
- scientific prose;
- mathematics;
- source code.

The objective is to determine whether the pretrained model has acquired coherent generative structure independently of instruction-following behavior.

---

24. Fine-Tuning Strategy

Following completion of pretraining, Maia Lite is expected to undergo unified supervised fine-tuning.

The planned major categories are:

1. multilingual conversation;
2. Python/programming;
3. computational modeling and scientific material.

English, Portuguese, and Spanish will remain central languages.

A unified fine-tuning strategy is preferred to isolated sequential fine-tuning because sequential specialization may increase catastrophic forgetting.

The final fine-tuned model will subsequently be evaluated separately from the pretrained base model.

---

25. Edge and Embedded Evaluation

A central research objective is deployment efficiency.

Following pretraining and fine-tuning, the model should therefore be evaluated in multiple numerical representations, potentially including:

- BF16/FP16;
- GGUF high-precision quantization;
- Q8;
- Q4 or comparable compact quantization.

Evaluation should include:

- model size;
- RAM requirement;
- initialization time;
- tokens per second;
- CPU inference;
- Apple Silicon inference;
- conventional x86 inference;
- potentially ARM/embedded environments where practical.

The objective is to characterize the trade-off:

[
\text{quality}
\leftrightarrow
\text{memory}
\leftrightarrow
\text{latency}
\leftrightarrow
\text{compute}.
]

---

26. Scientific Contributions Emerging from the Experiment

The Maia Lite project potentially contributes in several distinct ways.

26.1 Compact multilingual modeling

The model investigates useful multilingual capability at approximately 355M parameters rather than at multi-billion-parameter scale.

26.2 Controlled multilingual corpus composition

Corpus V2 explicitly balances English, Portuguese, and Spanish while incorporating code, mathematics, and scientific text.

26.3 Deterministic corpus materialization

The training artifact is physically materialized, sequence-aligned, audited, and cryptographically identified.

26.4 Frozen multidomain validation

The project uses a persistent non-training validation corpus across model checkpoints.

26.5 Failure analysis

The project documents a significant training failure associated with the original streaming pipeline and the experimental methodology used to isolate it.

26.6 Reliable checkpoint recovery

The corrected pipeline demonstrates full-state recovery without the previously observed degradation.

26.7 Training-budget analysis

Checkpoints near different tokens-per-parameter regimes enable empirical analysis of continued training beyond conventional scaling references.

26.8 Edge-oriented language modeling

The final model is explicitly intended for environments where very large models are impractical.

---

27. Reproducibility Assets

The project has preserved or can preserve:

- tokenizer;
- tokenizer vocabulary and merges;
- training scripts;
- corpus-construction scripts;
- source manifests;
- deterministic corpus shards;
- unified training corpus;
- corpus SHA-256;
- frozen validation corpus;
- checkpoints;
- Trainer state;
- optimizer state;
- scheduler state;
- training logs;
- validation results;
- hardware information;
- CUDA version;
- PyTorch version;
- training hyperparameters;
- random seeds;
- compute consumption.

This collection provides the basis for a substantially reproducible experimental report.

Before public redistribution of derived datasets, the licensing and redistribution conditions of every upstream source must be reviewed individually.

Where direct redistribution is not permitted, reconstruction scripts, source identifiers, filtering criteria, manifests, and hashes can be considered instead.

---

28. Important Methodological Distinction: Training Loss vs. Model Capability

A fixed numerical training-loss threshold should not be used as the sole criterion for declaring the model successful or unsuccessful.

Cross-entropy depends on several factors, including:

- tokenizer;
- vocabulary;
- corpus distribution;
- sequence construction;
- evaluation methodology;
- domain;
- model architecture.

Therefore, a claim such as “a useful language model must have loss between 1.5 and 2.0” is not methodologically sufficient.

The Maia Lite evaluation will instead combine:

1. frozen validation loss;
2. validation perplexity;
3. domain-specific validation;
4. natural text generation;
5. standardized external benchmarks;
6. multilingual evaluation;
7. code and mathematical evaluation;
8. post-SFT instruction following;
9. inference efficiency.

This multidimensional evaluation is substantially more informative than training loss alone.

---

29. Preliminary Interpretation

The evidence collected so far supports several preliminary conclusions.

First, the Maia Lite architecture is capable of stable optimization.

Second, the catastrophic loss increase observed during the earlier experiment was not sufficient evidence of irreversible model degradation. Frozen evaluation of the 106k checkpoint demonstrated that useful model weights remained intact.

Third, replacing the streaming pipeline with a deterministic materialized corpus eliminated the previously observed failure pattern during the subsequent controlled experiments.

Fourth, Corpus V2 produced substantial generalization improvements across every evaluated domain between equivalent steps 130k and 140k.

Fifth, the longer 140k → 216,567 trajectory has remained stable and continues to exhibit decreasing training loss.

These observations justify completing the current pretraining trajectory and evaluating the resulting 20-token-per-parameter checkpoint.

They do not yet establish the final quality or practical usefulness of Maia Lite.

Those conclusions require the planned validation, generation, benchmarking, fine-tuning, and deployment experiments.

---

30. Candidate Research Question

A central research question emerging from the project is:

«How far can a compact 355M-parameter multilingual language model be pushed through controlled corpus composition, reproducible pretraining, extended token budgets, and domain-aware fine-tuning?»

Secondary questions include:

- How does validation performance evolve between approximately 20 and 28 tokens per parameter?
- Which domains benefit most from continued pretraining?
- Can multilingual performance be maintained in a compact model without excessive specialization?
- How much capability survives aggressive quantization?
- What quality/compute trade-offs emerge for edge deployment?
- How does domain-aware SFT modify the capabilities acquired during pretraining?

---

31. Candidate Paper Direction

A possible working title is:

Maia Lite: Reproducible Pretraining and Evaluation of a 355M-Parameter Multilingual Language Model for Edge AI

The study could potentially be structured around:

1. motivation and research questions;
2. architecture;
3. tokenizer;
4. Corpus V2;
5. deterministic preprocessing;
6. validation methodology;
7. training-failure investigation;
8. controlled recovery;
9. scaling experiments;
10. supervised fine-tuning;
11. multilingual benchmarks;
12. domain-specific benchmarks;
13. quantization;
14. edge inference;
15. compute efficiency;
16. limitations;
17. reproducibility.

The final publication venue should be selected after the principal experimental results are available.

An engineering-oriented presentation emphasizing reproducibility, compact LLM training, edge deployment, and extensive evaluation may fit venues such as IEEE Access, while other venues may be considered depending on the final scientific contribution.

---

32. Current Status

At the time represented by this report:

- Maia Corpus V2 is complete;
- the unified training artifact has been cryptographically verified;
- Validation Corpus V2 is frozen;
- the 130k baseline has been evaluated;
- the 140k checkpoint has been evaluated;
- all validation domains improved from 130k to 140k;
- the 140k → 216,567 controlled trajectory is in progress;
- full-state checkpoint resume has been successfully demonstrated;
- training loss remains on a stable downward trajectory;
- equivalent step 216,567 is expected to represent approximately 20 tokens per parameter.

The immediate next experiment is therefore:

[
\boxed{\text{Frozen evaluation of Maia Lite at equivalent step 216,567}}
]

The result of this experiment will determine whether pretraining is concluded or extended toward approximately 300,000 equivalent steps.

---

33. Evidence Preservation Principle

No unsuccessful experiment should be discarded from the scientific record merely because it failed to produce the intended model.

For Maia Lite, the failed streaming trajectory, frozen-loss diagnosis, controlled checkpoint recovery, deterministic corpus redesign, and successful subsequent training constitute a coherent experimental sequence.

Accordingly, the project should preserve:

- raw logs;
- failed-run checkpoints where useful;
- recovery scripts;
- diagnostic results;
- corpus-generation scripts;
- hashes;
- successful-run logs;
- validation outputs;
- environment information.

The final scientific account should clearly distinguish:

- observations;
- hypotheses;
- interventions;
- experimental results;
- interpretations.

This distinction will be maintained throughout subsequent Maia Lite experiments.

---

34. Next Experimental Milestone

The next milestone is the completion and permanent preservation of:

Maia Lite — Equivalent Step 216,567

followed by:

1. integrity verification;
2. frozen Validation Corpus V2 evaluation;
3. domain-by-domain comparison with 130k and 140k;
4. base-model multilingual generation;
5. preliminary capability assessment;
6. decision regarding 216,567 → 300,000 continued pretraining.

Only after these results are available should the next pretraining trajectory be configured.

---

Report status: Living experimental document.
Next revision: After completion and evaluation of equivalent step 216,567.