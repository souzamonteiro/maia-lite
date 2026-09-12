# MAIA-Lite

![Maia](<images/ChatGPT Image 12 de set. de 2026, 10_05_17.png>)

### Machine Artificial Intelligence Architecture

**MAIA-Lite** is an open-source research project for building compact, understandable, multilingual Large Language Models from the ground up.

It is part of the **Maia Platform** ecosystem and explores the complete lifecycle of a language model: dataset construction, tokenization, Transformer architecture, pretraining, scientific knowledge acquisition, instruction fine-tuning, evaluation, and local deployment.

The project deliberately focuses on relatively small models that can be trained, studied, modified, and deployed using accessible computing infrastructure.

> **MAIA** stands for **Machine Artificial Intelligence Architecture**.

The long-term objective is not merely to fine-tune existing language models, but to understand and control the entire process required to construct specialized Maia language models.

---

## Maia-Lite-335M

The first model developed in this repository is:

**Maia-Lite-335M**

It is a decoder-only Transformer language model with approximately **335 million parameters**, designed as the first fully pretrained model in the Maia-Lite family.

The model is being developed using the Hugging Face ecosystem so that training artifacts remain compatible with standard open-source tools and can later be converted for efficient local inference, including deployment through Ollama.

The initial model is multilingual, with native training material in:

- English
- Portuguese
- Spanish

These languages were selected to support international academic and educational use while keeping the initial project computationally manageable.

---

# Project Goals

Maia-Lite has several complementary goals.

### Build an LLM from pretraining

Rather than beginning exclusively with an existing pretrained model, Maia-Lite investigates the complete construction of a language model starting from randomly initialized Transformer weights.

### Make the architecture understandable

The project treats the language model as an inspectable computational system rather than a remote black-box service.

Architecture, tokenizer, datasets, training configuration, checkpoints, evaluation procedures, and deployment artifacts should be documented and reproducible.

### Develop multilingual scientific capabilities

The first Maia-Lite models target English, Portuguese, and Spanish, with particular interest in scientific and academic language.

### Support local inference

Models should ultimately be deployable on locally controlled infrastructure whenever technically practical.

### Provide a foundation for future Maia models

Maia-Lite is intended as an experimental foundation from which larger and more specialized models can later be developed.

---

# Development Strategy

The first Maia-Lite training process is divided into two major stages:

```text
                    ┌──────────────────────────┐
                    │     Raw Text Corpora     │
                    │                          │
                    │  English                 │
                    │  Portuguese              │
                    │  Spanish                 │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Tokenization        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Pretraining        │
                    │                          │
                    │     Maia-Lite-335M       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Base Language Model    │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
                 ▼                                ▼
      ┌──────────────────────┐       ┌─────────────────────────┐
      │ Scientific Papers    │       │ General Instruction /   │
      │ Corpus               │       │ Evaluation Data         │
      │                      │       └─────────────────────────┘
      │ ~1,000 papers        │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ Text Extraction      │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ Question Generation  │
      │                      │
      │ ~10 questions/paper  │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ English QA Dataset   │
      └──────────┬───────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
   ┌─────────────┐ ┌─────────────┐
   │ Portuguese  │ │   Spanish   │
   │ Translation │ │ Translation │
   └──────┬──────┘ └──────┬──────┘
          │               │
          └───────┬───────┘
                  ▼
        ┌──────────────────────┐
        │ Multilingual         │
        │ Scientific Dataset   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Supervised           │
        │ Fine-Tuning          │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Scientific /         │
        │ Academic Maia-Lite   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Hugging Face Model   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ GGUF / Ollama        │
        │ Local Deployment     │
        └──────────────────────┘
```

---

# Stage 1 — Multilingual Pretraining

The initial **Maia-Lite-335M** model is pretrained using Wikipedia content in:

- English;
- Portuguese;
- Spanish.

Wikipedia provides a large multilingual corpus containing encyclopedic language across a broad range of subjects.

The objective of this stage is not yet to create an instruction-following assistant.

Pretraining teaches the model the statistical structure of language and provides broad representations of concepts, entities, syntax, relationships, and textual patterns.

The training objective is standard autoregressive next-token prediction:

```text
P(x1, x2, ..., xn)

=

P(x1)
P(x2 | x1)
P(x3 | x1, x2)
...
P(xn | x1, ..., xn-1)
```

or equivalently:

```text
Input:

Artificial intelligence is a branch of

Target:

intelligence is a branch of computer
```

Every position in the sequence therefore becomes a training example for next-token prediction.

---

# Stage 2 — Scientific Corpus

After pretraining, Maia-Lite undergoes specialization using scientific literature.

The initial corpus contains approximately:

**1,000 scientific papers**

selected from research areas relevant to Maia's academic objectives.

The papers are processed automatically to extract their textual content and associated metadata.

The objective is not simply to continue training on raw papers.

Instead, the scientific corpus is transformed into a structured instructional dataset.

---

# Scientific Question Generation

For every scientific paper, approximately:

**10 question-answer pairs**

are generated.

The first version of each question and answer is produced in **English**.

This results in an initial target of approximately:

```text
1,000 papers
      ×
10 questions per paper
      =
~10,000 English question-answer pairs
```

The generated dataset is subsequently filtered and validated before fine-tuning.

Questions should cover different levels of understanding whenever the source permits, including:

- factual understanding;
- definitions;
- methodology;
- interpretation of results;
- relationships between concepts;
- comparison of methods;
- mathematical reasoning;
- computational reasoning;
- limitations;
- implications of the research.

The objective is to teach the model to operate on scientific knowledge rather than merely reproduce sentences from papers.

---

# Multilingual Scientific Dataset

The English scientific question-answer dataset is translated into:

- Portuguese;
- Spanish.

The resulting dataset therefore contains three linguistic representations of the scientific material:

```text
                    Scientific Paper
                           │
                           ▼
                  English QA Generation
                           │
               ┌───────────┼───────────┐
               │           │           │
               ▼           ▼           ▼
            English    Portuguese    Spanish
               │           │           │
               └───────────┼───────────┘
                           │
                           ▼
                 Multilingual SFT
```

With approximately 10,000 validated questions represented in three languages, the theoretical dataset size becomes approximately:

```text
~10,000 English examples
~10,000 Portuguese examples
~10,000 Spanish examples
--------------------------------
~30,000 multilingual examples
```

The exact final size may be smaller because low-quality, redundant, ambiguous, unsupported, or otherwise unsuitable examples should be discarded.

Dataset quality is considered more important than reaching a predetermined example count.

---

# Dataset Quality

Automatically generated scientific datasets can easily reproduce hallucinations or interpretation errors from the teacher model.

For this reason, Maia-Lite treats dataset construction as an independent research problem.

The processing pipeline should progressively incorporate:

- source-document traceability;
- paper metadata;
- question-answer grounding;
- duplicate detection;
- semantic similarity detection;
- answer consistency checks;
- language validation;
- translation validation;
- difficulty classification;
- subject classification;
- automated rejection criteria;
- manual inspection of representative samples.

Whenever possible, every generated question should retain a reference to the source paper from which it was derived.

This allows later auditing and dataset reconstruction.

---

# Avoiding Evaluation Leakage

Questions generated from the same scientific paper are highly correlated.

Therefore, training, validation, and test datasets should preferably be separated at the **document level**, rather than randomly splitting individual questions.

For example:

```text
papers 0001–0800 → training
papers 0801–0900 → validation
papers 0901–1000 → test
```

rather than:

```text
randomly distributing questions from every paper
across training, validation, and test sets
```

Document-level separation provides a more realistic measurement of whether Maia-Lite can generalize scientific knowledge and reasoning to previously unseen documents.

---

# Technology Stack

The training pipeline is built around the open-source Python machine-learning ecosystem.

Primary technologies include:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Tokenizers
- Accelerate
- PEFT
- bitsandbytes where appropriate
- Google Colab during early experiments
- GGUF-compatible tooling for local inference
- Ollama for deployment and experimentation

Using standard Hugging Face model structures also makes checkpoints easier to inspect, publish, reuse, fine-tune, and convert.

---

# Google Colab

The first **Maia-Lite-335M** pretraining experiments are designed to run in Google Colab.

This provides a known and reproducible accelerator environment for validating:

- tokenizer construction;
- dataset preprocessing;
- Transformer configuration;
- training scripts;
- checkpointing;
- evaluation;
- model generation;
- Hugging Face compatibility.

Once the pipeline has been validated, the same architecture and datasets can be migrated to larger computational infrastructure.

---

# Model Lifecycle

The intended lifecycle of a Maia-Lite model is:

```text
Corpus
  ↓
Cleaning
  ↓
Tokenizer Training
  ↓
Tokenization
  ↓
Transformer Initialization
  ↓
Pretraining
  ↓
Base Checkpoint
  ↓
Scientific / Instruction Dataset
  ↓
Fine-Tuning
  ↓
Evaluation
  ↓
Hugging Face Model
  ↓
Conversion
  ↓
GGUF
  ↓
Ollama
  ↓
Local Maia Deployment
```

This separation is important.

The Hugging Face checkpoint is considered the primary trainable model representation.

GGUF and Ollama artifacts are deployment representations derived from that model.

---

# Repository Philosophy

Maia-Lite follows a simple principle:

> **Understanding language models by building them.**

The objective is not only to obtain a working `.gguf` file.

Every stage should remain understandable:

```text
Where did the data come from?

How was it cleaned?

How was the tokenizer trained?

Why does the Transformer have this architecture?

How many parameters does it contain?

How was it pretrained?

Which scientific material was used?

How were instruction examples generated?

Which examples were rejected?

How was the model evaluated?

How was the final deployment model produced?
```

These questions should be answerable from the repository itself.

---

# Reproducibility

Whenever practical, Maia-Lite experiments should record:

```text
experiment/
├── config.json
├── tokenizer/
├── dataset-manifest.json
├── training-arguments.json
├── checkpoints/
├── logs/
├── evaluations/
└── README.md
```

A model without information about its data and training procedure is difficult to reproduce scientifically.

For that reason, Maia-Lite treats documentation and provenance as part of the model itself.

---

# Maia-Lite Family

The 335M model is intended to be the first member of a family rather than the final objective.

```text
MAIA
Machine Artificial Intelligence Architecture
│
├── Maia-Lite
│   │
│   ├── Maia-Lite-335M
│   │
│   └── Future compact models
│
├── Maia Academic
│
├── Maia Chat
│
├── Maia RAG
│
└── Future Maia models and services
```

The smaller Maia-Lite models provide a controlled environment in which architectures, datasets, training strategies, multilingual behavior, and deployment methods can be tested before being applied to larger models.

---

# Technical Documentation

This repository includes a detailed technical introduction to Transformer-based language models:

**Transformer-Based Large Language Models: Algorithms, Mathematics, Pretraining, QLoRA Fine-Tuning, and Two Complete Educational Implementations in Python**

The guide covers the mathematics, algorithms, training process, QLoRA fine-tuning, and complete educational implementations of Transformer language models.

It is intended both as documentation for Maia-Lite and as educational material for readers interested in understanding how modern LLMs work internally.

---

# AI-Assisted Development

Artificial intelligence tools are used as part of the Maia-Lite research and software-development workflow.

Technical decisions, research objectives, dataset strategy, experiments, validation, and project direction remain human-directed.

For the Transformer technical guide included with this repository:

**Technical direction and review:**  
Roberto Luiz Souza Monteiro

**AI-assisted writing and code generation:**  
ChatGPT (GPT-5.6 Sol), OpenAI

**Project:**  
Maia-Lite / Maia Platform

**Date:**  
September 2026

Suggested citation:

> OpenAI. (2026). *Transformer-Based Large Language Models: Algorithms, Mathematics, Pretraining, QLoRA Fine-Tuning, and Two Complete Educational Implementations in Python*. Generated with ChatGPT (GPT-5.6 Sol), under the direction of Roberto Luiz Souza Monteiro. September 12, 2026.

---

# Research Status

**Maia-Lite is currently an experimental research project.**

The current development stage includes:

```text
[✓] Transformer architecture study
[✓] Technical documentation
[✓] Initial Maia-Lite-335M design

[→] Wikipedia multilingual corpus preparation
[→] Maia-Lite-335M pretraining
[→] Scientific paper extraction
[→] Scientific QA generation
[→] English dataset construction
[→] Portuguese translation
[→] Spanish translation

[ ] Dataset validation
[ ] Scientific supervised fine-tuning
[ ] Multilingual evaluation
[ ] Scientific benchmark evaluation
[ ] Hugging Face release
[ ] GGUF conversion
[ ] Ollama deployment
```

The roadmap will evolve as experimental results become available.

---

# License

Maia-Lite is developed as part of the open-source Maia Platform ecosystem.

Source code, model weights, datasets, generated datasets, and third-party corpora may have different licensing requirements.

Before redistribution, each artifact must be checked against the license and usage conditions of its original source.

Scientific papers and derived datasets require particular attention to copyright, redistribution rights, and dataset provenance.

---

# Maia Platform

Maia-Lite is part of **Maia Platform**, an open-source ecosystem connecting software engineering, computational modeling, artificial intelligence, education, and applied research.

The broader goal is to develop software and artificial-intelligence systems that can be studied, modified, deployed, and controlled by their users.

---

## MAIA

### Machine Artificial Intelligence Architecture

**Build it. Understand it. Own it.**