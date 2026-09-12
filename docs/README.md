# Transformer-Based Large Language Models — Technical Guide

This directory contains the technical guide:

**Transformer-Based Large Language Models: Algorithms, Mathematics, Pretraining, QLoRA Fine-Tuning, and Two Complete Educational Implementations in Python**

The document was created as supporting technical material for the **Maia-Lite** project and the broader **Maia Platform** ecosystem.

## Purpose

The goal of this guide is to provide a detailed but accessible explanation of how Transformer-based Large Language Models work, connecting four complementary perspectives:

- mathematical foundations;
- Transformer architecture and algorithms;
- training and fine-tuning procedures;
- executable Python implementations.

The intended reader is expected to understand Python programming, basic probability, algebra, matrices, and elementary calculus. No advanced background in mathematics or physics is assumed.

Rather than treating an LLM as a black box, the guide progressively decomposes the model into the operations from which it is constructed.

## Topics Covered

The guide includes detailed discussions of:

- autoregressive language modeling;
- tokenization and embeddings;
- positional representations;
- scaled dot-product attention;
- causal self-attention;
- multi-head attention;
- residual connections;
- LayerNorm and RMSNorm;
- feed-forward networks and SwiGLU;
- Transformer decoder blocks;
- cross-entropy and perplexity;
- backpropagation and the chain rule;
- AdamW optimization;
- learning-rate schedules;
- gradient clipping;
- pretraining;
- mixed-precision training;
- distributed training concepts;
- inference and autoregressive decoding;
- temperature, top-k, and top-p sampling;
- KV caching;
- LoRA;
- QLoRA;
- NF4 quantization;
- double quantization;
- parameter-efficient fine-tuning;
- evaluation strategies;
- grouped-query attention;
- Mixture of Experts;
- FlashAttention;
- long-context considerations.

## Python Implementations

An important objective of the document is to connect the mathematical description of a Transformer to actual executable code.

For this reason, the guide contains implementations at several abstraction levels.

### Transformer Components from First Principles

Small Python/NumPy examples demonstrate operations such as:

- stable softmax;
- scaled dot-product attention;
- causal masking;
- RMSNorm;
- feed-forward networks;
- Transformer blocks.

These examples are intended to make the correspondence between equations and code explicit.

### Standard-Library-Only Transformer

The guide also contains a complete educational decoder-only Transformer implemented using only the Python standard library.

It includes:

- character tokenization;
- embeddings;
- positional embeddings;
- causal self-attention;
- residual connections;
- normalization;
- MLP layers;
- cross-entropy;
- a small reverse-mode automatic differentiation engine;
- Adam optimization;
- gradient clipping;
- pretraining;
- autoregressive text generation.

No PyTorch, TensorFlow, JAX, NumPy, or specialized machine-learning library is required for this implementation.

The objective is educational rather than computational efficiency: it exposes what happens underneath modern deep-learning frameworks.

### PyTorch Implementation

A second complete implementation uses PyTorch to build a small GPT-style Transformer.

It includes:

- batching;
- multiple attention heads;
- multiple Transformer blocks;
- automatic differentiation;
- AdamW;
- validation;
- checkpointing;
- optimized scaled-dot-product attention;
- autoregressive generation.

### Hugging Face and QLoRA

The guide then moves to practical modern tooling using the Hugging Face ecosystem.

Examples include:

- training a Transformer from random initialization;
- training a BPE tokenizer;
- loading pretrained causal language models;
- LoRA adapters;
- 4-bit NF4 quantization;
- QLoRA;
- PEFT;
- supervised fine-tuning;
- domain-specific adaptation.

A practical QLoRA workflow based on **Qwen2.5-3B** is included as an example relevant to the Maia-Lite research and development process.

## Relationship to Maia-Lite

Maia-Lite investigates the construction, training, specialization, and deployment of compact language models within the Maia Platform ecosystem.

This document provides part of the theoretical and implementation background for that work.

In particular, it helps establish a progression from:

```text
Tokens
   ↓
Embeddings
   ↓
Transformer
   ↓
Next-token prediction
   ↓
Pretraining
   ↓
Instruction / domain fine-tuning
   ↓
LoRA / QLoRA
   ↓
Specialized Maia models
```

One of the principles behind Maia-Lite is that modern language models should not be treated exclusively as remote black-box services.

Their architecture, mathematics, training process, inference mechanisms, and limitations should be understandable and reproducible.

## AI-Assisted Documentation

This technical guide was produced through human-directed, AI-assisted technical writing.

**Technical direction and review:**  
Roberto Luiz Souza Monteiro

**AI-assisted writing and code generation:**  
ChatGPT (GPT-5.6 Sol), OpenAI

**Project:**  
Maia-Lite / Maia Platform

**Date:**  
September 2026

The scope, intended audience, structure, technical objectives, and relationship with the Maia-Lite project were defined under the direction of Roberto Luiz Souza Monteiro. The document and code examples were generated with ChatGPT and are intended to be reviewed, tested, and maintained as part of the project.

## Suggested Citation

When referring specifically to this technical guide, the following citation may be used:

> OpenAI. (2026). *Transformer-Based Large Language Models: Algorithms, Mathematics, Pretraining, QLoRA Fine-Tuning, and Two Complete Educational Implementations in Python*. Generated with ChatGPT (GPT-5.6 Sol), under the direction of Roberto Luiz Souza Monteiro. September 12, 2026.

For software, experiments, datasets, models, or research results derived from Maia-Lite, please cite the corresponding Maia-Lite project artifacts rather than this guide alone.

## Disclaimer

The implementations included in the guide are primarily educational.

The standard-library implementation, in particular, is intentionally small and computationally inefficient so that the internal operations of a Transformer can be inspected directly.

Production-scale LLM training requires additional considerations including distributed computing, optimized kernels, large-scale dataset engineering, checkpoint management, hardware-specific optimization, security, evaluation, and responsible deployment practices.

## Maia Platform

Maia-Lite is part of the broader **Maia Platform**, an open-source ecosystem focused on software engineering, computational modeling, artificial intelligence, education, and applied research.

The project emphasizes understandable software, local ownership of data and infrastructure, reproducibility, and the development of reusable foundations for research and practical applications.

---

**Maia-Lite**  
*Understanding language models by building them.*