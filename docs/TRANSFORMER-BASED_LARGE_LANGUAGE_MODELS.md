**TRANSFORMER-BASED\
LARGE LANGUAGE MODELS**

Algorithms, Mathematics, Pretraining, QLoRA Fine-Tuning,\
and Two Complete Educational Implementations in Python

*Audience: readers comfortable with Python, algebra, basic probability,
derivatives, and matrices*

**A practical, mathematical, implementation-oriented guide**

> This Markdown edition preserves the guide content and uses Mermaid for
> architecture and process diagrams.

# Preface

Large language models (LLMs) based on the Transformer are best
understood as differentiable programs that learn a probability
distribution over token sequences. Their apparent complexity emerges
from a relatively small collection of reusable ideas: vector
representations, matrix multiplication, normalization, attention,
multilayer perceptrons, residual connections, gradient-based
optimization, and autoregressive sampling.

This guide develops those ideas from first principles. It deliberately
assumes mathematical maturity without assuming specialist training in
physics or advanced pure mathematics. Whenever possible, the notation is
tied directly to Python data structures and tensor shapes.

The final chapters contain two educational implementations. The first
uses only the Python standard library and implements a tiny decoder-only
Transformer, including forward propagation, reverse-mode automatic
differentiation, optimization, pretraining, and generation. It is
intentionally small and slow, but exposes the mechanics. The second uses
PyTorch and Hugging Face libraries to show how the same concepts become
a practical modern workflow.

Important scope note: the from-scratch programs are pedagogical language
models, not production LLMs. The architecture is genuine, but the scale
is tiny. Modern LLM training requires accelerators, distributed systems,
optimized kernels, careful data engineering, and substantial compute.

# 1. The Language-Modeling Problem

``` mermaid
flowchart LR
    A[Input token sequence] --> B[Transformer]
    B --> C[Next-token logits]
    C --> D[Softmax]
    D --> E[Probability distribution]
    E --> F[Select or sample next token]
    F --> G[Append token to context]
    G --> B
```

## 1.1 Tokens and sequences

A language model receives a finite sequence of discrete symbols called
tokens. A token can represent a byte, character, word, subword, or
another learned unit. Let the vocabulary contain V distinct tokens. A
sequence is written x₁, x₂, ..., x_T, where each x_t is an integer in
{0, ..., V−1}.

The central modeling task is to estimate the probability of a sequence.
By the chain rule of probability:

$$P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})$$

A causal or autoregressive LLM therefore learns the next-token
distribution conditioned on all preceding tokens. During generation, it
repeatedly predicts a distribution, chooses one token, appends it to the
context, and repeats.

## 1.2 Logits, softmax, and probabilities

The neural network normally outputs a vector z ∈ ℝ\^V called logits.
Logits are unconstrained real numbers. Softmax converts them into a
probability distribution:

$$\operatorname{softmax}(z)_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

For numerical stability, implementations subtract max(z) before
exponentiation. This changes neither the resulting probabilities nor
their ratios:

$$\operatorname{softmax}(z)_i=\frac{\exp(z_i-m)}{\sum_j \exp(z_j-m)},\qquad m=\max_j z_j$$

## 1.3 Cross-entropy and negative log-likelihood

If the correct next token is y, the training loss for one prediction is
the negative logarithm of the assigned probability:

$$\mathcal{L}=-\log P(y\mid \text{context})$$

Across a batch of B sequences and T predicted positions, the usual
causal-language-model loss is the mean token negative log-likelihood:

$$\mathcal{L}_{\text{batch}}=-\frac{1}{N}\sum_{n=1}^{N}\log p_n(y_n),\qquad N=\text{number of non-masked target tokens}$$

Minimizing cross-entropy is equivalent to maximum-likelihood estimation.
It rewards the model for assigning high probability to the observed
continuation and penalizes confident incorrect predictions particularly
strongly.

## 1.4 Perplexity

$$\operatorname{Perplexity}=\exp(\mathcal{L})$$

Perplexity can be interpreted loosely as the model's effective
uncertainty over the next token. It is useful when comparing models
under the same tokenization and evaluation protocol, but values from
different tokenizers are not directly comparable.

# 2. From Discrete Tokens to Vectors

## 2.1 Token embeddings

Neural networks operate on continuous vectors. An embedding matrix E ∈
ℝ\^{V×d} stores one d-dimensional vector per vocabulary item. Looking up
token i selects row E_i. For a sequence of T tokens, embedding lookup
produces X ∈ ℝ\^{T×d}.

$$X_t=E[x_t]$$

The model dimension d (often written d_model) is the width of the
residual stream. All major Transformer sublayers communicate through
vectors of this width.

## 2.2 Positional information

Self-attention alone is permutation-equivariant: without positional
information, it does not intrinsically know whether one token appeared
before another. Position must therefore be encoded. Common approaches
include learned absolute embeddings, sinusoidal encodings,
relative-position biases, ALiBi, and rotary position embeddings (RoPE).

### Sinusoidal encoding

$$PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### Rotary position embeddings

RoPE rotates pairs of query and key coordinates by an angle that depends
on position. If a 2D pair is (a,b) and angle θ is determined by the
position and frequency, the rotation is:

$$
\begin{bmatrix}
a'\\
b'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
a\\
b
\end{bmatrix}
$$

Because the same structured rotation is applied to queries and keys,
their dot product naturally contains relative-position information. RoPE
is widely used in contemporary decoder-only LLMs.

# 3. Self-Attention

``` mermaid
flowchart LR
    X[Hidden states X] --> Q[Query projection Q = XW_Q]
    X --> K[Key projection K = XW_K]
    X --> V[Value projection V = XW_V]
    Q --> S[Scaled scores QK^T / sqrt(d_k)]
    K --> S
    S --> M[Causal mask]
    M --> P[Softmax attention weights]
    P --> O[Weighted sum]
    V --> O
    O --> Y[Attention output]
```

## 3.1 Queries, keys, and values

Given hidden states X ∈ ℝ\^{T×d}, learned projection matrices create
three representations:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$$

Intuitively, a query describes what a position is looking for; a key
describes what a position offers for matching; and a value contains
information to retrieve when the match is strong. These are useful
intuitions, not hard semantic constraints.

## 3.2 Scaled dot-product attention

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The matrix QKᵀ contains one similarity score for every query-key pair.
Dividing by √d_k keeps the variance of dot products under control as the
head dimension grows. Softmax is applied across the key dimension so
each query position receives a normalized distribution over source
positions.

For T tokens, naïve self-attention constructs a T×T score matrix. This
gives O(T²) attention-memory and score-computation scaling with context
length, one reason long-context inference and training require
specialized algorithms.

## 3.3 Causal masking

A decoder-only language model must not inspect future tokens while
predicting the next token. Before softmax, positions j \> i are assigned
−∞ (or a sufficiently negative value):

$$
\operatorname{score}(i,j)=
\begin{cases}
-\infty, & j>i\\
\operatorname{score}(i,j), & j\le i
\end{cases}
$$

After softmax, masked positions receive probability zero. The resulting
attention matrix is lower triangular.

## 3.4 Multi-head attention

Instead of one large attention operation, the representation is divided
into h heads. Each head has its own learned projections and can develop
different interaction patterns:

$$\operatorname{head}_i=\operatorname{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})$$

$$\operatorname{MHA}(X)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O$$

Typically d_head = d/h. Multi-head attention does not guarantee
human-interpretable roles, but it increases representational flexibility
by allowing multiple attention subspaces.

## 3.5 A small numerical example

Suppose one query has dot-product scores \[2.0, 1.0, 0.0\] after
scaling. Softmax gives approximately \[0.665, 0.245, 0.090\]. The output
is therefore 0.665·v₁ + 0.245·v₂ + 0.090·v₃. Attention is a
content-dependent weighted combination of value vectors.

# 4. The Transformer Block

``` mermaid
flowchart TD
    X[Residual stream X] --> N1[Normalization]
    N1 --> A[Causal multi-head attention]
    A --> R1[Residual addition]
    X --> R1
    R1 --> N2[Normalization]
    N2 --> M[Feed-forward / gated MLP]
    M --> R2[Residual addition]
    R1 --> R2
    R2 --> Y[Next block residual stream]
```

## 4.1 Residual connections

A residual connection adds a sublayer's output back to its input:

$$Y=X+F(X)$$

Residual pathways make deep optimization easier because information and
gradients can travel through an identity path. The residual stream is
the central state passed through the stack.

## 4.2 Layer normalization and RMSNorm

LayerNorm normalizes a vector using its mean and variance, then applies
learned scale and bias:

$$\mu=\frac{1}{d}\sum_i x_i$$

$$\sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2$$

LayerNorm(x)\_i = γ_i (x_i−μ)/√(σ²+ε) + β_i

Many modern LLMs use RMSNorm, which omits mean subtraction:

$$\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\varepsilon}$$

RMSNorm(x)\_i = γ_i x_i / RMS(x)

Pre-normalization architectures apply normalization before attention and
before the feed-forward network. A typical decoder block is:

$$X\leftarrow X+\operatorname{Attention}(\operatorname{Norm}(X))$$

$$X\leftarrow X+\operatorname{MLP}(\operatorname{Norm}(X))$$

## 4.3 Feed-forward network

Each token independently passes through the same position-wise
multilayer perceptron. A classical form is:

FFN(x) = W₂ φ(W₁x + b₁) + b₂

The hidden width is often several times d_model. Contemporary models
frequently use gated variants such as SwiGLU:

SwiGLU(x) = W₂ \[SiLU(W_g x) ⊙ (W_u x)\]

$$\operatorname{SiLU}(a)=a\,\sigma(a)$$

The attention sublayer mixes information across positions; the MLP
transforms information within each position. Both are essential.

# 5. A Decoder-Only Transformer as an Algorithm

``` mermaid
flowchart TD
    T[Token IDs] --> E[Token embeddings + positional information]
    E --> B1[Transformer block 1]
    B1 --> B2[Transformer block 2]
    B2 --> BN[Transformer block N]
    BN --> FN[Final normalization]
    FN --> L[Vocabulary projection]
    L --> Z[Logits for every position]
```

A GPT-like causal Transformer can be summarized as follows:

input: token IDs x\[0:T\]\
X = token_embedding(x) + positional_information

for block in blocks:\
A = norm(X)\
X = X + causal_multi_head_attention(A)\
M = norm(X)\
X = X + feed_forward(M)

X = final_norm(X)\
logits = X @ W_vocab\
return logits

Training shifts the sequence by one position: tokens x₀...x\_{T−2} are
inputs and x₁...x\_{T−1} are targets. In practice, all positions are
evaluated in parallel even though the probability factorization is
autoregressive.

# 6. Training Mathematics

``` mermaid
flowchart LR
    D[Training batch] --> F[Forward pass]
    F --> L[Cross-entropy loss]
    L --> B[Backpropagation]
    B --> G[Parameter gradients]
    G --> C[Gradient clipping]
    C --> O[AdamW update]
    O --> P[Updated parameters]
    P --> F
```

## 6.1 Parameters and gradients

Let θ denote all trainable parameters and L(θ) the scalar loss.
Gradient-based optimization computes ∇\_θ L, the vector of partial
derivatives. A basic gradient-descent update is:

θ ← θ − η ∇\_θ L

Here η is the learning rate. Backpropagation is an efficient application
of the chain rule that computes all required derivatives by traversing
the computational graph backward.

## 6.2 Chain rule

If y = f(u), u = g(x), and L = h(y), then:

$$\frac{d\mathcal{L}}{dx}=\frac{d\mathcal{L}}{dy}\frac{dy}{du}\frac{du}{dx}$$

For tensors, the same idea applies using Jacobian-vector products.
Deep-learning frameworks avoid explicitly materializing huge Jacobian
matrices; they propagate vector-Jacobian products backward.

## 6.3 AdamW

AdamW maintains exponential moving averages of the gradient and squared
gradient:

m_t = β₁m\_{t−1} + (1−β₁)g_t

v_t = β₂v\_{t−1} + (1−β₂)g_t²

m̂\_t = m_t/(1−β₁\^t), v̂\_t = v_t/(1−β₂\^t)

θ ← θ − η · m̂\_t/(√v̂\_t+ε) − ηλθ

The final term is decoupled weight decay. AdamW is a common default for
Transformer pretraining and fine-tuning.

## 6.4 Learning-rate schedules

Large Transformer training usually uses a warm-up phase followed by
decay. Warm-up reduces instability before optimizer statistics and
activations have settled. Cosine decay is common:

η(t) = η_min + 0.5(η_max−η_min)\[1 + cos(π·progress)\]

The exact schedule matters less than matching it to batch size, training
duration, optimizer, model size, and data.

## 6.5 Gradient clipping

g ← g · min(1, c / \|\|g\|\|₂)

Global-norm clipping limits unusually large updates. It is a safety
mechanism, not a substitute for a stable architecture and sensible
learning rate.

# 7. Data and Tokenization

``` mermaid
flowchart LR
    A[Raw corpus] --> B[License and provenance checks]
    B --> C[Normalization and cleaning]
    C --> D[Deduplication and filtering]
    D --> E[Tokenizer training / tokenization]
    E --> F[Sequence packing]
    F --> G[Train / validation / test splits]
```

## 7.1 Why tokenization matters

Tokenization determines the atomic symbols the model predicts.
Word-level vocabularies struggle with rare words and morphology.
Character or byte models use tiny vocabularies but require longer
sequences. Subword tokenizers provide a practical compromise.

## 7.2 Byte Pair Encoding (BPE)

A simplified BPE training procedure starts with a base vocabulary,
counts adjacent token pairs in the corpus, merges the most frequent
pair, and repeats until the target vocabulary size is reached. Modern
implementations include important details for byte handling, whitespace,
Unicode, special tokens, normalization, and deterministic segmentation.

tokens = initial_symbolization(corpus)\
while vocabulary_size \< target_size:\
pair_counts = count_adjacent_pairs(tokens)\
best_pair = argmax(pair_counts)\
new_token = merge(best_pair)\
replace_all_occurrences(tokens, best_pair, new_token)

## 7.3 Dataset construction

-   Collect data under appropriate licenses and governance rules.

-   Normalize encodings and remove corrupted records.

-   Deduplicate exact and near-duplicate content to reduce memorization
    and wasted compute.

-   Apply quality, language, safety, and domain filters appropriate to
    the model's purpose.

-   Split training, validation, and test data carefully to minimize
    leakage.

-   Tokenize, concatenate or pack examples, and create fixed or
    variable-length training sequences.

-   Track provenance, versions, filtering decisions, and reproducible
    dataset manifests.

Data quality is not a secondary concern. A well-curated smaller corpus
can outperform a larger noisy corpus for a targeted model.

# 8. Pretraining

``` mermaid
flowchart TD
    C[Tokenized corpus] --> B[Mini-batch]
    B --> M[Decoder-only Transformer]
    M --> P[Next-token predictions]
    P --> L[Causal cross-entropy]
    L --> A[Gradient accumulation]
    A --> O[Optimizer step]
    O --> K[Checkpoint]
    O --> M
```

## 8.1 Objective

Decoder-only pretraining is usually next-token prediction over a very
large corpus. The same causal loss is applied to every eligible token
position. A batch can therefore produce thousands or millions of
supervised token targets without manual labeling.

## 8.2 Mini-batches and gradient accumulation

If accelerator memory cannot hold the desired effective batch, gradients
can be accumulated over k micro-batches before an optimizer step:

$$B_{\mathrm{eff}}=B_{\mathrm{micro}}\times N_{\mathrm{accum}}\times N_{\mathrm{workers}}$$

## 8.3 Mixed precision

Training commonly uses FP16 or BF16 activations and selected
computations while retaining numerically sensitive operations in higher
precision. BF16 has a wider exponent range than FP16 and is often easier
to train with on supported hardware.

## 8.4 Distributed training

At large scale, data parallelism replicates the model and splits
batches; tensor parallelism partitions matrix operations; pipeline
parallelism partitions layers; and parameter/optimizer sharding
distributes model state. Real systems combine these techniques with
communication-aware kernels.

## 8.5 Checkpoints

A robust checkpoint stores model parameters, optimizer state, scheduler
state, training step, random-number-generator state,
tokenizer/configuration, and enough data-loader state to resume
deterministically when feasible.

# 9. Inference and Decoding

``` mermaid
flowchart LR
    P[Prompt / current context] --> M[Transformer forward pass]
    M --> K[Reuse KV cache]
    K --> L[Next-token logits]
    L --> T[Temperature]
    T --> S[Top-k / top-p filtering]
    S --> N[Sample next token]
    N --> A[Append to context]
    A --> M
```

## 9.1 Greedy decoding

x_next = argmax_i P(i \| context)

Greedy decoding is deterministic but can become repetitive or choose
locally attractive continuations that are globally poor.

## 9.2 Temperature

$$p_i=\operatorname{softmax}\left(\frac{z_i}{\tau}\right)$$

τ \< 1 sharpens the distribution; τ \> 1 flattens it. Temperature
changes randomness but does not add knowledge.

## 9.3 Top-k and top-p

Top-k sampling retains only the k highest-probability tokens. Nucleus or
top-p sampling retains the smallest set whose cumulative probability
reaches p, then renormalizes.

## 9.4 KV cache

During autoregressive generation, keys and values for previous positions
do not need to be recomputed at every step. A KV cache stores them. This
changes repeated full-prefix work into incremental decoding, at the cost
of memory proportional to context length, number of layers, and KV-head
dimensions.

## 9.5 Context windows

A model's usable context is limited by architecture, position
representation, training distribution, memory, and serving
implementation. Merely increasing a configured maximum does not
guarantee reliable long-context reasoning.

# 10. Fine-Tuning: From Base Model to Specialized Model

## 10.1 Supervised fine-tuning

Supervised fine-tuning (SFT) continues training on examples that express
desired behavior: instructions and responses, domain question-answer
pairs, structured transformations, code tasks, or other target
interactions. The objective remains token prediction, but loss is often
masked so only response tokens contribute.

## 10.2 Full fine-tuning versus parameter-efficient fine-tuning

Full fine-tuning updates every model parameter and can require
substantial accelerator memory for weights, gradients, optimizer
moments, and activations. Parameter-efficient fine-tuning (PEFT) freezes
most base weights and learns a small set of additional parameters.

# 11. LoRA

## 11.1 Low-rank adaptation

LoRA freezes a pretrained weight matrix W₀ and represents its learned
update as a low-rank product:

$$W=W_0+\Delta W,\qquad \Delta W=\frac{\alpha}{r}BA$$

If W₀ ∈ ℝ\^{d_out×d_in}, then A ∈ ℝ\^{r×d_in} and B ∈ ℝ\^{d_out×r},
where r is much smaller than d_in and d_out. Only A and B are trained.

The scaling α/r controls the magnitude of the adaptation. LoRA is
commonly attached to attention projections such as q_proj, k_proj,
v_proj, o_proj and sometimes MLP projections.

## 11.2 Parameter count

$$N_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})$$

For large square matrices with dimension d, this is approximately 2rd
rather than d², a dramatic reduction when r ≪ d.

# 12. QLoRA

``` mermaid
flowchart TD
    B[Pretrained base model] --> Q[4-bit NF4 quantization]
    Q --> F[Frozen base weights]
    F --> L[Insert trainable LoRA adapters]
    D[Supervised domain dataset] --> T[Tokenize and format]
    T --> L
    L --> R[QLoRA training]
    R --> V[Validation and task evaluation]
    V --> A[Save LoRA adapter]
    V --> M[Optional merged deployment model]
```

## 12.1 Core idea

QLoRA combines a quantized frozen base model with trainable LoRA
adapters. The base weights are stored in low precision---commonly
4-bit---while computations are dequantized to a suitable compute type
and LoRA parameters are optimized in higher precision. This greatly
reduces memory without requiring the base model itself to be updated.

## 12.2 NF4

NormalFloat4 (NF4) is a 4-bit quantization data type designed around the
approximate normal distribution often observed in pretrained
neural-network weights. A block of weights is scaled and mapped to one
of 16 representable quantization levels. The exact implementation is
handled by quantization libraries; conceptually:

$$w\approx s\,q,\qquad q\in\{\text{16 NF4 codebook values}\}$$

## 12.3 Double quantization

QLoRA can also quantize the quantization constants themselves. This
'double quantization' reduces the average memory cost of scaling
metadata.

## 12.4 Paged optimizers

Paged optimizer strategies reduce memory spikes by managing optimizer
state more flexibly. They are especially useful when long sequences
create temporary activation-memory pressure.

## 12.5 QLoRA training flow

1.  Load the pretrained causal LM with 4-bit quantization.

2.  Keep the quantized base parameters frozen.

3.  Prepare the model for low-bit training where required by the
    framework.

4.  Insert LoRA adapters into selected linear projections.

5.  Tokenize and format the supervised dataset.

6.  Train only the adapter parameters using causal cross-entropy.

7.  Validate on held-out data and task-specific evaluations.

8.  Save the adapter separately, or merge into a higher-precision base
    model when deployment requirements justify it.

## 12.6 Practical QLoRA example with Hugging Face

from transformers import (\
AutoTokenizer, AutoModelForCausalLM,\
BitsAndBytesConfig, TrainingArguments, Trainer,\
DataCollatorForLanguageModeling\
)\
from peft import LoraConfig, get_peft_model,
prepare_model_for_kbit_training\
import torch

model_name = "Qwen/Qwen2.5-3B"

bnb = BitsAndBytesConfig(\
load_in_4bit=True,\
bnb_4bit_quant_type="nf4",\
bnb_4bit_use_double_quant=True,\
bnb_4bit_compute_dtype=torch.bfloat16,\
)

tokenizer = AutoTokenizer.from_pretrained(model_name)\
model = AutoModelForCausalLM.from_pretrained(\
model_name,\
quantization_config=bnb,\
device_map="auto",\
)\
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(\
r=16,\
lora_alpha=32,\
lora_dropout=0.05,\
bias="none",\
task_type="CAUSAL_LM",\
target_modules=\["q_proj", "k_proj", "v_proj", "o_proj",\
"gate_proj", "up_proj", "down_proj"\],\
)\
model = get_peft_model(model, lora)\
model.print_trainable_parameters()

Target module names are architecture-specific and must be checked
against the chosen model. BF16 also depends on hardware support. For
some environments FP16 is the appropriate compute type.

## 12.7 Formatting an instruction dataset

def format_example(example):\
system = example.get("system", "You are a helpful assistant.")\
prompt = example\["prompt"\]\
answer = example\["answer"\]\
return (\
f"\<\|im_start\|\>system`\n{system}`{=tex}\<\|im_end\|\>`\n`{=tex}"\
f"\<\|im_start\|\>user`\n{prompt}`{=tex}\<\|im_end\|\>`\n`{=tex}"\
f"\<\|im_start\|\>assistant`\n{answer}`{=tex}\<\|im_end\|\>`\n`{=tex}"\
)

def tokenize(example):\
text = format_example(example)\
return tokenizer(\
text,\
truncation=True,\
max_length=2048,\
padding=False,\
)

For serious SFT, use the chat template defined by the model's tokenizer
rather than manually assuming special-token strings. Response-only loss
masking is also often preferable so the model is optimized primarily on
assistant output.

## 12.8 Training arguments

args = TrainingArguments(\
output_dir="qwen25-3b-domain-lora",\
per_device_train_batch_size=2,\
gradient_accumulation_steps=8,\
learning_rate=2e-4,\
num_train_epochs=2,\
warmup_ratio=0.03,\
lr_scheduler_type="cosine",\
logging_steps=10,\
save_steps=250,\
bf16=True,\
gradient_checkpointing=True,\
optim="paged_adamw_8bit",\
report_to="none",\
)

trainer = Trainer(\
model=model,\
args=args,\
train_dataset=train_dataset,\
eval_dataset=eval_dataset,\
data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),\
)\
trainer.train()\
model.save_pretrained("qwen25-3b-domain-lora")

# 13. Evaluation

Loss alone is insufficient. A useful evaluation suite combines intrinsic
metrics with task-specific and behavioral tests.

-   Validation loss and perplexity on clean held-out data.

-   Exact-match, F1, pass@k, retrieval-grounded accuracy, or domain
    metrics when appropriate.

-   Human evaluation using explicit rubrics and blinded comparisons.

-   Robustness to paraphrases, long contexts, malformed input, and
    adversarially difficult cases.

-   Calibration: whether confidence correlates with correctness.

-   Memorization and contamination checks.

-   Safety, privacy, bias, and policy evaluations appropriate to the
    deployment domain.

-   Regression tests that compare new checkpoints against a known
    baseline.

For domain adaptation, construct a test set before fine-tuning and keep
it isolated. Otherwise it is easy to optimize the training pipeline
toward the benchmark unintentionally.

# 14. Building Transformer Components from Scratch in Python

The following fragments use NumPy because it makes matrix operations
explicit while avoiding a deep-learning framework. The complete
standard-library-only model appears later.

## 14.1 Stable softmax

import numpy as np

def softmax(x, axis=-1):\
x = x - np.max(x, axis=axis, keepdims=True)\
e = np.exp(x)\
return e / np.sum(e, axis=axis, keepdims=True)

## 14.2 Scaled dot-product causal attention

def causal_attention(x, Wq, Wk, Wv):\
\# x: \[T, d\]\
q = x @ Wq\
k = x @ Wk\
v = x @ Wv\
dk = q.shape\[-1\]

scores = (q @ k.T) / np.sqrt(dk)\
T = x.shape\[0\]\
mask = np.triu(np.ones((T, T), dtype=bool), k=1)\
scores\[mask\] = -1e30

weights = softmax(scores, axis=-1)\
return weights @ v

## 14.3 RMSNorm

def rmsnorm(x, weight, eps=1e-6):\
rms = np.sqrt(np.mean(x \* x, axis=-1, keepdims=True) + eps)\
return (x / rms) \* weight

## 14.4 Feed-forward network

def silu(x):\
return x / (1.0 + np.exp(-x))

def swiglu(x, W_gate, W_up, W_down):\
gate = silu(x @ W_gate)\
up = x @ W_up\
return (gate \* up) @ W_down

## 14.5 One pre-norm block

def block(x, p):\
a = rmsnorm(x, p\["attn_norm"\])\
x = x + causal_attention(a, p\["Wq"\], p\["Wk"\], p\["Wv"\]) @ p\["Wo"\]

m = rmsnorm(x, p\["ffn_norm"\])\
x = x + swiglu(m, p\["W_gate"\], p\["W_up"\], p\["W_down"\])\
return x

These fragments implement only forward propagation. Training requires
derivatives for every operation. Frameworks such as PyTorch construct a
computational graph and perform reverse-mode automatic differentiation
automatically.

# 15. Reverse-Mode Automatic Differentiation

``` mermaid
flowchart LR
    X[Inputs and parameters] --> O1[Operation nodes]
    O1 --> O2[More operations]
    O2 --> L[Scalar loss]
    L --> R[Reverse topological traversal]
    R --> G[Accumulate local derivatives]
    G --> P[Gradients for all parameters]
```

A minimal autograd engine represents every scalar as a node containing
its value, gradient, parent nodes, and a backward function. Each
arithmetic operation creates a new node and records how its derivative
flows to its inputs.

\# Conceptual scalar autograd node\
class Value:\
def \_\_init\_\_(self, data, parents=(), backward=lambda: None):\
self.data = float(data)\
self.grad = 0.0\
self.parents = tuple(parents)\
self.\_backward = backward

def \_\_add\_\_(self, other):\
other = other if isinstance(other, Value) else Value(other)\
out = Value(self.data + other.data, (self, other))\
def backward():\
self.grad += out.grad\
other.grad += out.grad\
out.\_backward = backward\
return out

To backpropagate, topologically sort the graph, set the final scalar
loss gradient to 1, and execute each node's backward rule in reverse
topological order. The complete pure-Python implementation later in this
document extends this idea with multiplication, exponentials,
logarithms, tanh, and power operations.

# 16. Complete Educational Implementation A: Standard-Library-Only Tiny Transformer

This program intentionally uses no NumPy, PyTorch, TensorFlow, JAX,
Transformers, or other external numerical library. It uses scalar
automatic differentiation and Python lists. It trains a very small,
single-head, decoder-only Transformer on character tokens. Because every
matrix operation is expanded into Python scalar objects, it is extremely
slow. Keep dimensions and datasets tiny.

Despite the small scale, the program contains the essential causal-LM
path: character tokenizer, embeddings, positional embeddings, causal
self-attention, residual connections, layer normalization, MLP,
tied-style vocabulary projection (separate learned matrix here for
clarity), cross-entropy, Adam optimization, pretraining, and
autoregressive generation.

\# tiny_transformer_stdlib.py\
\# Educational decoder-only Transformer using only Python's standard
library.

import math\
import random

random.seed(42)

class Value:\
def \_\_init\_\_(self, data, \_prev=(), \_op=""):\
self.data = float(data)\
self.grad = 0.0\
self.\_prev = set(\_prev)\
self.\_op = \_op\
self.\_backward = lambda: None

def \_\_add\_\_(self, other):\
other = other if isinstance(other, Value) else Value(other)\
out = Value(self.data + other.data, (self, other), "+")\
def \_backward():\
self.grad += out.grad\
other.grad += out.grad\
out.\_backward = \_backward\
return out

\_\_radd\_\_ = \_\_add\_\_

def \_\_neg\_\_(self):\
return self \* -1.0

def \_\_sub\_\_(self, other):\
return self + (-other)

def \_\_rsub\_\_(self, other):\
return other + (-self)

def \_\_mul\_\_(self, other):\
other = other if isinstance(other, Value) else Value(other)\
out = Value(self.data \* other.data, (self, other), "\*")\
def \_backward():\
self.grad += other.data \* out.grad\
other.grad += self.data \* out.grad\
out.\_backward = \_backward\
return out

\_\_rmul\_\_ = \_\_mul\_\_

def \_\_pow\_\_(self, power):\
assert isinstance(power, (int, float))\
out = Value(self.data \*\* power, (self,), f"\*\*{power}")\
def \_backward():\
self.grad += power \* (self.data \*\* (power - 1)) \* out.grad\
out.\_backward = \_backward\
return out

def \_\_truediv\_\_(self, other):\
return self \* (other \*\* -1 if isinstance(other, Value) else
Value(other) \*\* -1)

def exp(self):\
v = math.exp(self.data)\
out = Value(v, (self,), "exp")\
def \_backward():\
self.grad += v \* out.grad\
out.\_backward = \_backward\
return out

def log(self):\
out = Value(math.log(self.data), (self,), "log")\
def \_backward():\
self.grad += (1.0 / self.data) \* out.grad\
out.\_backward = \_backward\
return out

def tanh(self):\
t = math.tanh(self.data)\
out = Value(t, (self,), "tanh")\
def \_backward():\
self.grad += (1.0 - t \* t) \* out.grad\
out.\_backward = \_backward\
return out

def backward(self):\
topo, visited = \[\], set()\
def build(v):\
if v not in visited:\
visited.add(v)\
for child in v.\_prev:\
build(child)\
topo.append(v)\
build(self)\
self.grad = 1.0\
for node in reversed(topo):\
node.\_backward()

def parameter(shape, scale=0.02):\
if len(shape) == 1:\
return \[Value(random.gauss(0.0, scale)) for \_ in range(shape\[0\])\]\
return \[parameter(shape\[1:\], scale) for \_ in range(shape\[0\])\]

def flatten(x):\
if isinstance(x, Value):\
return \[x\]\
out = \[\]\
for item in x:\
out.extend(flatten(item))\
return out

def dot(a, b):\
return sum((x \* y for x, y in zip(a, b)), Value(0.0))

def matvec(x, W):\
\# W: \[out_dim\]\[in_dim\]\
return \[dot(row, x) for row in W\]

def vec_add(a, b):\
return \[x + y for x, y in zip(a, b)\]

def layer_norm(x, gamma, beta, eps=1e-5):\
n = len(x)\
mean = sum(x, Value(0.0)) / n\
centered = \[v - mean for v in x\]\
var = sum((v \* v for v in centered), Value(0.0)) / n\
inv_std = (var + eps) \*\* -0.5\
return \[gamma\[i\] \* centered\[i\] \* inv_std + beta\[i\] for i in
range(n)\]

def softmax(xs):\
\# Subtracting max uses detached scalar max only for stability.\
m = max(x.data for x in xs)\
exps = \[(x - m).exp() for x in xs\]\
denom = sum(exps, Value(0.0))\
return \[e / denom for e in exps\]

def cross_entropy(logits, target):\
probs = softmax(logits)\
return -(probs\[target\] + 1e-12).log()

class TinyTransformer:\
def \_\_init\_\_(self, vocab_size, d_model=12, d_ff=24, context=24):\
self.V = vocab_size\
self.d = d_model\
self.context = context

self.tok = parameter((vocab_size, d_model))\
self.pos = parameter((context, d_model))

self.ln1_g = \[Value(1.0) for \_ in range(d_model)\]\
self.ln1_b = \[Value(0.0) for \_ in range(d_model)\]\
self.Wq = parameter((d_model, d_model))\
self.Wk = parameter((d_model, d_model))\
self.Wv = parameter((d_model, d_model))\
self.Wo = parameter((d_model, d_model))

self.ln2_g = \[Value(1.0) for \_ in range(d_model)\]\
self.ln2_b = \[Value(0.0) for \_ in range(d_model)\]\
self.W1 = parameter((d_ff, d_model))\
self.b1 = \[Value(0.0) for \_ in range(d_ff)\]\
self.W2 = parameter((d_model, d_ff))\
self.b2 = \[Value(0.0) for \_ in range(d_model)\]

self.lnf_g = \[Value(1.0) for \_ in range(d_model)\]\
self.lnf_b = \[Value(0.0) for \_ in range(d_model)\]\
self.W_vocab = parameter((vocab_size, d_model))\
self.b_vocab = \[Value(0.0) for \_ in range(vocab_size)\]

def parameters(self):\
return flatten(\[\
self.tok, self.pos,\
self.ln1_g, self.ln1_b, self.Wq, self.Wk, self.Wv, self.Wo,\
self.ln2_g, self.ln2_b, self.W1, self.b1, self.W2, self.b2,\
self.lnf_g, self.lnf_b, self.W_vocab, self.b_vocab\
\])

def forward(self, ids):\
T = len(ids)\
assert T \<= self.context

x = \[\
\[self.tok\[ids\[t\]\]\[j\] + self.pos\[t\]\[j\] for j in
range(self.d)\]\
for t in range(T)\
\]

\# Pre-norm causal self-attention, one head.\
n1 = \[layer_norm(v, self.ln1_g, self.ln1_b) for v in x\]\
Q = \[matvec(v, self.Wq) for v in n1\]\
K = \[matvec(v, self.Wk) for v in n1\]\
V = \[matvec(v, self.Wv) for v in n1\]

attn_out = \[\]\
scale = 1.0 / math.sqrt(self.d)\
for t in range(T):\
scores = \[dot(Q\[t\], K\[s\]) \* scale for s in range(t + 1)\]\
weights = softmax(scores)\
mixed = \[\]\
for j in range(self.d):\
mixed.append(sum(\
(weights\[s\] \* V\[s\]\[j\] for s in range(t + 1)),\
Value(0.0)\
))\
attn_out.append(matvec(mixed, self.Wo))

x = \[vec_add(x\[t\], attn_out\[t\]) for t in range(T)\]

\# Pre-norm MLP with tanh activation.\
n2 = \[layer_norm(v, self.ln2_g, self.ln2_b) for v in x\]\
mlp_out = \[\]\
for v in n2:\
h = vec_add(matvec(v, self.W1), self.b1)\
h = \[z.tanh() for z in h\]\
y = vec_add(matvec(h, self.W2), self.b2)\
mlp_out.append(y)\
x = \[vec_add(x\[t\], mlp_out\[t\]) for t in range(T)\]

x = \[layer_norm(v, self.lnf_g, self.lnf_b) for v in x\]\
logits = \[vec_add(matvec(v, self.W_vocab), self.b_vocab) for v in x\]\
return logits

class Adam:\
def \_\_init\_\_(self, params, lr=3e-3, beta1=0.9, beta2=0.999,
eps=1e-8):\
self.params = params\
self.lr = lr\
self.b1 = beta1\
self.b2 = beta2\
self.eps = eps\
self.m = \[0.0\] \* len(params)\
self.v = \[0.0\] \* len(params)\
self.t = 0

def zero_grad(self):\
for p in self.params:\
p.grad = 0.0

def step(self):\
self.t += 1\
for i, p in enumerate(self.params):\
g = p.grad\
self.m\[i\] = self.b1 \* self.m\[i\] + (1 - self.b1) \* g\
self.v\[i\] = self.b2 \* self.v\[i\] + (1 - self.b2) \* g \* g\
mh = self.m\[i\] / (1 - self.b1 \*\* self.t)\
vh = self.v\[i\] / (1 - self.b2 \*\* self.t)\
p.data -= self.lr \* mh / (math.sqrt(vh) + self.eps)

def sample_from_probs(probs, temperature=0.9):\
if temperature \<= 0:\
return max(range(len(probs)), key=lambda i: probs\[i\])\
logits = \[math.log(max(p, 1e-12)) / temperature for p in probs\]\
m = max(logits)\
ws = \[math.exp(z - m) for z in logits\]\
total = sum(ws)\
r = random.random() \* total\
acc = 0.0\
for i, w in enumerate(ws):\
acc += w\
if r \<= acc:\
return i\
return len(ws) - 1

def generate(model, prompt_ids, max_new_tokens=80, temperature=0.9):\
ids = list(prompt_ids)\
for \_ in range(max_new_tokens):\
ctx = ids\[-model.context:\]\
logits = model.forward(ctx)\[-1\]\
probs = softmax(logits)\
detached = \[p.data for p in probs\]\
nxt = sample_from_probs(detached, temperature)\
ids.append(nxt)\
return ids

if \_\_name\_\_ == "\_\_main\_\_":\
text = (\
"transformers learn patterns in sequences."\
"attention mixes information across positions."\
"language models predict the next token."\
) \* 6

chars = sorted(set(text))\
stoi = {ch: i for i, ch in enumerate(chars)}\
itos = {i: ch for ch, i in stoi.items()}\
data = \[stoi\[ch\] for ch in text\]

model = TinyTransformer(\
vocab_size=len(chars),\
d_model=12,\
d_ff=24,\
context=20\
)\
opt = Adam(model.parameters(), lr=2e-3)

steps = 120\
seq_len = 12

for step in range(steps):\
start = random.randint(0, len(data) - seq_len - 2)\
chunk = data\[start:start + seq_len + 1\]\
inp = chunk\[:-1\]\
targets = chunk\[1:\]

logits = model.forward(inp)\
losses = \[cross_entropy(logits\[t\], targets\[t\]) for t in
range(seq_len)\]\
loss = sum(losses, Value(0.0)) / seq_len

opt.zero_grad()\
loss.backward()

\# Simple gradient clipping.\
norm = math.sqrt(sum(p.grad \* p.grad for p in model.parameters()))\
if norm \> 1.0:\
scale = 1.0 / norm\
for p in model.parameters():\
p.grad \*= scale

opt.step()

if step % 10 == 0:\
print(f"step={step:4d} loss={loss.data:.4f}")

prompt = "trans"\
prompt_ids = \[stoi\[c\] for c in prompt\]\
out = generate(model, prompt_ids, max_new_tokens=100, temperature=0.8)\
print("".join(itos\[i\] for i in out))

## 16.1 What this implementation teaches

-   Every trainable number is ultimately just a scalar parameter with a
    derivative.

-   Matrix multiplication is repeated dot products; attention is dot
    products plus softmax plus weighted sums.

-   Causal attention can be implemented by restricting each query
    position to keys at positions ≤ t.

-   Backpropagation is bookkeeping for the chain rule over the
    computational graph.

-   The optimizer is separate from the model: it only consumes parameter
    gradients.

-   Generation simply reruns the model on the available context and
    samples the next-token distribution.

## 16.2 What it deliberately omits

It has one layer, one attention head, character tokenization, learned
absolute positions, tanh rather than a modern gated MLP, no batching, no
KV cache, no checkpoint format, no mixed precision, and no vectorized
kernels. Those omissions keep the mechanism inspectable. Scaling this
exact scalar implementation would be computationally infeasible.

# 17. Complete Educational Implementation B: PyTorch Transformer Pretraining and Generation

The next program uses PyTorch for tensors and automatic differentiation
but implements the Transformer architecture directly rather than calling
a ready-made Transformer layer. It is suitable for experimenting with a
small character-level model on a text file. It supports batching,
multiple heads, multiple blocks, AdamW, validation loss, checkpointing,
and generation.

\# tiny_transformer_torch.py\
\# pip install torch\
\#\
\# Put training text in input.txt, then:\
\# python tiny_transformer_torch.py

import math\
import torch\
import torch.nn as nn\
import torch.nn.functional as F

torch.manual_seed(42)

device = (\
"cuda" if torch.cuda.is_available()\
else "mps" if torch.backends.mps.is_available()\
else "cpu"\
)

batch_size = 32\
block_size = 128\
d_model = 192\
n_heads = 6\
n_layers = 6\
dropout = 0.1\
learning_rate = 3e-4\
max_steps = 3000\
eval_interval = 200\
eval_batches = 30

text = open("input.txt", "r", encoding="utf-8").read()\
chars = sorted(set(text))\
vocab_size = len(chars)\
stoi = {ch: i for i, ch in enumerate(chars)}\
itos = {i: ch for ch, i in stoi.items()}

def encode(s):\
return \[stoi\[c\] for c in s\]

def decode(ids):\
return "".join(itos\[i\] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)\
split = int(0.9 \* len(data))\
train_data = data\[:split\]\
val_data = data\[split:\]

def get_batch(which):\
source = train_data if which == "train" else val_data\
starts = torch.randint(0, len(source) - block_size - 1, (batch_size,))\
x = torch.stack(\[source\[i:i+block_size\] for i in starts\])\
y = torch.stack(\[source\[i+1:i+block_size+1\] for i in starts\])\
return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):\
def \_\_init\_\_(self):\
super().\_\_init\_\_()\
assert d_model % n_heads == 0\
self.n_heads = n_heads\
self.head_dim = d_model // n_heads\
self.qkv = nn.Linear(d_model, 3 \* d_model, bias=False)\
self.out = nn.Linear(d_model, d_model, bias=False)\
self.dropout = nn.Dropout(dropout)

def forward(self, x):\
B, T, C = x.shape\
qkv = self.qkv(x)\
q, k, v = qkv.chunk(3, dim=-1)

q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\
k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\
v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

\# PyTorch uses an optimized scaled-dot-product attention kernel\
\# when possible. is_causal=True applies the triangular mask.\
y = F.scaled_dot_product_attention(\
q, k, v,\
dropout_p=dropout if self.training else 0.0,\
is_causal=True,\
)

y = y.transpose(1, 2).contiguous().view(B, T, C)\
return self.out(self.dropout(y))

class MLP(nn.Module):\
def \_\_init\_\_(self):\
super().\_\_init\_\_()\
hidden = 4 \* d_model\
self.fc1 = nn.Linear(d_model, hidden)\
self.fc2 = nn.Linear(hidden, d_model)\
self.dropout = nn.Dropout(dropout)

def forward(self, x):\
return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):\
def \_\_init\_\_(self):\
super().\_\_init\_\_()\
self.ln1 = nn.LayerNorm(d_model)\
self.attn = CausalSelfAttention()\
self.ln2 = nn.LayerNorm(d_model)\
self.mlp = MLP()

def forward(self, x):\
x = x + self.attn(self.ln1(x))\
x = x + self.mlp(self.ln2(x))\
return x

class TinyGPT(nn.Module):\
def \_\_init\_\_(self):\
super().\_\_init\_\_()\
self.token_emb = nn.Embedding(vocab_size, d_model)\
self.pos_emb = nn.Embedding(block_size, d_model)\
self.blocks = nn.Sequential(\*\[Block() for \_ in range(n_layers)\])\
self.ln_f = nn.LayerNorm(d_model)\
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

\# Weight tying reduces parameters and is common in language models.\
self.lm_head.weight = self.token_emb.weight

self.apply(self.\_init_weights)

def \_init_weights(self, module):\
if isinstance(module, nn.Linear):\
nn.init.normal\_(module.weight, mean=0.0, std=0.02)\
if module.bias is not None:\
nn.init.zeros\_(module.bias)\
elif isinstance(module, nn.Embedding):\
nn.init.normal\_(module.weight, mean=0.0, std=0.02)

def forward(self, idx, targets=None):\
B, T = idx.shape\
assert T \<= block_size\
pos = torch.arange(T, device=idx.device)

x = self.token_emb(idx) + self.pos_emb(pos)\[None, :, :\]\
x = self.blocks(x)\
x = self.ln_f(x)\
logits = self.lm_head(x)

loss = None\
if targets is not None:\
loss = F.cross_entropy(\
logits.reshape(B \* T, vocab_size),\
targets.reshape(B \* T),\
)\
return logits, loss

@torch.no_grad()\
def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):\
for \_ in range(max_new_tokens):\
ctx = idx\[:, -block_size:\]\
logits, \_ = self(ctx)\
logits = logits\[:, -1, :\] / max(temperature, 1e-5)

if top_k is not None:\
k = min(top_k, logits.size(-1))\
values, \_ = torch.topk(logits, k)\
cutoff = values\[:, \[-1\]\]\
logits = torch.where(\
logits \< cutoff,\
torch.full_like(logits, float("-inf")),\
logits,\
)

probs = F.softmax(logits, dim=-1)\
nxt = torch.multinomial(probs, num_samples=1)\
idx = torch.cat(\[idx, nxt\], dim=1)\
return idx

@torch.no_grad()\
def estimate_loss(model):\
model.eval()\
result = {}\
for split_name in \["train", "val"\]:\
losses = \[\]\
for \_ in range(eval_batches):\
x, y = get_batch(split_name)\
\_, loss = model(x, y)\
losses.append(loss.item())\
result\[split_name\] = sum(losses) / len(losses)\
model.train()\
return result

model = TinyGPT().to(device)\
print(f"device={device}")\
print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(\
model.parameters(),\
lr=learning_rate,\
betas=(0.9, 0.95),\
weight_decay=0.1,\
)

for step in range(max_steps):\
if step % eval_interval == 0:\
losses = estimate_loss(model)\
print(\
f"step={step:5d} "\
f"train={losses\['train'\]:.4f} "\
f"val={losses\['val'\]:.4f}"\
)

x, y = get_batch("train")\
\_, loss = model(x, y)

optimizer.zero_grad(set_to_none=True)\
loss.backward()\
torch.nn.utils.clip_grad_norm\_(model.parameters(), 1.0)\
optimizer.step()

torch.save(\
{\
"model": model.state_dict(),\
"stoi": stoi,\
"itos": itos,\
},\
"tiny_gpt.pt",\
)

prompt = text\[:1\]\
idx = torch.tensor(\[encode(prompt)\], dtype=torch.long, device=device)\
out = model.generate(idx, max_new_tokens=500, temperature=0.8,
top_k=50)\
print(decode(out\[0\].tolist()))

# 18. Practical Pretraining with the Hugging Face Ecosystem

For a real subword model, it is usually better to use established
tokenizers, model configurations, datasets, and training infrastructure.
The following sketch creates a small GPT-2-style model from
configuration and trains it with the Transformers Trainer. It
initializes random weights; it does not download pretrained model
weights.

\# pip install torch transformers datasets tokenizers accelerate

from datasets import load_dataset\
from transformers import (\
AutoTokenizer,\
GPT2Config,\
GPT2LMHeadModel,\
DataCollatorForLanguageModeling,\
TrainingArguments,\
Trainer,\
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")\
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(\
vocab_size=len(tokenizer),\
n_positions=512,\
n_ctx=512,\
n_embd=384,\
n_layer=8,\
n_head=6,\
resid_pdrop=0.1,\
embd_pdrop=0.1,\
attn_pdrop=0.1,\
)

model = GPT2LMHeadModel(config) \# random initialization

dataset = load_dataset(\
"text",\
data_files={"train": "train.txt", "validation": "validation.txt"}\
)

def tokenize(batch):\
return tokenizer(batch\["text"\])

tokenized = dataset.map(\
tokenize,\
batched=True,\
remove_columns=\["text"\],\
)

block_size = 512

def group_texts(examples):\
concatenated = sum(examples\["input_ids"\], \[\])\
total = (len(concatenated) // block_size) \* block_size\
chunks = \[\
concatenated\[i:i+block_size\]\
for i in range(0, total, block_size)\
\]\
return {"input_ids": chunks, "labels": \[x\[:\] for x in chunks\]}

lm_data = tokenized.map(group_texts, batched=True)

args = TrainingArguments(\
output_dir="small-transformer-from-scratch",\
per_device_train_batch_size=8,\
per_device_eval_batch_size=8,\
gradient_accumulation_steps=4,\
learning_rate=3e-4,\
weight_decay=0.1,\
warmup_ratio=0.02,\
num_train_epochs=3,\
logging_steps=20,\
save_steps=500,\
eval_strategy="steps",\
eval_steps=500,\
bf16=True, \# use fp16=True instead if appropriate for your hardware\
)

trainer = Trainer(\
model=model,\
args=args,\
train_dataset=lm_data\["train"\],\
eval_dataset=lm_data\["validation"\],\
data_collator=DataCollatorForLanguageModeling(\
tokenizer=tokenizer,\
mlm=False,\
),\
)

trainer.train()\
trainer.save_model("small-transformer-from-scratch/final")\
tokenizer.save_pretrained("small-transformer-from-scratch/final")

For a serious new model, train a tokenizer on the target corpus rather
than reusing GPT-2's tokenizer. Also stream or memory-map large corpora
instead of concatenating Python lists as this compact example does.

# 19. Training a Tokenizer

from tokenizers import Tokenizer, models, trainers, pre_tokenizers,
decoders

tokenizer = Tokenizer(models.BPE(unk_token="\<unk\>"))\
tokenizer.pre_tokenizer =
pre_tokenizers.ByteLevel(add_prefix_space=False)\
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(\
vocab_size=32000,\
min_frequency=2,\
special_tokens=\["\<pad\>", "\<unk\>", "\<bos\>", "\<eos\>"\],\
)

tokenizer.train(\["corpus.txt"\], trainer)\
tokenizer.save("tokenizer.json")

A production tokenizer design should be tested on all supported
languages, source code, numbers, punctuation, Unicode edge cases, and
domain terminology. Token efficiency affects both training cost and
usable context.

# 20. Memory and Compute Intuition

## 20.1 Parameter storage

Ignoring overhead, N parameters stored at b bits each require
approximately:

$$\operatorname{memory}\approx\frac{N\,b}{8}\ \text{bytes}$$

A 3-billion-parameter model at 16 bits therefore needs roughly 6 GB for
weights alone. Training requires additional memory for gradients,
optimizer states, activations, temporary buffers, and framework
overhead.

## 20.2 Why QLoRA helps

A 4-bit representation reduces base-weight storage to roughly one
quarter of 16-bit storage before quantization metadata and runtime
overhead. Because the base model is frozen, gradients and optimizer
states are needed only for the much smaller adapter parameter set.

## 20.3 Attention cost

For sequence length T and width d, dense attention score computation
scales approximately as O(T²d) across heads, while the linear
projections and MLP scale approximately linearly in T but quadratically
in model width. Which component dominates depends on model shape and
context length.

# 21. Architectural Variants in Modern LLMs

## 21.1 Grouped-query attention

Multi-query attention shares one set of keys and values across query
heads. Grouped-query attention (GQA) uses fewer KV heads than query
heads. Both reduce KV-cache memory and decoding bandwidth while
retaining multiple query heads.

## 21.2 Mixture of Experts

A mixture-of-experts (MoE) layer contains multiple expert MLPs and a
router that sends each token to a small subset. The total parameter
count can be large while only a fraction of experts is active per token.
This increases model capacity without proportionally increasing
every-token compute, but introduces routing and distributed-systems
complexity.

## 21.3 FlashAttention

FlashAttention is an exact attention algorithm organized to reduce
expensive memory traffic by tiling the computation and avoiding
materialization of the full attention matrix in high-bandwidth memory.
It changes implementation efficiency, not the mathematical definition of
attention.

## 21.4 RoPE scaling and long context

Long-context extensions often modify positional-frequency behavior,
retrain or continue pretraining on longer sequences, and use efficient
attention kernels. Extrapolation beyond the lengths seen during training
should be evaluated rather than assumed.

# 22. Fine-Tuning Design for Domain Models

``` mermaid
flowchart TD
    S[Source domain material] --> C[Curate and clean]
    C --> Q[Create instruction / QA / code examples]
    Q --> QC[Quality and contamination controls]
    QC --> SPLIT[Train / validation / test split]
    SPLIT --> FT[QLoRA / PEFT fine-tuning]
    FT --> E[Domain and behavioral evaluation]
    E --> D[Deployment candidate]
```

A domain model should begin with a clearly defined capability target.
For example, 'answer graduate-level computational modeling questions in
English, Portuguese, and Spanish' is more operational than 'know
science.' Build data and evaluations around observable behavior.

## 22.1 Recommended data mixture

-   High-quality instruction-response examples written or reviewed by
    domain experts.

-   Synthetic questions generated from trusted source documents,
    followed by filtering and deduplication.

-   Reasoning and calculation problems with independently verifiable
    answers.

-   Code examples with executable tests where possible.

-   Negative or contrastive examples that teach refusal, uncertainty, or
    correction when appropriate.

-   A smaller amount of general instruction data to preserve broad
    conversational behavior.

## 22.2 Synthetic-data quality controls

Synthetic generation can scale dataset construction, but generated
answers can propagate the teacher model's errors. Useful controls
include source grounding, answer verification, diversity constraints,
semantic deduplication, difficulty balancing, automated consistency
checks, and a human-reviewed gold subset.

## 22.3 Train/validation/test discipline

Split by source document or source cluster, not merely by individual
question, when many questions come from the same paper. Otherwise nearly
identical knowledge can leak across splits and make evaluation look much
stronger than real generalization.

# 23. A QLoRA Recipe for a 3B Domain Model

The following values are reasonable starting points rather than
universal constants.

  ------------------------------------------------------------------------------
  Item           Starting point        Why
  -------------- --------------------- -----------------------------------------
  Base model     3B causal             Small enough for rapid iteration; strong
                 instruction-capable   enough for useful specialization
                 model                 

  Quantization   4-bit NF4 + double    Reduces frozen-base memory
                 quantization          

  LoRA rank      r = 16 or 32          Good initial capacity/cost compromise

  LoRA alpha     2r                    Common initial scaling; tune if necessary

  LoRA dropout   0.0--0.05             Small regularization; data size dependent

  Sequence       1024--4096            Choose from actual task distribution and
  length                               hardware

  Learning rate  1e−4 to 2e−4          Typical PEFT starting range

  Epochs         1--3                  Prefer validation-driven stopping over
                                       many passes

  Effective      32--128 sequences     Use accumulation if memory is limited
  batch                                

  Warm-up        \~3%                  Stabilizes early optimization
  ------------------------------------------------------------------------------

Always inspect actual loss curves and task metrics. If validation
performance degrades while training loss continues to improve, the
adapter is overfitting. More epochs are not automatically better.

# 24. Common Failure Modes

-   Training loss decreases but generation is poor: formatting,
    tokenization, loss masking, or evaluation may be wrong.

-   Model repeats itself: decoding settings, narrow training data,
    overfitting, or insufficient diversity may contribute.

-   Catastrophic forgetting: too much narrow-domain training or an
    overly aggressive learning rate can damage general capabilities.

-   No improvement from LoRA: wrong target modules, too little data,
    poor data quality, insufficient rank, or a task the base model
    cannot represent well.

-   Out-of-memory errors: reduce sequence length or micro-batch size,
    enable gradient checkpointing, use QLoRA, or choose a smaller base
    model.

-   Training becomes unstable: inspect learning rate, precision mode,
    gradient norms, corrupted examples, and tokenizer/model
    compatibility.

-   Benchmark looks excellent but deployment disappoints: contamination,
    leakage, narrow test distribution, or metric mismatch.

# 25. Conceptual Interpretation: What the Model Is Actually Computing

At the implementation level, a Transformer LLM is a parameterized
function. Given token IDs and positions, it repeatedly transforms
vectors using affine maps, normalization, nonlinearities, and
content-dependent weighted combinations. Training adjusts parameters so
the final logits approximate conditional token probabilities observed in
data.

That mechanistic description does not by itself settle philosophical
questions about understanding, representation, reasoning, or
consciousness. It does, however, give a precise operational account of
the computation. Internal vectors can encode features and relations
useful for prediction even though no individual coordinate needs to
correspond to a human-defined concept.

A useful distinction is between mechanism and capability. The mechanism
can be expressed as matrix operations and nonlinear functions; the
learned capability is a property of the high-dimensional function
produced by optimization over data. Describing a neural system as
'matrix multiplication' is mathematically correct but incomplete in the
same way that describing a conventional program as 'transistor
switching' omits the organization that implements algorithms.

# 26. Glossary

**Autoregressive model:** A model that factorizes a sequence probability
into next-token conditional probabilities.

**Attention:** A content-dependent weighted aggregation of value vectors
using query-key similarity.

**Backpropagation:** Reverse-mode differentiation of a computational
graph.

**Causal mask:** A mask preventing a token position from attending to
future positions.

**Embedding:** A learned continuous vector associated with a discrete
token.

**Fine-tuning:** Continued training of a pretrained model on a narrower
objective or dataset.

**Gradient:** A collection of partial derivatives of a scalar objective
with respect to parameters.

**KV cache:** Stored attention keys and values from previous decoding
positions.

**Logit:** An unnormalized score converted to probability by softmax.

**LoRA:** Low-rank trainable updates attached to frozen weight matrices.

**Perplexity:** Exponentiated average negative log-likelihood.

**Pretraining:** Large-scale initial training, usually on broad
unlabeled text/code corpora.

**QLoRA:** LoRA training on top of a quantized frozen base model.

**Residual stream:** The sequence of hidden vectors carried through
residual connections.

**RoPE:** Rotary positional embedding applied to query/key coordinates.

**SFT:** Supervised fine-tuning on examples of desired input-output
behavior.

**Softmax:** Function mapping logits to a normalized categorical
distribution.

**Token:** A discrete unit predicted by the language model.

**Transformer:** Neural architecture built around attention, MLPs,
normalization, and residual connections.

# 27. Suggested Experiments

9.  Train the PyTorch character model on a small public-domain book and
    plot training versus validation loss.

10. Replace learned positional embeddings with sinusoidal embeddings and
    compare convergence.

11. Implement multi-query or grouped-query attention and measure
    KV-cache size during generation.

12. Replace GELU with SwiGLU and compare parameter count and validation
    loss under a fixed compute budget.

13. Train two tokenizers with different vocabulary sizes and measure
    tokens per character/word on your target languages.

14. Fine-tune a 3B model with QLoRA using ranks 8, 16, and 32; compare
    trainable parameters, memory, and held-out accuracy.

15. Create a source-separated synthetic QA dataset and compare
    random-example splitting with document-level splitting to observe
    leakage effects.

16. Implement top-p sampling and compare it qualitatively with greedy
    and top-k decoding.

# 28. Further Reading

The following works are foundational or especially useful for
understanding the concepts developed here:

-   Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.

-   Radford, A. et al. (2018/2019). Generative Pre-Training and Language
    Models are Unsupervised Multitask Learners.

-   Brown, T. et al. (2020). Language Models are Few-Shot Learners.
    NeurIPS.

-   Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language
    Models.

-   Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized
    LLMs.

-   Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary
    Position Embedding.

-   Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All
    You Need.

-   Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints.

-   Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient
    Exact Attention with IO-Awareness.

-   Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer
    Normalization.

# 29. Closing Summary

A Transformer language model begins with a simple probabilistic
objective: predict the next token. Tokens become vectors; attention
mixes information across positions; MLPs transform each position;
residual connections preserve and refine a shared representation; and
gradient-based optimization adjusts millions or billions of parameters
to reduce cross-entropy over enormous datasets.

Pretraining creates broad statistical and representational competence.
Fine-tuning shapes that competence toward particular tasks. LoRA
expresses adaptation as low-rank weight updates, while QLoRA keeps the
base model quantized and frozen so useful specialization can be
performed with far less memory.

The most productive way to understand LLMs is to move repeatedly between
three levels: equations, tensor shapes, and executable code. The
equations explain the operations, tensor shapes explain how information
flows, and code exposes the engineering decisions that make the
mathematics trainable at scale.
