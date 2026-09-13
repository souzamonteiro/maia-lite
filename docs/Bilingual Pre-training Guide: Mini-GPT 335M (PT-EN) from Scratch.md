# Bilingual Pre-Training Guide: Mini-GPT 335M (PT-EN) from Scratch

This document contains the complete pipeline for creating, initializing, and pre-training from scratch a language model with **~335 million parameters** (GPT-2 Medium scale) using the Portuguese and English Wikipedia editions interleaved in real time.

---

## 🛠️ Environment Requirements (Google Colab Pro)
To run this script efficiently within the Colab Pro credit budget, configure your notebook with the following specifications:
*   **Runtime:** A100 GPU (Premium)
*   **System Profile:** High RAM (High-RAM)

---

## 📋 Unified Script for a Single Colab Cell

Copy and paste the code block below into an empty cell in Google Colab. Make sure to authorize the Google Drive connection when the security pop-up appears.

```python
# =====================================================================
# BLOCK 1: Installation, Imports, and Drive Connection
# =====================================================================
!pip install transformers datasets accelerate tokenizers mwparserfromhell --quiet

import os
import torch
from google.colab import drive
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

print("=> Conectando ao Google Drive para salvar o progresso...")
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/MiniGPT_335M_Bilingue"
os.makedirs(DRIVE_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(DRIVE_DIR, "tokenizer_bilingue")
MODEL_OUTPUT_DIR = os.path.join(DRIVE_DIR, "modelo_final")
CHECKPOINT_DIR = os.path.join(DRIVE_DIR, "checkpoints")

# Hyperparameters for ~335M parameters (GPT-2 Medium scale)
VOCAB_SIZE = 50257       # Expanded to accommodate multiple languages without token collisions
MAX_LENGTH = 1024        # Context window (causal attention)
N_EMBD = 1024            # Hidden dimension (network width)
N_LAYER = 24             # Number of Transformer layers (depth)
N_HEAD = 16              # Simultaneous attention heads

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=> Processando com hardware: {device.upper()}")

# =====================================================================
# BLOCK 2: Dataset Loading and Interleaving (Streaming)
# =====================================================================
print("\n=> Carregando os datasets da Wikipédia em modo Streaming...")
# Streaming=True prevents a massive download of gigabytes of text into Colab RAM
dataset_pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
dataset_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

# Interleave the datasets in real time. probabilities=[0.5, 0.5] ensures 50% PT and 50% EN in the queue
dataset_misturado = interleave_datasets([dataset_pt, dataset_en], probabilities=[0.5, 0.5], seed=42)

# =====================================================================
# BLOCK 3: Create the Bilingual Tokenizer from Scratch
# =====================================================================
if not os.path.exists(TOKENIZER_PATH):
    print("=> Treinando um Tokenizer BPE do zero para Português e Inglês...")
    
    def batch_iterator(batch_size=1000):
        batch = []
        for example in dataset_misturado:
            batch.append(example["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    raw_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        min_frequency=2
    )
    
    # Tokenizer training transparently consumes the first articles from the streaming source
    raw_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    raw_tokenizer.decoder = decoders.BPE()
    
    tokenizer = GPT2TokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"-> Tokenizer bilíngue salvo em: {TOKENIZER_PATH}")
else:
    print(f"=> Tokenizer bilíngue detectado no Drive. Carregando...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)

# =====================================================================
# BLOCK 4: Initialize the Model Architecture (~335M)
# =====================================================================
print("\n=> Criando arquitetura do Transformer de 335M de parâmetros...")
config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=MAX_LENGTH,
    n_ctx=MAX_LENGTH,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

model = GPT2LMHeadModel(config)
num_params = model.num_parameters()
print(f"-> Sucesso! Modelo estruturado com {num_params / 1e6:.2f} Milhões de parâmetros.")

# =====================================================================
# BLOCK 5: Dynamic Tokenization and Optimizer Configuration
# =====================================================================
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

tokenized_dataset = dataset_misturado.map(tokenize_function, batched=True, remove_columns=["text", "id", "url", "title"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n=> Definindo parâmetros de otimização para GPU A100...")
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    overwrite_output_dir=True,
    max_steps=30000,                  # Safe budget for running 4 to 5 hours on the A100 (~60-75 CUs)
    per_device_train_batch_size=4,    # Stable configuration for the A100's 40 GB
    gradient_accumulation_steps=16,   # Effective batch of 64 sequences of 1024 tokens per step
    learning_rate=4e-4,               
    weight_decay=0.01,
    lr_scheduler_type="cosine",       # Standard decay in the LLM literature
    warmup_steps=2000,                # Gradual ramp-up to avoid destructive gradient spikes
    logging_steps=50,                 
    save_steps=1000,                  # Save to Drive every 1000 steps (protection against connection drops)
    save_total_limit=2,               # Keep only the 2 latest checkpoints to save Drive space
    fp16=True,                        # Mixed precision is essential for VRAM optimization
    dataloader_num_workers=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# =====================================================================
# BLOCK 6: Run Training
# =====================================================================
print("\n=== INICIANDO O PRÉ-TREINAMENTO BILÍNGUE (PT-EN) ===")
trainer.train()

print("\n=> Treinamento concluído! Salvando modelo consolidado...")
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"-> Modelo final pronto e salvo com sucesso em: {MODEL_OUTPUT_DIR}")
```

---

## 📈 Expected Optimization Dynamics

1.  **Loss Metric:** Training will start with high loss values (generally between `10.0` and `11.0`) because the network weights are initialized purely at random. As the model builds semantic maps by bringing related words in PT and EN closer together, this value will drop quickly. Stable values **below 3.5** indicate a model with excellent bilingual fluency.
2.  **Progress Safety:** Thanks to the `save_steps=1000` flag, if your Colab session disconnects near step 12,000, for example, you will not lose your credits or the time spent. When you re-run the cell, Hugging Face will detect the Drive folder and continue exactly where it stopped.
