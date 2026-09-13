# Pre-Training and Fine-Tuning of a 335M LLM

This document contains the complete pipeline for creating a **335-million-parameter** model from scratch, performing multilingual pre-training in the cloud (Google Colab), and fine-tuning locally on a **Mac Mini M4 (16 GB)**.

---

## ☁️ PART 1: Google Colab Script (Pre-Training from Scratch on a T4)

*Configure the notebook environment for a **T4 GPU** before running it.*

### Cell 1: Install dependencies and connect Google Drive
```python
# Install the latest Hugging Face libraries
!pip install -q transformers datasets accelerate tokenizers

from google.colab import drive
import os

# Mount Drive to save files securely
drive.mount('/content/drive')

# Create the project directory in Drive if it does not exist
PASTA_PROJETO = "/content/drive/MyDrive/meu_projeto_llm"
os.makedirs(PASTA_PROJETO, exist_ok=True)
print(f"Diretório de trabalho pronto em: {PASTA_PROJETO}")
```

### Cell 2: Create the Trilingual Tokenizer (Run only once)
```python
from datasets import load_dataset, interleave_datasets
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
import os

PASTA_PROJETO = "/content/drive/MyDrive/meu_projeto_llm"

# 1. Load a fraction via streaming to train the vocabulary
wiki_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
wiki_pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
wiki_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
dataset_misto = interleave_datasets([wiki_en, wiki_pt, wiki_es], probabilities=[0.5, 0.25, 0.25], seed=42)

# Generator used to feed the tokenizer trainer
def extrair_texto():
    for item in dataset_misto.take(50000): # 50k artigos mapeiam o vocabulário básico
        yield item["text"]

print("Treinando o tokenizador... Aguarde alguns minutos.")
tokenizer_raw = ByteLevelBPETokenizer()
tokenizer_raw.train_from_iterator(
    extrair_texto(),
    vocab_size=50257, # Padrão clássico do GPT-2
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Save the tokenizer to Drive
pasta_tok = os.path.join(PASTA_PROJETO, "tokenizador")
os.makedirs(pasta_tok, exist_ok=True)
tokenizer_raw.save_model(pasta_tok)

# Convert to the format usable by the Hugging Face Trainer
tokenizer = GPT2TokenizerFast.from_pretrained(pasta_tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
tokenizer.save_pretrained(pasta_tok)
print(f"Tokenizador salvo com sucesso em: {pasta_tok}")
```

### Cell 3: Define the 335M Architecture and Start/Resume Pre-Training
```python
import os
import torch
from datasets import load_dataset, interleave_datasets
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer, DataCollatorForLanguageModeling

PASTA_PROJETO = "/content/drive/MyDrive/meu_projeto_llm"
pasta_tok = os.path.join(PASTA_PROJETO, "tokenizador")
pasta_saida = os.path.join(PASTA_PROJETO, "checkpoints_pretreino")

# 1. Reload the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(pasta_tok)

# 2. Configure the network to have exactly ~335 million parameters
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,   # Maximum supported context
    n_ctx=1024,
    n_embd=1024,        # Hidden dimension
    n_layer=24,         # 24 Transformer block layers
    n_head=16,          # 16 attention heads
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
print(f"Modelo inicializado com pesos aleatórios. Total de parâmetros: {model.num_parameters():,}")

# 3. Load the datasets in streaming mode
wiki_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
wiki_pt = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
wiki_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
dataset_misto = interleave_datasets([wiki_en, wiki_pt, wiki_es], probabilities=[0.5, 0.25, 0.25], seed=42)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# Dynamic mapping because of streaming
tokenized_dataset = dataset_misto.map(tokenize_function, batched=True, remove_columns=["id", "url", "title", "text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Training Parameters Optimized for the T4
training_args = TrainingArguments(
    output_dir=pasta_saida,
    max_steps=200000,              
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4, # Effective batch size of 16
    save_steps=2500,               # Save to Drive every ~1.5 hours of training
    save_total_limit=2,            # Keep the 2 latest checkpoints to save space
    logging_steps=500,
    fp16=True,                     # Mixed precision (essential on the T4)
    learning_rate=5e-4,            
    weight_decay=0.01,
    warmup_steps=2000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# If the checkpoint directory contains data, resume automatically from where it stopped
checar_checkpoint = True if os.path.exists(pasta_saida) and len(os.listdir(pasta_saida)) > 0 else False

print(f"Iniciando treinamento. Continuando de checkpoint anterior? {checar_checkpoint}")
trainer.train(resume_from_checkpoint=checar_checkpoint)

# Save the consolidated final model
model.save_pretrained(os.path.join(PASTA_PROJETO, "modelo_335M_final"))
print("PRÉ-TREINO CONCLUÍDO COM SUCESSO!")
```

---

## 🍏 PART 2: Mac Mini M4 Script (Local Fine-Tuning)

*Download the `modelo_335M_final` and `tokenizador` directories from Google Drive to the Mac before running.*

```python
import os
import torch
from datasets import load_dataset, interleave_datasets
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer, DataCollatorForLanguageModeling

CAMINHO_MODELO_LOCAL = "./modelo_335M_final"
CAMINHO_TOKENIZER_LOCAL = "./tokenizador"
CAMINHO_SAIDA_FINETUNING = "./modelo_335M_instruido"

print("Verificando se o chip gráfico M4 (MPS) está acessível...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Sucesso! O script usará a GPU integrada do Mac M4.")
else:
    device = torch.device("cpu")
    print("Aviso: MPS não disponível. Usando CPU (Lento).")

# 1. Load the pre-trained model and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(CAMINHO_TOKENIZER_LOCAL)
model = GPT2LMHeadModel.from_pretrained(CAMINHO_MODELO_LOCAL).to(device)

# 2. Load instruction/conversation and Python datasets from Hugging Face
dataset_conversa = load_dataset("lmsys/chatbot_arena_conversations", split="train")
dataset_python = load_dataset("flytech/python-codes-dataset", split="train")

def formatar_dados(exemplo):
    if "conversation" in exemplo:
        texto = ""
        for turno in ejemplo["conversation"]:
            role = "Usuário" if turno["role"] == "user" else "Assistente"
            texto += f"### {role}:\n{turno['text']}\n"
        return {"texto_final": texto + "</s>"}
    elif "instruction" in exemplo and "output" in exemplo:
        return {"texto_final": f"### Instrução:\n{exemplo['instruction']}\n### Resposta:\n{exemplo['output']}</s>"}
    else:
        colunas = list(exemplo.keys())
        return {"texto_final": str(exemplo[colunas]) + "</s>"}

dataset_conversa = dataset_conversa.map(formatar_dados)
dataset_python = dataset_python.map(formatar_dados)

# Combine the two datasets (50% conversation, 50% programming)
dataset_tuning_completo = interleave_datasets([dataset_conversa, dataset_python], probabilities=[0.5, 0.5], seed=42)

def tokenizar_finetuning(examples):
    return tokenizer(examples["texto_final"], truncation=True, max_length=512) # Smaller window saves RAM

tokenized_tuning = dataset_tuning_completo.map(tokenizar_finetuning, batched=True, remove_columns=dataset_tuning_completo.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. Training Settings on the Mac Mini M4
training_args = TrainingArguments(
    output_dir=CAMINHO_SAIDA_FINETUNING,
    num_train_epochs=3,            
    per_device_train_batch_size=2, # Keep low to avoid memory overload/swapping on the Mac
    gradient_accumulation_steps=8, # Keep the batch mathematically stable
    logging_steps=100,
    save_steps=1000,
    learning_rate=2e-5,            # Lower rate preserves pre-training knowledge
    weight_decay=0.01,
    use_mps_device=True,           # Force Apple Silicon (Metal) usage
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_tuning,
    data_collator=data_collator,
)

print("Iniciando o Fine-Tuning local no Mac Mini M4...")
trainer.train()

# Save the unified intelligence of the final model
model.save_pretrained(CAMINHO_SAIDA_FINETUNING)
tokenizer.save_pretrained(CAMINHO_SAIDA_FINETUNING)
print(f"Tudo pronto! Seu modelo final e inteligente foi salvo em: {CAMINHO_SAIDA_FINETUNING}")
```