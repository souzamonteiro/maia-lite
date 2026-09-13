# Instruction Fine-Tuning Guide (Chat): Fitness Assistant

This pipeline teaches your **~335M parameter** base model (pre-trained in the previous script) to act as a **Virtual Fitness Assistant**. It transforms the model from a simple "encyclopedia completer" into a responsive chatbot that understands question-and-answer dynamics.

---

## 📊 The Chat Template Format (How the AI learns to converse)
To help the model understand who the user and assistant are, we structure the text using special control markers:
*   `<|user|>`: Indicates that the following text is the user's question or command.
*   `<|assistant|>`: Indicates the response that the model should learn to generate.
*   `</s>`: The end-of-sequence token, teaching the model to stop speaking when it finishes the response.

---

## 📋 Fine-Tuning Script for Colab Pro
You can run this training on Colab's **T4** or **L4 GPU** to save your premium credits, since instruction data is considerably lighter than pre-training.

```python
# =====================================================================
# BLOCK 1: Installations and Loading the Base Model from Drive
# =====================================================================
!pip install transformers datasets accelerate --quiet

import os
import torch
from google.colab import drive
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments, Trainer, DataCollatorForLanguageModeling

print("=> Conectando ao Google Drive...")
drive.mount('/content/drive')

# Directory paths (must match the pre-training paths)
PRETRAINED_MODEL_DIR = "/content/drive/MyDrive/MiniGPT_335M_Bilingue/modelo_final"
FINETUNED_OUTPUT_DIR = "/content/drive/MyDrive/MiniGPT_335M_Bilingue/modelo_assistente_fitness"

print("=> Carregando Modelo Base e Tokenizer do Drive...")
tokenizer = GPT2TokenizerFast.from_pretrained(PRETRAINED_MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_DIR)

# Add the special chat tokens to the model vocabulary
tokens_especiais = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(tokens_especiais)
model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =====================================================================
# BLOCK 2: Create the Instruction Dataset (Simulated Examples)
# =====================================================================
print("\n=> Criando dados de treino para o Assistente de Fitness...")

# Example of data structured as a Chat Template (Bilingual)
dados_exercicios = [
    {
        "prompt": "<|user|> Como faço para ganhar massa muscular nas pernas?\n<|assistant|> Para hipertrofia de membros inferiores, foque em exercícios compostos pesados como o Agachamento Livre, Leg Press 45º e Cadeira Extensora. Mantenha uma faixa de 8 a 12 repetições com carga progressiva e garanta uma ingestão proteica adequada.</s>"
    },
    {
        "prompt": "<|user|> What is the best strategy to lose fat quickly?\n<|assistant|> The most effective approach combines a sustainable caloric deficit (eating fewer calories than you burn), regular resistance training to preserve lean muscle, and increased daily activity (NEAT) alongside structured cardio sessions.</s>"
    },
    {
        "prompt": "<|user|> Quantas vezes por semana devo treinar um músculo?\n<|assistant|> Para a maioria dos praticantes, treinar cada grupo muscular de 2 a 3 vezes por semana apresenta os melhores resultados em termos de síntese proteica e volume semanal de treino ideal.</s>"
    },
    {
        "prompt": "<|user|> Is it necessary to take whey protein right after working out?\n<|assistant|> No, the total daily protein intake is far more critical than immediate post-workout timing. However, consuming protein within a 2-3 hour window after training is beneficial for optimal muscle recovery.</s>"
    },
    {
        "prompt": "<|user|> Estou com dores muito fortes na lombar após o treino. O que eu faço?\n<|assistant|> Reduza imediatamente as cargas ou interrompa exercícios que sobrecarreguem a região, como o Levantamento Terra. Verifique sua execução no espelho e, caso a dor persista por mais de 48 horas, consulte um médico ou fisioterapeuta.</s>"
    }
]

# Convert the list into a Hugging Face Dataset object
dataset_chat = Dataset.from_list(dados_exercicios)

# =====================================================================
# BLOCK 3: Prompt Processing and Tokenization
# =====================================================================
def tokenize_chat(examples):
    return tokenizer(examples["prompt"], truncation=True, max_length=512)

tokenized_chat_dataset = dataset_chat.map(tokenize_chat, remove_columns=["prompt"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =====================================================================
# BLOCK 4: Supervised Fine-Tuning (SFT) Parameters
# =====================================================================
print("\n=> Configurando o ambiente de Fine-Tuning...")
args_tuning = TrainingArguments(
    output_dir=FINETUNED_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=15,               # Because the dataset is small, run multiple epochs to reinforce the behavior
    per_device_train_batch_size=2,
    learning_rate=5e-5,                # Much lower learning rate to avoid destroying prior knowledge
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="no",                # Small dataset; save only at the end of the process
    fp16=True if torch.cuda.is_available() else False
)

trainer = Trainer(
    model=model,
    args=args_tuning,
    data_collator=data_collator,
    train_dataset=tokenized_chat_dataset
)

print("\n=== EXECUTANDO O FINE-TUNING DE INSTRUÇÃO ===")
trainer.train()

# Final save of the Conversational Assistant
print("\n=> Salvando o Assistente de Fitness consolidado...")
model.save_pretrained(FINETUNED_OUTPUT_DIR)
tokenizer.save_pretrained(FINETUNED_OUTPUT_DIR)
print(f"-> Sucesso! Modelo alinhado e pronto para chat salvo em: {FINETUNED_OUTPUT_DIR}")
```

---

## 💬 How to Test and Chat with Your Assistant

After fine-tuning finishes, use the inference cell below to open a chat with the model directly in Colab:

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load the finalized chat model
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/MiniGPT_335M_Bilingue/modelo_assistente_fitness").to("cuda")
tokenizer = GPT2TokenizerFast.from_pretrained("/content/drive/MyDrive/MiniGPT_335M_Bilingue/modelo_assistente_fitness")

# Format the input using the pattern learned by the model
pergunta_usuario = "Qual o melhor exercício para pernas?"
prompt_formatado = f"<|user|> {pergunta_usuario}\n<|assistant|>"

inputs = tokenizer(prompt_formatado, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.encode("</s>")[0], # Force the AI to stop when generating the end token
    pad_token_id=tokenizer.pad_token_id
)

# Remove the instruction tokens from the display to focus only on the clean response
resposta_crua = tokenizer.decode(outputs[0], skip_special_tokens=False)
resposta_limpa = resposta_crua.split("<|assistant|>")[-1].replace("</s>", "").strip()

print(f"🏋️ Assistente Fitness: {resposta_limpa}")
```
