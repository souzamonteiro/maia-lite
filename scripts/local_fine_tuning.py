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