import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Path where the fine-tuned model was saved on your Mac Mini
CAMINHO_MODELO_FINAL = "./modelo_335M_instruido"

# Configure the device to use the Mac M4 GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

# 1. Load the trained model and tokenizer
print("Carregando o modelo na memória...")
tokenizer = GPT2TokenizerFast.from_pretrained(CAMINHO_MODELO_FINAL)
model = GPT2LMHeadModel.from_pretrained(CAMINHO_MODELO_FINAL).to(device)
model.eval() # Put the model in inference (evaluation) mode

# 2. Generate responses using the training instruction structure
def responder_usuario(prompt_usuario, max_tokens=150):
    # Format the input exactly as the model learned during fine-tuning
    prompt_formatado = f"### Usuário:\n{prompt_usuario}\n### Assistente:\n"
    
    # Tokenize the text and send it to the Mac GPU
    inputs = tokenizer(prompt_formatado, return_tensors="pt").to(device)
    
    # Generate text using sampling for more creative responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated tokens back into plain text
    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt so only the assistant's output is displayed
    resposta_assistente = texto_gerado.replace(prompt_formatado, "").strip()
    return resposta_assistente

# 3. Practical model tests
print("\n--- Testando o Modelo ---")

# Test 1: Portuguese conversation
print("\nPergunta: Olá, quem é você?")
print("Resposta:", responder_usuario("Olá, quem é você?"))

# Test 2: Python programming
print("\nPergunta: Escreva uma função em Python para somar dois números.")
print("Resposta:", responder_usuario("Escreva uma função em Python para somar dois números."))