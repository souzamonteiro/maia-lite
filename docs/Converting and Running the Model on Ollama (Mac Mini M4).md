# 🦙 PART 3: Converting and Running the Model on Ollama (Mac Mini M4)

Ollama does not read native Hugging Face files directly. To run the model there, we need to convert it to the **GGUF** format using the tools from the `llama.cpp` project.

---

### 📦 Step 1: Download the `llama.cpp` tools

Open Terminal on your Mac Mini and clone the official repository to use its Python conversion scripts:

```bash
# Clone the official llama.cpp repository
git clone https://github.com
cd llama.cpp

# Install the Python packages required for conversion
pip install -r requirements.txt
```

---

### 🔄 Step 2: Convert the model to GGUF

With the environment ready, run the conversion script and point it to the directory where you saved the final model after fine-tuning.

```bash
python convert_hf_to_gguf.py --outtype f16 --outfile ./meu_modelo.gguf /caminho/para/modelo_335M_instruido
```
> 💡 *Note: Replace `/caminho/para/modelo_335M_instruido` with the actual directory path on your Mac. Since the 335M model is small (about 670 MB in FP16), quantizing it to Q4 is not required; it will run extremely fast on the M4 at full precision.*

---

### 📝 Step 3: Create the Configuration File (Modelfile)

Ollama needs a "recipe" to understand how the model was trained to converse.

1. Go to the directory where the `meu_modelo.gguf` file was generated.
2. Create a plain text file named **`Modelfile`** (with no extension).
3. Add the following content to it:

```dockerfile
# Point to the generated GGUF binary
FROM ./meu_modelo.gguf

# Define the text generation parameters
PARAMETER temperature 0.7
PARAMETER stop "<s>"
PARAMETER stop "</s>"

# Define the chat format (the same template used during fine-tuning)
TEMPLATE """
{{- if .System }}### Sistema:
{{ .System }}
{{ end }}
{{- if .Prompt }}### Usuário:
{{ .Prompt }}
{{ end }}### Assistente:
{{ .Response }}</s>
"""
```

---

### 🔨 Step 4: Import and Run on Ollama

With the `Modelfile` saved in the same directory as the `.gguf` file, use the terminal to build and register your custom model in Ollama:

```bash
# Create the model in the Ollama ecosystem
ollama create meu-modelo-335m -f ./Modelfile
```

Done! Your model created from scratch is now part of Ollama. To start a chat directly in your Mac Mini terminal, run:

```bash
ollama run meu-modelo-335m
```