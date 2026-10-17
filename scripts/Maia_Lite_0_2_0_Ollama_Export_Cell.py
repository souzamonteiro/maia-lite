# Maia Lite 0.2.0 -> GGUF -> Ollama release preparation (Google Colab, single cell)
# This does not retrain, overwrite model checkpoints, or assume converter support.
import json, os, subprocess, sys, shutil, hashlib
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

ROOT = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_Stage2')
HF = ROOT / 'releases/maia-lite-0.2.0-hf'
DEST = ROOT / 'releases/maia-lite-0.2.0-ollama'
LLAMA = Path('/content/llama.cpp')
MODEL_NAME = 'maia-lite:0.2.0'

if not HF.is_dir():
    raise FileNotFoundError(f'Merged HF release not found: {HF}. Run the existing 0.2.0 validation/export notebook first.')
for filename in ('config.json','tokenizer.json'):
    if not (HF / filename).is_file():
        raise FileNotFoundError(f'Missing {filename} in {HF}')
if not any(HF.glob('*.safetensors')) and not (HF / 'model.safetensors.index.json').exists():
    raise FileNotFoundError('No safetensors weights found in HF release')
config = json.loads((HF/'config.json').read_text())
print('Architecture:', config.get('model_type'), 'vocab:', config.get('vocab_size'))
if config.get('model_type') != 'gpt2':
    raise RuntimeError('Unexpected architecture; review export before conversion')

if not (LLAMA/'convert_hf_to_gguf.py').exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/ggml-org/llama.cpp',str(LLAMA)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(LLAMA/'requirements/requirements-convert_hf_to_gguf.txt')],check=True)

# Compatibility preflight: not every llama.cpp version supports GPT-2 conversion.
converter = (LLAMA/'convert_hf_to_gguf.py').read_text(encoding='utf-8')
if 'GPT2Model' not in converter and 'GPT2LMHeadModel' not in converter:
    print('\nSTOP: This llama.cpp checkout does not advertise a GPT-2 converter class.')
    print('No GGUF was created. Do not relabel safetensors as GGUF.')
    print('Recommended fallback: run the HF release using a Transformers inference server,')
    print('or implement/validate GPT-2 conversion and runtime support separately.')
else:
    DEST.mkdir(parents=True,exist_ok=True)
    gguf = DEST/'maia-lite-0.2.0-f16.gguf'
    if gguf.exists():
        raise FileExistsError(f'Output already exists: {gguf}; not overwriting')
    subprocess.run([sys.executable,str(LLAMA/'convert_hf_to_gguf.py'),str(HF),
                    '--outfile',str(gguf),'--outtype','f16'],check=True)
    if not gguf.is_file() or gguf.stat().st_size < 100_000_000:
        raise RuntimeError('GGUF absent or unexpectedly small')
    modelfile = DEST/'Modelfile'
    modelfile.write_text('''FROM ./maia-lite-0.2.0-f16.gguf
TEMPLATE """System: You are Maia Lite, a helpful assistant.
User: {{ .Prompt }}
Assistant:"""
PARAMETER temperature 0
PARAMETER num_ctx 1024
PARAMETER stop "User:"
PARAMETER stop "System:"
''',encoding='utf-8')
    digest = hashlib.sha256()
    with gguf.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    (DEST/'SHA256SUMS.txt').write_text(f'{digest.hexdigest()}  {gguf.name}\n')
    print('GGUF ready:',gguf,'size MiB:',round(gguf.stat().st_size/2**20,1))
    print('Modelfile:',modelfile)
    print('\nCopy both files to your Ollama host, cd to their directory, then run:')
    print(f'  ollama create {MODEL_NAME} -f Modelfile')
    print(f'  ollama run {MODEL_NAME}')
    print('Check actual model behavior; conversion success alone does not guarantee compatibility.')
