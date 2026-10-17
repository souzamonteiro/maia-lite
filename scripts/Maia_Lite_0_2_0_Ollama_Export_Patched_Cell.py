# Maia Lite 0.2.0 -> GGUF / Ollama (single Google Colab cell)
# Reuses a previously patched llama.cpp converter when available.
# Does not claim to recreate the 0.1.0 patch; avoids git pull and overwrites.
import os, sys, json, hashlib, subprocess
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_Stage2')
HF = ROOT / 'releases/maia-lite-0.2.0-hf'
OUT = ROOT / 'releases/maia-lite-0.2.0-ollama'
GGUF = OUT / 'maia-lite-0.2.0-f16.gguf'
MODEL = 'maia-lite:0.2.0'

assert HF.is_dir(), f'HF release not found: {HF}'
config = json.loads((HF / 'config.json').read_text())
assert config.get('model_type') == 'gpt2', config.get('model_type')
assert (HF / 'tokenizer.json').exists(), 'Missing tokenizer.json'
assert list(HF.glob('*.safetensors')), 'Missing safetensors weights'
print('HF model:', HF)
print('Architecture:', config['model_type'], 'vocab:', config.get('vocab_size'))

# Prefer the already patched 0.1.0 checkout. Never update it automatically.
# If necessary, edit LLAMA_CPP_DIR to the directory containing that checkout.
LLAMA_CPP_DIR = os.environ.get('MAIA_PATCHED_LLAMA_CPP', '').strip()
candidates = ([Path(LLAMA_CPP_DIR)] if LLAMA_CPP_DIR else []) + [
    Path('/content/llama.cpp-maia'),
    Path('/content/llama.cpp'),
    ROOT / 'llama.cpp',
    ROOT.parent / 'llama.cpp',
    Path('/content/drive/MyDrive/Colab_LLMs/llama.cpp'),
]
llama = next((p for p in candidates if (p / 'convert_hf_to_gguf.py').is_file()), None)
if llama is None:
    print('No llama.cpp converter found in usual locations.')
    print('Searching Maia Lite folders for the existing patched converter...')
    matches = list(ROOT.rglob('convert_hf_to_gguf.py'))[:15]
    for m in matches: print('  ', m)
    if matches: llama = matches[0].parent
if llama is None:
    raise RuntimeError('Patched llama.cpp checkout not found. Restore the converter used for Maia Lite 0.1.0 and set MAIA_PATCHED_LLAMA_CPP to its directory. No output was changed.')
print('Using converter:', llama / 'convert_hf_to_gguf.py')
print('NOTE: verify this is the patched 0.1.0 checkout; the script cannot prove that automatically.')

# Conversion is the definitive test. Do not reject based on converter class-name heuristics.
OUT.mkdir(parents=True, exist_ok=True)
if GGUF.exists():
    raise FileExistsError(f'Output already exists; refusing to overwrite: {GGUF}')
cmd = [sys.executable, str(llama / 'convert_hf_to_gguf.py'), str(HF), '--outfile', str(GGUF), '--outtype', 'f16']
print('Converting with the existing converter...')
result = subprocess.run(cmd, cwd=str(llama), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-14000:])
if result.returncode:
    if GGUF.exists():
        print('Partial output exists; inspect and remove manually before retry:', GGUF)
    raise RuntimeError('GGUF conversion failed. Check that the 0.1.0 GPT-2 patch is present and compatible with this checkout.')
if not GGUF.is_file() or GGUF.stat().st_size < 100_000_000:
    raise RuntimeError('Converted file missing or unexpectedly small; do not register it in Ollama.')

modelfile = OUT / 'Modelfile'
if modelfile.exists():
    raise FileExistsError(f'Modelfile already exists; inspect before replacing: {modelfile}')
modelfile.write_text('''FROM ./maia-lite-0.2.0-f16.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0
PARAMETER num_ctx 1024
''', encoding='utf-8')
sha = hashlib.sha256()
with GGUF.open('rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024), b''): sha.update(chunk)
(OUT / 'SHA256SUMS.txt').write_text(f'{sha.hexdigest()}  {GGUF.name}\n')
print('\nGGUF exported:', GGUF, 'MiB:', round(GGUF.stat().st_size / 2**20, 1))
print('Modelfile:', modelfile)
print('Next, on a host with Ollama installed:')
print(f'  cd {OUT}')
print(f'  ollama create {MODEL} -f Modelfile')
print(f'  ollama run {MODEL}')
print('Compare generated outputs against the HF model before calling the export validated.')
