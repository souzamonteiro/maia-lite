# Maia Lite 0.2.0 — one Colab cell: targeted GPT-2 BPE patch + GGUF F16 export
# Never updates llama.cpp or changes model weights.
import sys, json, re, hashlib, shutil, subprocess, datetime
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_Stage2')
hf = root / 'releases/maia-lite-0.2.0-hf'
out = root / 'releases/maia-lite-0.2.0-ollama'
gguf = out / 'maia-lite-0.2.0-f16.gguf'
llama = Path('/content/llama.cpp')
base = llama / 'conversion/base.py'
converter = llama / 'convert_hf_to_gguf.py'
hash_value = 'a727209002a4bef5c366d1ebedf90abc1d4c85d919dfa18a2364d27747dcb599'

assert (hf / 'config.json').is_file() and (hf / 'tokenizer.json').is_file()
assert json.loads((hf / 'config.json').read_text())['model_type'] == 'gpt2'
assert list(hf.glob('*.safetensors'))
assert base.is_file() and converter.is_file(), 'Existing llama.cpp checkout not found at /content/llama.cpp'

source = base.read_text(encoding='utf-8')
start = re.search(r'(?m)^([ \t]*)def get_vocab_base_pre\s*\(', source)
assert start, 'get_vocab_base_pre() not found; refusing blind patch'
indent = len(start.group(1))
next_def = re.search(r'(?m)^' + re.escape(start.group(1)) + r'(?:def |class |@)', source[start.end():])
end = start.end() + next_def.start() if next_def else len(source)
function_text = source[start.start():end]

# Insert only at the documented exception branch, not elsewhere in the converter.
if hash_value in function_text and ('"gpt-2"' in function_text or "'gpt-2'" in function_text):
    print('The Maia GPT-2 tokenizer hash is already mapped in get_vocab_base_pre().')
else:
    exception = re.search(r'(?m)^(?P<space>[ \t]*)raise NotImplementedError\([\'\"]BPE pre-tokenizer was not recognized', function_text)
    if not exception:
        raise RuntimeError('Expected BPE error branch not found. No patch applied; inspect conversion/base.py manually.')
    # Direct return preserves the surrounding function logic and avoids relying on a local variable name.
    spaces = exception.group('space')
    insertion = f'{spaces}if chkhsh == "{hash_value}":\n{spaces}    return "gpt-2"\n'
    patch_at = start.start() + exception.start()
    # Validate that chkhsh is used in this function before applying a patch.
    assert re.search(r'\bchkhsh\b', function_text), 'Hash variable not found; refusing patch'
    backup = base.with_name('base.py.maia-backup-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(base, backup)
    base.write_text(source[:patch_at] + insertion + source[patch_at:], encoding='utf-8')
    print('Patched:', base)
    print('Backup:', backup)

# Syntax check before starting expensive conversion.
subprocess.run([sys.executable, '-m', 'py_compile', str(base)], check=True)
out.mkdir(parents=True, exist_ok=True)
if gguf.exists():
    if gguf.stat().st_size >= 100_000_000:
        raise FileExistsError(f'Existing GGUF found: {gguf}. Verify it before re-running; nothing overwritten.')
    raise FileExistsError(f'Possible partial GGUF found: {gguf}. Inspect/remove manually before retry.')

command = [sys.executable, str(converter), str(hf), '--outfile', str(gguf), '--outtype', 'f16']
print('Converting HF to GGUF F16...')
result = subprocess.run(command, cwd=llama, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout[-10000:])
if result.returncode != 0:
    raise RuntimeError(f'Conversion failed (exit {result.returncode}). Inspect log above; model unchanged.')
if not gguf.exists() or gguf.stat().st_size < 100_000_000:
    raise RuntimeError('Output GGUF missing or too small; do not register it.')

modelfile = out / 'Modelfile'
if not modelfile.exists():
    modelfile.write_text('FROM ./maia-lite-0.2.0-f16.gguf\nTEMPLATE """{{ .Prompt }}"""\nPARAMETER temperature 0\nPARAMETER num_ctx 1024\n', encoding='utf-8')
sha = hashlib.sha256()
with gguf.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        sha.update(block)
(out / 'SHA256SUMS.txt').write_text(f'{sha.hexdigest()}  {gguf.name}\n')
print('GGUF:', gguf, 'size MiB:', round(gguf.stat().st_size / 1048576, 1))
print('Modelfile:', modelfile)
print('Next: ollama create maia-lite:0.2.0 -f Modelfile')
print('Then compare generated tokens/output with the HF release; conversion alone is not runtime validation.')
