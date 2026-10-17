# MAIA LITE 0.2.0 — ROBUST COMPARATIVE DIAGNOSTICS
# Run as ONE Colab cell after mounting Google Drive.
# Works with pre-existing torchao 0.10.0 without a runtime restart.
# No training, no overwrites of model files, no execution of generated code.
import os, sys, gc, re, ast, csv, json, math, time, hashlib, importlib.util
from pathlib import Path
from collections import defaultdict

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from peft import PeftModel

ROOT = Path('/content/drive/MyDrive/Colab_LLMs')
BASE_DIR = ROOT/'Maia_Lite_Stage2/checkpoints/full/checkpoint-213624'
ADAPTER_DIR = ROOT/'Maia_Lite_Stage2/sft_runs/stage2_multidomain_v3/checkpoint-75000'
RELEASE_DIR = ROOT/'Maia_Lite_Stage2/releases/maia-lite-0.2.0-hf'
TOKENIZER_DIR = ROOT/'Maia_Lite_v3/tokenizer'
OUT = ROOT/'Maia_Lite_Stage2/reports/maia-lite-0.2.0/robust_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

# Change only these knobs to adjust runtime.
MODELS = ['base', 'lora', 'merged']  # Set ['lora','merged'] to shorten runtime.
MAX_NEW_TOKENS = 96
REPETITION_PENALTY = 1.0  # Unmodified greedy decoding for a controlled comparison.
LIMIT_TESTS = None         # e.g. 12 for a quick trial; None runs everything.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
SEED = 1729

# GPT-2 uses ordinary Conv1D/Linear layers, not torchao quantized layers.
# PEFT may raise on an old, unrelated torchao installation (e.g. 0.10.0).
# Disable ONLY the optional torchao LoRA dispatcher in this process, without
# uninstalling packages or restarting the notebook.
try:
    import peft.tuners.lora.torchao as _peft_torchao_dispatcher
    _peft_torchao_dispatcher.is_torchao_available = lambda: False
    print('Optional PEFT torchao dispatcher disabled for standard GPT-2 weights.')
except (ImportError, AttributeError) as exc:
    print('Optional torchao dispatcher not applicable:', type(exc).__name__)

for name, path in [('base',BASE_DIR),('adapter',ADAPTER_DIR),('release',RELEASE_DIR),('tokenizer',TOKENIZER_DIR)]:
    if not path.is_dir(): raise FileNotFoundError(f'{name}: {path}')

torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True, trust_remote_code=False)
if tokenizer.eos_token_id is None: raise RuntimeError('EOS missing')
if tokenizer.pad_token_id is None: raise RuntimeError('PAD missing; verify the original tokenizer')

# Objective math problems have explicit, independently known answers.
# Other tests are qualitative and MUST be reviewed by a human.
TESTS = []
def add(id, lang, domain, question, expected=None, kind='qualitative'):
    TESTS.append(dict(id=id, language=lang, domain=domain, question=question, expected=expected, kind=kind))

# Conversation, instructions and scientific reasoning (18)
for lang, questions in {
'pt': [
('Apresente-se em duas frases.', None),
('Responda apenas com a palavra SIM.', 'SIM'),
('Qual é a capital do Brasil? Responda em uma palavra.', 'Brasília'),
('Explique a diferença entre correlação e causalidade em duas frases.', None),
('Traduza para inglês: O experimento terminou.', None),
('Qual é maior: 0,8 ou 0,75? Responda apenas com o número.', '0,8'),
],
'en': [
('Introduce yourself in two sentences.', None),
('Reply with only the word YES.', 'YES'),
('What is the capital of France? Answer with one word.', 'Paris'),
('Explain the difference between correlation and causation in two sentences.', None),
('Translate to Portuguese: The experiment is complete.', None),
('Which is greater: 0.8 or 0.75? Answer with the number only.', '0.8'),
],
'es': [
('Preséntate en dos frases.', None),
('Responde únicamente con la palabra SÍ.', 'SÍ'),
('¿Cuál es la capital de Argentina? Responde con una palabra.', 'Buenos Aires'),
('Explica la diferencia entre correlación y causalidad en dos frases.', None),
('Traduce al inglés: El experimento ha terminado.', None),
('¿Cuál es mayor: 0,8 o 0,75? Responde solo con el número.', '0,8'),
]}.items():
    for i,(q,expect) in enumerate(questions,1):
        add(f'{lang}_language_{i:02}',lang,'language',q,expect,'exact' if expect else 'qualitative')

# Mathematics: use short final-answer prompts to allow strict numerical scoring (24)
math_cases = [
('What is 7 + 8? Answer with the number only.',15),
('What is 19 - 27? Answer with the number only.',-8),
('What is 12 times 13? Answer with the number only.',156),
('What is 144 divided by 12? Answer with the number only.',12),
('What is 2 to the power of 8? Answer with the number only.',256),
('What is 25 percent of 80? Answer with the number only.',20),
('Solve 3x + 7 = 22. Give the value of x only.',5),
('Solve 5x - 9 = 16. Give the value of x only.',5),
('Solve 2x + 3 = 13. Give the value of x only.',5),
('Solve x/4 = 6. Give the value of x only.',24),
('What is the square root of 169? Answer with the positive number only.',13),
('What is 3 squared plus 4 squared? Answer with the number only.',25),
('What is the arithmetic mean of 2, 4, 6, 8? Answer with the number only.',5),
('What is the median of 1, 3, 9? Answer with the number only.',3),
('What is the determinant of the matrix [[1,2],[3,4]]? Answer with the number only.',-2),
('What is the determinant of the matrix [[2,0],[0,5]]? Answer with the number only.',10),
('What is the dot product of [1,2,3] and [4,5,6]? Answer with the number only.',32),
('What is the Euclidean distance between (0,0) and (3,4)? Answer with the number only.',5),
('What is the derivative of x^3 at x=2? Answer with the number only.',12),
('What is the derivative of x^2 + 2x at x=3? Answer with the number only.',8),
('What is the definite integral of 2x from 0 to 3? Answer with the number only.',9),
('What is the definite integral of x from 0 to 4? Answer with the number only.',8),
('How many degrees are in a right angle? Answer with the number only.',90),
('How many sides does a hexagon have? Answer with the number only.',6),
]
for i,(q,v) in enumerate(math_cases,1): add(f'math_{i:02}','en','mathematics',q,v,'numeric')

# Programming: AST syntax is a weak diagnostic, not functional correctness.
programming = [
('Write a Python function factorial(n) returning n! for nonnegative integers.', 'factorial'),
('Write a Python function mean_list(nums) returning the arithmetic mean of a nonempty list.', 'mean_list'),
('Write a Python function is_prime(n) returning True only for prime integers.', 'is_prime'),
('Write a Python function fibonacci(n) returning the nth Fibonacci number.', 'fibonacci'),
('Write a Python function reverse_string(s) returning the reversed string.', 'reverse_string'),
('Write a Python function count_vowels(s) counting vowels in a string.', 'count_vowels'),
('Write a Python function maximum(nums) returning the largest element of a nonempty list.', 'maximum'),
('Write a Python function gcd(a, b) returning the greatest common divisor.', 'gcd'),
('Write a Python function is_palindrome(s) returning whether a string is a palindrome.', 'is_palindrome'),
('Write a Python function variance(nums) returning the population variance of a nonempty list.', 'variance'),
('Write a Python function celsius_to_fahrenheit(c) converting Celsius to Fahrenheit.', 'celsius_to_fahrenheit'),
('Write a Python function binary_search(nums, target) returning the index or -1 for a sorted list.', 'binary_search'),
]
for i,(q,fn) in enumerate(programming,1): add(f'code_{i:02}','en','programming',q,fn,'python_syntax')

# Scientific reasoning and robustness (12), manually assessed.
science = [
('Why does correlation not necessarily imply causation?'),
('What is overfitting in machine learning?'),
('What is the difference between training loss and validation loss?'),
('What is the difference between a CPU and a GPU?'),
('What does the derivative of a function represent?'),
('What does an integral represent geometrically?'),
('Explain what a neural network weight is.'),
('What is a graph in graph theory?'),
('What is the purpose of a random seed in an experiment?'),
('What is the difference between precision and recall?'),
('Explain why a test dataset must be separate from training data.'),
('What is the difference between a probability and a frequency?'),
]
for i,q in enumerate(science,1): add(f'science_{i:02}','en','scientific_reasoning',q)

if LIMIT_TESTS is not None: TESTS = TESTS[:LIMIT_TESTS]
assert len({t['id'] for t in TESTS}) == len(TESTS)

SYSTEM = 'You are Maia Lite, a helpful assistant.'
def make_prompt(question): return f'System: {SYSTEM}\nUser: {question}\nAssistant:'

def strict_number(text):
    # Score only a standalone number (or 'x = NUMBER'), never a number buried in prose.
    s=text.strip().replace('−','-').replace('\\(','').replace('\\)','').strip()
    s=re.sub(r'^(?:x\s*=\s*|answer\s*:\s*|final answer\s*:\s*)','',s,flags=re.I)
    m=re.fullmatch(r'([-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+))(?:\s*[.!])?',s)
    return float(m.group(1).replace(',','.')) if m else None

def extract_python(text):
    match=re.search(r'```(?:python)?\s*([\s\S]*?)```',text,re.I)
    return match.group(1).strip() if match else text.strip()

def score(test, answer):
    kind=test['kind']; expected=test['expected']
    if kind=='qualitative': return None,'manual review required'
    if kind=='numeric':
        number=strict_number(answer)
        if number is None: return False,'not a standalone numeric answer'
        return math.isclose(number,float(expected),rel_tol=1e-8,abs_tol=1e-8),'strict numeric comparison'
    if kind=='exact':
        normalize=lambda s: re.sub(r'[.!。\s]+$','',s.strip()).casefold()
        return normalize(answer)==normalize(str(expected)),'strict text comparison'
    if kind=='python_syntax':
        code=extract_python(answer)
        try:
            tree=ast.parse(code)
            found=any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==expected for node in ast.walk(tree))
            return found,'AST parses and expected function is defined; NOT functional validation'
        except SyntaxError: return False,'invalid Python syntax'
    raise ValueError(kind)

def release_tokenizer_audit():
    rt=AutoTokenizer.from_pretrained(RELEASE_DIR,use_fast=True,trust_remote_code=False)
    assert len(rt)==len(tokenizer),'Tokenizer vocabulary mismatch'
    assert rt.eos_token_id==tokenizer.eos_token_id,'EOS mismatch'
    assert rt.pad_token_id==tokenizer.pad_token_id,'PAD mismatch'
    probes=['Olá, mundo!', 'Hello world!', '¡Hola!', 'System: X\nUser: Y\nAssistant:']
    for p in probes: assert rt.encode(p,add_special_tokens=False)==tokenizer.encode(p,add_special_tokens=False),f'Tokenizer encoding differs: {p}'
    return {'vocab_size':len(rt),'eos_id':rt.eos_token_id,'pad_id':rt.pad_token_id,'probe_encodings_equal':True}

audit=release_tokenizer_audit()
print('Tokenizer audit:',audit)
print('Device:',DEVICE,'dtype:',DTYPE,'tests:',len(TESTS),'models:',MODELS)
print('Total generations:',len(TESTS)*len(MODELS),'max new tokens:',MAX_NEW_TOKENS)

# Single model in memory at a time to avoid A100/Colab OOM.
def load_variant(variant):
    if variant=='merged':
        return GPT2LMHeadModel.from_pretrained(RELEASE_DIR,dtype=DTYPE,low_cpu_mem_usage=True).to(DEVICE).eval()
    base=GPT2LMHeadModel.from_pretrained(BASE_DIR,dtype=DTYPE,low_cpu_mem_usage=True)
    if variant=='base': return base.to(DEVICE).eval()
    if variant=='lora': return PeftModel.from_pretrained(base,ADAPTER_DIR,is_trainable=False).to(DEVICE).eval()
    raise ValueError(variant)

@torch.inference_mode()
def generate(model, question):
    prompt=make_prompt(question)
    inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
    inputs={k:v.to(DEVICE) for k,v in inputs.items()}
    n_input=inputs['input_ids'].shape[1]
    context=getattr(model.config,'n_positions',1024)
    if n_input+MAX_NEW_TOKENS>context: raise ValueError('Context length exceeded')
    output=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                          repetition_penalty=REPETITION_PENALTY,
                          eos_token_id=tokenizer.eos_token_id,
                          pad_token_id=tokenizer.pad_token_id)
    generated=output[0,n_input:]
    return tokenizer.decode(generated,skip_special_tokens=True).strip(),len(generated)

rows=[]
config={'seed':SEED,'max_new_tokens':MAX_NEW_TOKENS,'do_sample':False,
        'repetition_penalty':REPETITION_PENALTY,'prompt_template':'System: ...\\nUser: ...\\nAssistant:',
        'models':MODELS,'test_count':len(TESTS),'tokenizer_audit':audit,
        'base_path':str(BASE_DIR),'adapter_path':str(ADAPTER_DIR),'merged_path':str(RELEASE_DIR),
        'torch_version':torch.__version__}
(OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
for variant in MODELS:
    print(f'\n========== MODEL: {variant.upper()} ==========')
    model=load_variant(variant)
    for idx,test in enumerate(TESTS,1):
        start=time.monotonic()
        try:
            answer,n_tokens=generate(model,test['question'])
            passed,notes=score(test,answer)
            error=None
        except Exception as exc:
            answer='';n_tokens=0;passed=None;notes='generation error';error=f'{type(exc).__name__}: {exc}'
        row={**test,'model':variant,'answer':answer,'passed':passed,'notes':notes,
             'tokens_generated':n_tokens,'seconds':round(time.monotonic()-start,3),'error':error}
        rows.append(row)
        print(f"{idx:02}/{len(TESTS)} {test['id']:17} {str(passed):5} {answer[:95]!r}")
    (OUT/f'{variant}_results.json').write_text(json.dumps([r for r in rows if r['model']==variant],indent=2,ensure_ascii=False),encoding='utf-8')
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

(OUT/'all_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
with (OUT/'all_results.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    writer.writeheader();writer.writerows(rows)

print('\n========== OBJECTIVE SCORES (EXCLUDES QUALITATIVE ITEMS) ==========')
for variant in MODELS:
    subset=[r for r in rows if r['model']==variant]
    for domain in sorted(set(r['domain'] for r in subset)):
        scored=[r for r in subset if r['domain']==domain and r['passed'] is not None]
        if scored:
            n_ok=sum(r['passed'] is True for r in scored)
            print(f'{variant:7} {domain:22} {n_ok:2}/{len(scored):2} ({n_ok/len(scored):.1%})')

# Deterministic text match is useful, but different low-precision kernels can
# occasionally yield distinct tokens even when the model is correctly merged.
if 'lora' in MODELS and 'merged' in MODELS:
    a={r['id']:r for r in rows if r['model']=='lora'}
    b={r['id']:r for r in rows if r['model']=='merged'}
    equal=[k for k in a if a[k]['answer']==b[k]['answer'] and not a[k]['error'] and not b[k]['error']]
    differences=[{'id':k,'lora':a[k]['answer'],'merged':b[k]['answer']} for k in a if k not in equal]
    print(f'\nLoRA vs merged EXACT TEXT MATCH: {len(equal)}/{len(a)}')
    (OUT/'lora_vs_merged_differences.json').write_text(json.dumps(differences,indent=2,ensure_ascii=False),encoding='utf-8')
    if differences: print('Differences saved. Inspect logits and numerical precision before concluding merge failed.')

print('\nDONE. Files saved to:',OUT)
print('Review all qualitative answers manually. Syntax-valid code is NOT necessarily correct.')
