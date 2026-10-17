# MAIA LITE 0.2.0 — EVALUATION II (ONE GOOGLE COLAB CELL)
# Reuses 198 existing generations, evaluates mathematical content and Python
# behavior, and optionally generates a small, fresh holdout probe set.
# No training or model files are modified. Mount Google Drive first.
import os, re, sys, ast, gc, csv, json, math, time, tempfile, subprocess, textwrap, shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path('/content/drive/MyDrive/Colab_LLMs')
STAGE = ROOT/'Maia_Lite_Stage2'
SOURCE = STAGE/'reports/maia-lite-0.2.0/robust_diagnostics/all_results.json'
OUT = STAGE/'reports/maia-lite-0.2.0/evaluation_2'
OUT.mkdir(parents=True, exist_ok=True)
RUN_FRESH_HOLDOUT = True       # Set False to rescore the 198 previous answers only.
FRESH_MODELS = ['base','lora','merged']
MAX_NEW_TOKENS = 128
FUNCTION_TIMEOUT_SECONDS = 2.5

if not SOURCE.is_file():
    raise FileNotFoundError(f'First evaluation not found: {SOURCE}\nRun the earlier diagnostic first.')
original = json.loads(SOURCE.read_text(encoding='utf-8'))
print(f'Loaded {len(original)} previous answers; no need to regenerate them.')

# The scorer distinguishes explicit final answers from intermediate calculations.
# It intentionally returns "ambiguous" instead of credit when multiple unrelated
# numbers are present and no final answer is identified.
def numeric_candidates(answer):
    s = answer.replace('−','-').replace('×','*').replace('\\boxed{','').replace('}','')
    # Prefer the LAST explicitly labelled final answer; this prevents giving credit
    # for a correct intermediate number when the model concludes incorrectly.
    labels = list(re.finditer(r'(?i)(?:final\s+answer|answer|resultado|respuesta)\s*[:=]\s*',s))
    tail = s[labels[-1].end():].splitlines()[0] if labels else None
    if tail is None:
        lines = [x.strip() for x in s.splitlines() if x.strip()]
        if len(lines) == 1:
            tail = lines[0]
        else:
            # Last line only if it is a clear numeric answer or x=...
            last = lines[-1] if lines else ''
            tail = last if re.fullmatch(r'(?i)(?:x\s*=\s*)?[-+]?\d+(?:[.,]\d+)?\s*[.!]?',last) else None
    if tail is None: return None, 'no unambiguous final answer'
    # Accept x=5, 5., "is 5", and arithmetic equation ending in = 12;
    # never mine arbitrary numbers from the question echoed in the response.
    tail = tail.strip().rstrip('.! ').strip()
    m = re.search(r'(?:=|\bis\b|\bes\b|\bé\b)\s*([-+]?\d+(?:[.,]\d+)?)\s*$',tail,re.I)
    if not m: m = re.fullmatch(r'([-+]?\d+(?:[.,]\d+)?)',tail)
    if not m: return None, 'final answer not a single numeric value'
    return float(m.group(1).replace(',','.')), 'explicit final numeric answer'

# Some tasks require symbolic equivalents; use SymPy only for parsing a
# standalone final expression, never evaluate arbitrary model-produced Python.
try:
    import sympy as sp
    SYMPY_AVAILABLE=True
except ImportError:
    SYMPY_AVAILABLE=False
    print('SymPy unavailable; numeric checks still run. Optional: %pip install sympy')

def score_math(row):
    value, why = numeric_candidates(row['answer'])
    if value is None: return 'unresolved',why
    expected = float(row['expected'])
    return ('pass' if math.isclose(value,expected,rel_tol=1e-7,abs_tol=1e-7) else 'fail'), f'extracted={value:g}; expected={expected:g}'

# Execute generated functions only in a separate, restricted child process.
# This is a best-effort guard against accidental runaway code, NOT a hardened
# security sandbox. Run only with locally trusted model outputs in Colab.
# Restrict the AST to plain Python functions and common basic operations.
BLOCKED_AST=(ast.Import,ast.ImportFrom,ast.With,ast.AsyncWith,ast.Try,ast.Raise,
             ast.ClassDef,ast.Lambda,ast.Global,ast.Nonlocal,ast.Delete,
             ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom,
             ast.Attribute,ast.NamedExpr)
BLOCKED_NAMES={'eval','exec','compile','open','input','globals','locals','vars',
               '__import__','getattr','setattr','delattr','breakpoint','help',
               'memoryview','object','type','super','classmethod','staticmethod'}
SAFE_BUILTINS=['abs','all','any','bool','dict','enumerate','filter','float','int',
               'isinstance','len','list','map','max','min','pow','range','reversed',
               'round','set','sorted','str','sum','tuple','zip']

def extract_code(text):
    m=re.search(r'```(?:python)?\s*([\s\S]*?)```',text,re.I)
    text=(m.group(1) if m else text).strip()
    # If a model appends explanation after the function, prefer a parseable prefix.
    return text

def audit_ast(code,fn):
    try: tree=ast.parse(code)
    except SyntaxError as e: return None,f'syntax error: {e.msg}'
    if any(isinstance(n,BLOCKED_AST) for n in ast.walk(tree)):
        return None,'blocked AST construct (imports, attributes, etc.)'
    if any(isinstance(n,ast.Name) and (n.id in BLOCKED_NAMES or n.id.startswith('__')) for n in ast.walk(tree)):
        return None,'blocked identifier'
    if any(isinstance(n,ast.Call) and not isinstance(n.func,ast.Name) for n in ast.walk(tree)):
        return None,'only direct function calls allowed'
    if any(isinstance(n,ast.FunctionDef) and n.decorator_list for n in ast.walk(tree)):
        return None,'decorators not allowed'
    # Only function definitions and literal constants at top level: no generated
    # example calls, prints, infinite loops or top-level side effects.
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if not any(n.name==fn for n in functions): return None,f'missing function {fn}'
    filtered=ast.Module(body=functions,type_ignores=[])
    return ast.unparse(filtered),'safe AST subset accepted'

# Multiple independent examples per function; use explicit expected outputs.
CASES={
'factorial': [([0],1),([1],1),([5],120),([7],5040)],
'mean_list': [([[2,4,6,8]],5),([[10]],10),([[1,2,6]],3)],
'is_prime': [([0],False),([1],False),([2],True),([3],True),([4],False),([9],False),([17],True)],
'fibonacci': [([0],0),([1],1),([2],1),([7],13),([10],55)],
'reverse_string': [(['abc'],'cba'),([''],'') ,(['Maia'],'aiaM')],
'count_vowels': [(['hello'],2),(['xyz'],0),(['AEIOU'],5)],
'maximum': [([[1,9,3]],9),([[-4,-2,-9]],-2),([[5]],5)],
'gcd': [([12,18],6),([7,13],1),([0,9],9)],
'is_palindrome': [(['radar'],True),(['abc'],False),(['a'],True)],
'variance': [([[2,4,6]],8/3),([[5,5,5]],0),([[1,3]],1)],
'celsius_to_fahrenheit': [([0],32),([100],212),([-40],-40)],
'binary_search': [([[1,3,5,7],5],2),([[1,3,5,7],4],-1),([[8],8],0)],
}

CHILD = r'''
import sys,json,math
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(2,3))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
except ImportError: pass
p=json.loads(sys.stdin.read())
allowed={name:getattr(__builtins__,name) for name in p['builtins']}
ns={'__builtins__':allowed}
try:
    exec(compile(p['code'],'<generated>','exec'),ns,ns)
    fn=ns[p['function']]
    outputs=[]
    for args,expected in p['cases']:
        try:
            result=fn(*args)
            correct=(isinstance(result,(int,float)) and not isinstance(result,bool) and
                     isinstance(expected,(int,float)) and not isinstance(expected,bool) and
                     math.isclose(result,expected,rel_tol=1e-7,abs_tol=1e-7)) or result==expected
            outputs.append({'args':args,'expected':expected,'actual':repr(result)[:180],'passed':bool(correct)})
        except BaseException as e:
            outputs.append({'args':args,'expected':expected,'actual':type(e).__name__+': '+str(e)[:100],'passed':False})
    print(json.dumps({'cases':outputs}))
except BaseException as e:
    print(json.dumps({'error':type(e).__name__+': '+str(e)[:180]}))
'''

def score_code(row):
    fn=row['expected']
    if fn not in CASES: return 'unresolved','no unit tests',[]
    code,reason=audit_ast(extract_code(row['answer']),fn)
    if code is None: return 'fail',reason,[]
    payload={'code':code,'function':fn,'cases':CASES[fn],'builtins':SAFE_BUILTINS}
    try:
        with tempfile.TemporaryDirectory(prefix='maia_eval_') as d:
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONIOENCODING':'utf-8'}
            proc=subprocess.run([sys.executable,'-I','-S','-c',CHILD],input=json.dumps(payload),
                capture_output=True,text=True,timeout=FUNCTION_TIMEOUT_SECONDS,cwd=d,env=env)
        data=json.loads(proc.stdout.strip())
        if 'error' in data: return 'fail',data['error'],[]
        cases=data['cases']; ok=sum(x['passed'] for x in cases)
        return ('pass' if ok==len(cases) else 'fail'),f'{ok}/{len(cases)} functional cases',cases
    except subprocess.TimeoutExpired: return 'fail','timeout (possible infinite loop/recursion)',[]
    except Exception as e: return 'unresolved',f'runner error: {type(e).__name__}: {e}',[]

def score_language(row):
    # Conservative: check only clear short-answer cases; no claim of semantic QA.
    ans=row['answer'].strip()
    if row['expected'] is None: return 'manual','qualitative: human review required'
    norm=lambda s:re.sub(r'[\s.!¡¿]+$','',s.strip()).casefold()
    if norm(ans)==norm(str(row['expected'])): return 'pass','exact instruction match'
    return 'fail',f'expected exact: {row["expected"]}'

def evaluate(row):
    r=dict(row)
    if row.get('error'): status,reason='unresolved',row['error']; cases=[]
    elif row['kind']=='numeric': status,reason=score_math(row); cases=[]
    elif row['kind']=='python_syntax': status,reason,cases=score_code(row)
    elif row['kind']=='exact': status,reason=score_language(row); cases=[]
    else: status,reason='manual','qualitative: human review required'; cases=[]
    r['eval2_status']=status; r['eval2_reason']=reason; r['unit_cases']=cases
    return r

print('\n===== RE-EVALUATING PREVIOUS RESPONSES =====')
results=[]
for row in original:
    r=evaluate(row); results.append(r)
    if row['kind'] in ('numeric','python_syntax'):
        print(f'{row["model"]:7} {row["id"]:16} {r["eval2_status"]:10} {r["eval2_reason"][:65]}')

# Small NEW holdout set, explicitly separated from the first 66 prompts.
# These are fresh prompts, but not guaranteed absent from SFT training data.
FRESH=[
('hold_math_01','mathematics','numeric','What is 17 plus 26? Give only the final number.',43),
('hold_math_02','mathematics','numeric','Calculate 81 divided by 9. Final answer only.',9),
('hold_math_03','mathematics','numeric','Solve 4x + 3 = 31. Final value of x only.',7),
('hold_math_04','mathematics','numeric','What is 15 percent of 200? Final answer only.',30),
('hold_math_05','mathematics','numeric','Compute the determinant of [[3,1],[2,4]]. Final number only.',10),
('hold_math_06','mathematics','numeric','Find the derivative of x^2 at x=5. Give its numerical value.',10),
('hold_code_01','programming','python_syntax','Write only a Python function factorial(n) for n >= 0.','factorial'),
('hold_code_02','programming','python_syntax','Write only a Python function is_prime(n).','is_prime'),
('hold_code_03','programming','python_syntax','Write only a Python function celsius_to_fahrenheit(c).','celsius_to_fahrenheit'),
('hold_code_04','programming','python_syntax','Write only a Python function mean_list(nums).','mean_list'),
('hold_pt_01','language','exact','Responda somente SIM, sem explicações.','SIM'),
('hold_en_01','language','exact','Reply with exactly YES and nothing else.','YES'),
]

if RUN_FRESH_HOLDOUT:
    import torch
    from transformers import GPT2LMHeadModel,AutoTokenizer
    from peft import PeftModel
    try:
        import peft.tuners.lora.torchao as _peft_torchao
        _peft_torchao.is_torchao_available=lambda:False
    except (ImportError,AttributeError): pass
    BASE=STAGE/'checkpoints/full/checkpoint-213624'
    ADAPTER=STAGE/'sft_runs/stage2_multidomain_v3/checkpoint-75000'
    MERGED=STAGE/'releases/maia-lite-0.2.0-hf'
    TOK=ROOT/'Maia_Lite_v3/tokenizer'
    tokenizer=AutoTokenizer.from_pretrained(TOK,use_fast=True,trust_remote_code=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dtype=torch.float16 if device=='cuda' else torch.float32
    torch.manual_seed(1729)
    def load_model(variant):
        if variant=='merged': return GPT2LMHeadModel.from_pretrained(MERGED,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
        model=GPT2LMHeadModel.from_pretrained(BASE,dtype=dtype,low_cpu_mem_usage=True)
        if variant=='lora': model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False)
        return model.to(device).eval()
    @torch.inference_mode()
    def generate(model,q):
        prompt=f'System: You are Maia Lite, a helpful assistant.\nUser: {q}\nAssistant:'
        encoded=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        encoded={k:v.to(device) for k,v in encoded.items()}
        count=encoded['input_ids'].shape[-1]
        ids=model.generate(**encoded,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                           eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(ids[0,count:],skip_special_tokens=True).strip()
    print('\n===== FRESH HOLDOUT: 12 PROMPTS PER MODEL =====')
    for variant in FRESH_MODELS:
        model=load_model(variant)
        for i,(tid,domain,kind,question,expected) in enumerate(FRESH,1):
            try: answer=generate(model,question);error=None
            except Exception as e: answer='';error=f'{type(e).__name__}: {e}'
            row=dict(id=tid,language='pt' if tid.startswith('hold_pt') else 'en',
                     domain=domain,kind=kind,question=question,expected=expected,
                     model=variant,answer=answer,error=error,source='fresh_holdout')
            r=evaluate(row);results.append(r)
            print(f'{variant:7} {i:02}/{len(FRESH)} {tid:16} {r["eval2_status"]:10} {answer[:72]!r}')
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

for r in results:
    r.setdefault('source','previous_66')

print('\n===== SCORES (PASS/FAIL ONLY; MANUAL AND UNRESOLVED EXCLUDED) =====')
summary=[]
for source in sorted(set(r['source'] for r in results)):
    for model in sorted(set(r['model'] for r in results)):
        for domain in ('language','mathematics','programming'):
            subset=[r for r in results if r['source']==source and r['model']==model and r['domain']==domain]
            assessed=[r for r in subset if r['eval2_status'] in ('pass','fail')]
            passed=sum(r['eval2_status']=='pass' for r in assessed)
            rec=dict(source=source,model=model,domain=domain,passed=passed,assessed=len(assessed),
                     manual=sum(r['eval2_status']=='manual' for r in subset),
                     unresolved=sum(r['eval2_status']=='unresolved' for r in subset))
            if subset:
                summary.append(rec)
                print(f'{source:14} {model:7} {domain:12} {passed:2}/{len(assessed):2} | manual={rec["manual"]} unresolved={rec["unresolved"]}')

(OUT/'evaluation2_full.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
(OUT/'evaluation2_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
flat=[{k:v for k,v in r.items() if k!='unit_cases'} for r in results]
with (OUT/'evaluation2_results.csv').open('w',newline='',encoding='utf-8') as f:
    columns=list(dict.fromkeys(k for row in flat for k in row.keys()))
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(flat)
print('\nDONE —',OUT)
print('Send evaluation2_summary.json and evaluation2_results.csv for analysis.')
print('NOTE: generated Python is executed with AST restrictions, isolated process, and timeout;')
print('this is NOT a hardened security sandbox. Qualitative science answers need human review.')
