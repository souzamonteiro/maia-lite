# MAIA LITE — REFERENCE MODEL BENCHMARK (ONE COLAB CELL)
# Same 66 questions as the first Maia Lite benchmark plus 12 Evaluation II holdouts.
# Loads ONE model at a time. Does not train or change Maia checkpoints.
import os, sys, gc, re, ast, csv, json, math, time, tempfile, subprocess, textwrap, shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/content/drive/MyDrive/Colab_LLMs')
STAGE = ROOT / 'Maia_Lite_Stage2'
SOURCE = STAGE / 'reports/maia-lite-0.2.0/robust_diagnostics/all_results.json'
OUT = STAGE / 'reports/maia-lite-0.2.0/reference_models'
OUT.mkdir(parents=True, exist_ok=True)
FUNCTION_TIMEOUT_SECONDS = 2.5
MAX_NEW_TOKENS = 96  # identical to the original 66-test benchmark
SEED = 1729
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = (torch.bfloat16 if DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32)
# FP16 is intentionally not used: Gemma 3 270M generated only PAD tokens in FP16.
SANITY_PROMPT = 'Reply with exactly the word READY.'
LIMIT_TESTS = None  # None = all 66; 6 = quick smoke test
RUN_FRESH_HOLDOUT = True  # also include the 12 Evaluation II holdout questions

# Instruction-tuned models (comparable in task, not identical training regime).
# Base models below are for a separate pretraining comparison, NOT an instruction leaderboard.
REFERENCES = [
    {'name':'gemma3_270m_it','repo':'google/gemma-3-270m-it','mode':'chat','enabled':True,'remote_code':False,'family':'instruct'},
    {'name':'smollm2_360m_it','repo':'HuggingFaceTB/SmolLM2-360M-Instruct','mode':'chat','enabled':True,'remote_code':False,'family':'instruct'},
    {'name':'openelm_270m_it','repo':'apple/OpenELM-270M-Instruct','mode':'plain','enabled':False,'remote_code':True,'family':'instruct'},
    {'name':'mobilellm_350m','repo':'facebook/MobileLLM-350M','mode':'plain','enabled':False,'remote_code':True,'family':'base'},
    {'name':'pythia_410m','repo':'EleutherAI/pythia-410m','mode':'plain','enabled':False,'remote_code':False,'family':'base'},
]
# For remote-code models, review the repository's Python files before enabling.
ALLOW_REMOTE_CODE = False
# Gemma is access-controlled in some regions/accounts; accept license and log in
# to Hugging Face in Colab if required. Failed loads are recorded, not silently scored.

if not SOURCE.is_file():
    raise FileNotFoundError(f'Maia benchmark not found: {SOURCE}. Run the 66-test diagnostic first.')
previous = json.loads(SOURCE.read_text(encoding='utf-8'))
questions_by_id = {}
for row in previous:
    questions_by_id.setdefault(row['id'], {k:row.get(k) for k in ('id','language','domain','question','expected','kind')})
TESTS = list(questions_by_id.values())
if LIMIT_TESTS is not None: TESTS = TESTS[:LIMIT_TESTS]
if not TESTS: raise RuntimeError('Empty reference benchmark.')
print('Reference benchmark:',len(TESTS),'same questions as Maia Lite 0.2.0')

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



FRESH = [
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
    for tid,domain,kind,question,expected in FRESH:
        TESTS.append({'id':tid,'domain':domain,'kind':kind,'question':question,'expected':expected,
                      'language':'pt' if tid.startswith('hold_pt') else 'en'})

# The original Maia benchmark used its own plain System/User/Assistant format.
# Reference instruction models use their OFFICIAL chat templates; this is a
# fairer task-level comparison, not a tokenizer/prompt-identical experiment.
def make_inputs(tok, test, spec):
    question=test['question']
    if spec['mode']=='chat':
        if not getattr(tok,'chat_template',None):
            raise ValueError('Chat template unavailable; refusing to guess a template')
        messages=[{'role':'user','content':question}]
        encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                        return_dict=True,return_tensors='pt')
    else:
        encoded=tok(question+'\nAnswer:',return_tensors='pt',add_special_tokens=True)
    return {k:v.to(DEVICE) for k,v in encoded.items()}

@torch.inference_mode()
def generate_ref(model,tok,test,spec):
    encoded=make_inputs(tok,test,spec)
    input_len=encoded['input_ids'].shape[-1]
    kwargs={'max_new_tokens':MAX_NEW_TOKENS,'do_sample':False,'repetition_penalty':1.0,
            'pad_token_id':tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}
    # Do NOT override model.generation_config.eos_token_id: Gemma uses [1,106].
    output=model.generate(**encoded,**kwargs)
    text=tok.decode(output[0,input_len:],skip_special_tokens=True).strip()
    return text,int(output.shape[-1]-input_len)

# Load all old Maia rows for side-by-side reporting; never regenerate Maia.
results=[]
ids={t['id'] for t in TESTS}
for row in previous:
    if row['id'] in ids and row['model'] in ('base','lora','merged'):
        r=evaluate(row)
        r['family']='maia'
        r['model_repo']='local checkpoint'
        r['prompt_style']='maia_system_user_assistant'
        results.append(r)

# Include prior Evaluation II holdout results if present.
EVAL2 = STAGE / 'reports/maia-lite-0.2.0/evaluation_2/evaluation2_results.csv'
if RUN_FRESH_HOLDOUT and EVAL2.exists():
    with EVAL2.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r.get('id','').startswith('hold_') and r.get('model') in ('base','lora','merged'):
                r['family']='maia'; r['model_repo']='local checkpoint';r['prompt_style']='maia_system_user_assistant'
                if r.get('eval2_status') not in ('pass','fail','manual','unresolved'):
                    r=evaluate(r)
                results.append(r)

status=[]
torch.manual_seed(SEED)
for spec in REFERENCES:
    if not spec['enabled']: continue
    repo=spec['repo'];name=spec['name']
    print(f'\n===== {name} | {repo} =====')
    if spec['remote_code'] and not ALLOW_REMOTE_CODE:
        print('SKIPPED: requires explicit ALLOW_REMOTE_CODE=True after reviewing repository')
        status.append({'model':name,'status':'skipped','reason':'remote code opt-in required'})
        continue
    model=None;tok=None
    try:
        tok=AutoTokenizer.from_pretrained(repo,trust_remote_code=spec['remote_code'],use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token=tok.eos_token if tok.eos_token is not None else tok.unk_token
        model=AutoModelForCausalLM.from_pretrained(repo,dtype=DTYPE,
                   trust_remote_code=spec['remote_code'],low_cpu_mem_usage=True).to(DEVICE).eval()
        n_params=sum(p.numel() for p in model.parameters())
        print('Parameters:',f'{n_params:,}','device:',DEVICE,'dtype:',DTYPE)
        print('Generation EOS:',model.generation_config.eos_token_id)
        # Sanity-check decoding before processing the benchmark. A nonempty
        # answer is necessary but NOT proof that its content is correct.
        sanity_test={'question':SANITY_PROMPT}
        sanity_text,sanity_tokens=generate_ref(model,tok,sanity_test,spec)
        print('Sanity output:',repr(sanity_text),'generated tokens:',sanity_tokens)
        if not sanity_text.strip():
            raise RuntimeError('Sanity check failed: empty generation. Skipping model rather than scoring invalid answers.')
        for i,test in enumerate(TESTS,1):
            start=time.monotonic()
            try:
                answer,n_tokens=generate_ref(model,tok,test,spec)
                error=None
            except Exception as exc:
                answer='';n_tokens=0;error=f'{type(exc).__name__}: {exc}'
            row={**test,'model':name,'answer':answer,'error':error,'tokens_generated':n_tokens,
                 'seconds':round(time.monotonic()-start,3),'family':spec['family'],
                 'model_repo':repo,'parameters':n_params,'prompt_style':spec['mode']}
            assessed=evaluate(row)
            results.append(assessed)
            print(f'{i:02}/{len(TESTS)} {test["id"]:17} {assessed["eval2_status"]:10} {answer[:80]!r}')
        status.append({'model':name,'status':'completed','parameters':n_params,'dtype':str(DTYPE),'sanity_output':sanity_text})
    except Exception as exc:
        msg=f'{type(exc).__name__}: {exc}'
        print('MODEL LOAD FAILED:',msg[:600])
        status.append({'model':name,'status':'failed','reason':msg})
    finally:
        del model,tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Save incrementally in case Colab disconnects.
        (OUT/'reference_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
        (OUT/'model_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')

# Drop duplicate rows if a user reruns the cell or earlier CSV contains repeats.
unique={}
for row in results: unique[(row['model'],row['id'])]=row
results=list(unique.values())
(OUT/'reference_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
fields=sorted(set().union(*(r.keys() for r in results)))
with (OUT/'reference_results.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
    for row in results:
        w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

summary=[]
for model in sorted({r['model'] for r in results}):
    for domain in ('language','mathematics','programming','scientific_reasoning'):
        for suite, pred in [('original66',lambda r:not r['id'].startswith('hold_')),
                            ('holdout12',lambda r:r['id'].startswith('hold_'))]:
            subset=[r for r in results if r['model']==model and r['domain']==domain and pred(r)]
            if not subset:continue
            passes=sum(r['eval2_status']=='pass' for r in subset)
            fails=sum(r['eval2_status']=='fail' for r in subset)
            unresolved=sum(r['eval2_status']=='unresolved' for r in subset)
            manual=sum(r['eval2_status']=='manual' for r in subset)
            summary.append({'model':model,'suite':suite,'domain':domain,'total':len(subset),
                            'pass':passes,'fail':fails,'unresolved':unresolved,'manual':manual,
                            'confirmed_pass_rate_all':round(passes/len(subset),4)})
(OUT/'reference_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'run_config.json').write_text(json.dumps({'date':datetime.now(timezone.utc).isoformat(),
     'max_new_tokens':MAX_NEW_TOKENS,'seed':SEED,'device':DEVICE,'dtype':str(DTYPE),'sanity_prompt':SANITY_PROMPT,
     'references':REFERENCES,'run_holdout':RUN_FRESH_HOLDOUT,
     'method':'same questions and scorer; official chat templates for instruct models; Maia historic prompt'},
     indent=2,ensure_ascii=False),encoding='utf-8')
print('\n===== VERIFIED RESULTS =====')
for s in summary:
    if s['domain']!='scientific_reasoning':
        print(f"{s['model']:19} {s['suite']:11} {s['domain']:14} "
              f"pass={s['pass']:2}/{s['total']:2} fail={s['fail']:2} "
              f"unresolved={s['unresolved']:2} manual={s['manual']:2}")
print('\nSAVED:',OUT)
print('reference_results.csv | reference_summary.json | reference_results.json | model_status.json')
print('IMPORTANT: model generations have not been run in this authoring environment.')
