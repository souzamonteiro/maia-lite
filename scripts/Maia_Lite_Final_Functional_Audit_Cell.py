# MAIA LITE — FUNCTIONAL BENCHMARK AUDIT v3 (ONE GOOGLE COLAB CELL)
# Uses saved generations only; NO GPU, NO new inference, NO overwriting earlier reports.
# Generated Python is untrusted. The child process has restricted AST, limited
# builtins, -I -S, temp working dir and OS limits. THIS IS NOT A HARDENED SANDBOX.
# For adversarial/untrusted submissions, run in an isolated disposable container/VM
# with network disabled and resource quotas, NOT on your personal Colab session.
import ast, csv, json, math, os, re, sys, subprocess, tempfile, hashlib, collections, textwrap
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_Stage2/reports/maia-lite-0.2.0')
SOURCES = [ROOT/'reference_models'/'reference_results.json',
           ROOT/'reference_models_05b'/'reference_results.json']
META_SOURCE = ROOT/'robust_diagnostics'/'all_results.json'
OUT = ROOT/'reference_functional_audit_v3'
OUT.mkdir(parents=True, exist_ok=True)
TIMEOUT_SECONDS = 4

# Test cases copied from the benchmark reference-model cell, so comparisons
# remain compatible. Each case is (positional arguments, expected result).
CASES = {
 'factorial':[([0],1),([1],1),([5],120),([7],5040)],
 'mean_list':[([[2,4,6,8]],5),([[10]],10),([[1,2,6]],3)],
 'is_prime':[([0],False),([1],False),([2],True),([3],True),([4],False),([9],False),([17],True)],
 'fibonacci':[([0],0),([1],1),([2],1),([7],13),([10],55)],
 'reverse_string':[(['abc'],'cba'),([''],''),(['Maia'],'aiaM')],
 'count_vowels':[(['hello'],2),(['xyz'],0),(['AEIOU'],5)],
 'maximum':[([[1,9,3]],9),([[-4,-2,-9]],-2),([[5]],5)],
 'gcd':[([12,18],6),([7,13],1),([0,9],9)],
 'is_palindrome':[(['radar'],True),(['abc'],False),(['a'],True)],
 'variance':[([[2,4,6]],8/3),([[5,5,5]],0),([[1,3]],1)],
 'celsius_to_fahrenheit':[([0],32),([100],212),([-40],-40)],
 'binary_search':[([[1,3,5,7],5],2),([[1,3,5,7],4],-1),([[8],8],0)]
}
HOLD = {
 'hold_math_01':('mathematics','numeric',43),
 'hold_math_02':('mathematics','numeric',9),
 'hold_math_03':('mathematics','numeric',7),
 'hold_math_04':('mathematics','numeric',30),
 'hold_math_05':('mathematics','numeric',10),
 'hold_math_06':('mathematics','numeric',10),
 'hold_code_01':('programming','python_syntax','factorial'),
 'hold_code_02':('programming','python_syntax','is_prime'),
 'hold_code_03':('programming','python_syntax','celsius_to_fahrenheit'),
 'hold_code_04':('programming','python_syntax','mean_list'),
 'hold_pt_01':('language','exact','SIM'),
 'hold_en_01':('language','exact','YES')
}

if not META_SOURCE.exists(): raise FileNotFoundError(META_SOURCE)
meta_raw=json.loads(META_SOURCE.read_text(encoding='utf-8'))
if isinstance(meta_raw,dict): meta_raw=meta_raw.get('results',meta_raw.get('rows',[]))
meta={str(r['id']):{'domain':r.get('domain'),'kind':r.get('kind'),'expected':r.get('expected')}
      for r in meta_raw if isinstance(r,dict) and r.get('id')}
meta.update({k:{'domain':v[0],'kind':v[1],'expected':v[2]} for k,v in HOLD.items()})

# Restrictive allowlist: intentionally excludes imports, attributes, classes,
# decorators, comprehensions with side effects, context managers, etc.
# A rejected answer is REVIEW, not FAIL: this avoids artificial false negatives.
ALLOWED_NODES = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,
 ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Name,ast.Load,ast.Store,ast.Constant,
 ast.Expr,ast.If,ast.For,ast.While,ast.Break,ast.Continue,ast.Pass,
 ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.IfExp,ast.Call,
 ast.List,ast.Tuple,ast.Dict,ast.Set,ast.Subscript,ast.Slice,
 ast.ListComp,ast.GeneratorExp,ast.comprehension,ast.keyword,
 ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,
 ast.USub,ast.UAdd,ast.Not,ast.And,ast.Or,ast.Eq,ast.NotEq,
 ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Is,ast.IsNot,ast.In,ast.NotIn)
SAFE_BUILTINS=['abs','all','any','bool','dict','enumerate','filter','float','int',
 'isinstance','len','list','map','max','min','pow','range','reversed','round',
 'set','sorted','str','sum','tuple','zip']
DENY={'eval','exec','compile','open','input','globals','locals','vars','__import__',
 'getattr','setattr','delattr','breakpoint','help','dir','object','type','super'}

def extract_function(answer, fn):
    if not isinstance(answer,str) or not answer.strip(): return None,'empty output'
    fenced=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S|re.I)
    candidates=fenced+[answer]
    for candidate in candidates:
        lines=candidate.replace('\r','').splitlines()
        for i,line in enumerate(lines):
            if not re.match(r'^\s*def\s+'+re.escape(fn)+r'\s*\(',line): continue
            indent=len(line)-len(line.lstrip())
            selected=[line]
            for nxt in lines[i+1:]:
                if nxt.strip() and len(nxt)-len(nxt.lstrip())<=indent: break
                selected.append(nxt)
            code=textwrap.dedent('\n'.join(selected)).strip()
            try:
                tree=ast.parse(code)
                if any(isinstance(n,ast.FunctionDef) and n.name==fn for n in tree.body):
                    return code,'function extracted'
            except SyntaxError: pass
    return None,'no parseable function definition found'

def audit_function(code,fn):
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        return None,'multiple/top-level statements'
    f=tree.body[0]
    if f.name!=fn or f.decorator_list or f.returns is not None:
        return None,'function mismatch/decorators/annotations'
    for node in ast.walk(tree):
        if not isinstance(node,ALLOWED_NODES):
            return None,'restricted syntax: '+type(node).__name__
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in DENY):
            return None,'restricted identifier: '+node.id
        if isinstance(node,ast.Call) and not isinstance(node.func,ast.Name):
            return None,'method/attribute call not permitted'
        if isinstance(node,ast.Call) and node.func.id not in SAFE_BUILTINS+[fn]:
            return None,'external function call: '+node.func.id
        if isinstance(node,ast.Constant) and isinstance(node.value,(bytes,complex)):
            return None,'unsupported constant'
    if len(list(ast.walk(tree)))>600: return None,'AST too large'
    return ast.unparse(tree),'restricted AST accepted'

# Separate child with no site imports. Only whitelisted builtins exposed.
# RLIMIT_AS/CPU/FSIZE are applied where supported (Linux Colab).
CHILD = r'''
import sys,json,math
try:
 import resource
 resource.setrlimit(resource.RLIMIT_CPU,(2,3))
 resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
 resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
except (ImportError,OSError,ValueError): pass
p=json.loads(sys.stdin.read())
allowed={name:getattr(__builtins__,name) for name in p['builtins']}
ns={'__builtins__':allowed}
try:
 exec(compile(p['code'],'<candidate>','exec'),ns,ns)
 fn=ns[p['function']]
 results=[]
 for args,expected in p['cases']:
  try:
   actual=fn(*args)
   if type(expected)==bool: passed=type(actual)==bool and actual==expected
   elif isinstance(expected,(int,float)):
    passed=type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-7)
   else: passed=type(actual)==type(expected) and actual==expected
   results.append({'args':args,'expected':expected,'actual':repr(actual)[:150],'passed':bool(passed)})
  except BaseException as e:
   results.append({'args':args,'expected':expected,'actual':type(e).__name__+': '+str(e)[:120],'passed':False})
 print(json.dumps({'cases':results}))
except BaseException as e:
 print(json.dumps({'error':type(e).__name__+': '+str(e)[:160]}))
'''

def evaluate_code(answer,fn):
    if fn not in CASES: return 'review','missing functional test suite',[],None
    code,why=extract_function(answer,fn)
    if code is None: return 'review',why,[],None
    checked,why=audit_function(code,fn)
    if checked is None: return 'review',why,[],code
    payload={'code':checked,'function':fn,'cases':CASES[fn],'builtins':SAFE_BUILTINS}
    try:
        with tempfile.TemporaryDirectory(prefix='maia_func_audit_') as temp:
            env={'PATH':'/usr/bin:/bin','PYTHONIOENCODING':'utf-8'}
            proc=subprocess.run([sys.executable,'-I','-S','-c',CHILD],
                input=json.dumps(payload),text=True,capture_output=True,
                timeout=TIMEOUT_SECONDS,cwd=temp,env=env)
        if proc.returncode!=0: return 'review',f'child exit {proc.returncode}: {proc.stderr[:150]}',[],code
        result=json.loads(proc.stdout.strip())
        if 'error' in result: return 'fail',result['error'],[],code
        cases=result['cases']; passed=sum(bool(x['passed']) for x in cases)
        status='pass' if passed==len(cases) else 'fail'
        return status,f'{passed}/{len(cases)} functional tests',cases,code
    except subprocess.TimeoutExpired:
        return 'fail','timeout during functional tests',[],code
    except Exception as e:
        return 'review',f'runner exception: {type(e).__name__}: {str(e)[:120]}',[],code

# Conservative numeric audit: parse a single final numeric result, final
# equation, or explicit answer phrase. Never mine intermediate calculations.
NUMBER=r'[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?'
WORDS={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,
 'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,
 'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,
 'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50}

def parse_simple(s):
    s=s.strip().strip('`$').strip().rstrip('.! ').strip()
    s=s.replace('−','-').replace('\\(','').replace('\\)','')
    s=re.sub(r'\\boxed\{([^{}]+)\}',r'\1',s)
    if s.casefold() in WORDS: return float(WORDS[s.casefold()])
    if re.fullmatch(NUMBER,s): return float(s.replace(',','.'))
    return None

def evaluate_math(answer,expected):
    try: target=float(expected)
    except (TypeError,ValueError): return 'review','invalid expected numeric value',None
    if not isinstance(answer,str) or not answer.strip(): return 'fail','empty output',None
    s=answer.strip().replace('\r','')
    val=parse_simple(s)
    reason='standalone answer'
    if val is None:
        # Last line can be a final equality or a short final statement.
        last=next((line.strip() for line in reversed(s.splitlines()) if line.strip()),'')
        val=parse_simple(last)
        reason='last line'
        if val is None:
            m=re.search(r'(?:\bx\s*=|=|\bis\b|\bequals\b|\bé\b|\bes\b)\s*('+NUMBER+r')\s*[.!]?\s*$',last,re.I)
            if m:
                val=float(m.group(1).replace(',','.'));reason='final equality'
        if val is None:
            m=re.search(r'(?i)(?:final answer|answer|resultado|respuesta)\s*(?:is|es|é|:|=)\s*('+NUMBER+r')\s*[.!]?\s*$',s)
            if m:
                val=float(m.group(1).replace(',','.'));reason='explicit final answer'
    if val is None: return 'review','final numeric answer ambiguous; human review required',None
    status='pass' if math.isclose(val,target,rel_tol=1e-8,abs_tol=1e-8) else 'fail'
    return status,f'{reason}: {val:g} vs {target:g}',val

def evaluate_exact(answer,expected):
    ans=str(answer or '').strip(); exp=str(expected or '').strip()
    if ans==exp: return 'pass','exact match'
    if ans.strip(' .!¡¿').casefold()==exp.casefold(): return 'review','format/case variation'
    return 'fail','did not follow exact output requirement'

rows=[]; seen={}; provenance=[]; duplicates=collections.Counter(); conflicts=[]
for source in SOURCES:
    if not source.is_file(): raise FileNotFoundError(source)
    raw=json.loads(source.read_text(encoding='utf-8'))
    if isinstance(raw,dict): raw=raw.get('results',raw.get('rows',[]))
    if not isinstance(raw,list): raise ValueError(f'Expected list in {source}')
    provenance.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(raw)})
    for r in raw:
        if not isinstance(r,dict): continue
        rid=str(r.get('id') or r.get('test_id') or '')
        model=str(r.get('model') or r.get('model_name') or r.get('name') or '')
        if rid not in meta: raise KeyError(f'Missing metadata for {rid}')
        answer=r.get('answer',r.get('response',r.get('output','')))
        if not isinstance(answer,str): answer=str(answer or '')
        key=(model,rid)
        if key in seen:
            duplicates[model]+=1
            if seen[key]['answer']!=answer:
                conflicts.append({'model':model,'id':rid,'first_source':seen[key]['source'],'second_source':str(source)})
            continue
        m=meta[rid];kind=m['kind'];expected=m['expected']
        cases=[];extracted=None;numeric=None
        if kind=='python_syntax':
            status,reason,cases,extracted=evaluate_code(answer,str(expected))
        elif kind=='numeric':
            status,reason,numeric=evaluate_math(answer,expected)
        elif kind=='exact':
            status,reason=evaluate_exact(answer,expected)
        else: status,reason='manual','qualitative response requires human rubric'
        item={'model':model,'id':rid,'split':'holdout12' if rid.startswith('hold_') else 'original66',
              'domain':m['domain'],'kind':kind,'expected':expected,
              'old_status':r.get('eval2_status',r.get('status','')),
              'status':status,'reason':reason,'numeric_extracted':numeric,
              'functional_cases':cases,'extracted_code':extracted,'answer':answer,
              'source':str(source)}
        seen[key]=item;rows.append(item)

if conflicts:
    print('WARNING: conflicting duplicate generations:',len(conflicts),
          '(saved in duplicate_conflicts.json; first version retained)')
print('Loaded unique model/question rows:',len(rows))
print('Identical/duplicate keys by model:',dict(duplicates))

summary=[];groups=collections.defaultdict(list)
for row in rows: groups[(row['model'],row['domain'])].append(row)
print('\n=== FUNCTIONAL AUDIT v3 ===')
for (model,domain),items in sorted(groups.items()):
    cnt=collections.Counter(x['status'] for x in items)
    n=len(items)
    print(f'{model:27s} {domain:21s} pass={cnt["pass"]:2d}/{n:2d} fail={cnt["fail"]:2d} review={cnt["review"]:2d} manual={cnt["manual"]:2d}')
    summary.append({'model':model,'domain':domain,'total':n,
                    'pass':cnt['pass'],'fail':cnt['fail'],
                    'review':cnt['review'],'manual':cnt['manual']})

(OUT/'audit_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
columns=['model','id','split','domain','kind','expected','old_status','status','reason','numeric_extracted','answer']
with (OUT/'audit_results.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r.get(k) for k in columns} for r in rows)
with (OUT/'review_queue.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r.get(k) for k in columns} for r in rows if r['status'] in ('review','manual'))
(OUT/'duplicate_conflicts.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf-8')
report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'inputs':provenance,'summary':summary,'duplicate_counts':dict(duplicates),
        'duplicate_conflicts':len(conflicts),
        'method':'saved generation audit; restricted subprocess functional tests; no inference',
        'cautions':[
          'Restricted subprocess is not a hardened sandbox; use disposable network-isolated VM/container for adversarial code.',
          'Unsupported Python syntax is marked review, never automatically fail.',
          'A functional pass means passing the small fixed test suite, not formal correctness.',
          'Numeric extraction is conservative; review responses may contain correct results.',
          'Qualitative language/science answers remain manual; exact-format tasks are distinct.',
          'Duplicates with conflicting answers are retained from first source and flagged.'
        ]}
(OUT/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('\nSaved to:',OUT)
print('audit_results.json | audit_results.csv | review_queue.csv | audit_summary.json | duplicate_conflicts.json')
print('NOTE: manual review remains necessary; this is NOT a fully validated accuracy estimate.')
