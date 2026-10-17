# MAIA LITE — RE-SCORE SAVED REFERENCE GENERATIONS (single Colab cell)
# No model loading, no GPU, no new inference, no changes to original reports.
# This cell uses the ORIGINAL benchmark metadata to recover expected values.
import ast, csv, json, math, re, hashlib, collections
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/content/drive/MyDrive/Colab_LLMs/Maia_Lite_Stage2/reports/maia-lite-0.2.0')
INPUT_DIRS = [ROOT/'reference_models', ROOT/'reference_models_05b']
ORIGINAL = ROOT/'robust_diagnostics/all_results.json'
OUTPUT = ROOT/'reference_rescored_v2'
OUTPUT.mkdir(parents=True, exist_ok=True)
RUN_GENERATED_CODE = False  # IMPORTANT: do not execute arbitrary generated code in Colab.
# Only static code validation is done by default. Dynamic tests need a proper sandbox.

FRESH = {
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
 'hold_en_01':('language','exact','YES'),
}

if not ORIGINAL.is_file():
    raise FileNotFoundError(f'Original benchmark metadata not found: {ORIGINAL}')
original_rows=json.loads(ORIGINAL.read_text(encoding='utf-8'))
if isinstance(original_rows,dict):
    original_rows=original_rows.get('results',original_rows.get('rows',[]))
metadata={}
for r in original_rows:
    if isinstance(r,dict) and r.get('id'):
        metadata.setdefault(str(r['id']),{'domain':r.get('domain'),'kind':r.get('kind'),'expected':r.get('expected')})
for k,(d,kind,expected) in FRESH.items():
    metadata[k]={'domain':d,'kind':kind,'expected':expected}

# Read the FULL generated answers (not the truncated console previews).
def load_results(directory):
    f=directory/'reference_results.json'
    if f.exists():
        data=json.loads(f.read_text(encoding='utf-8'))
        if isinstance(data,dict): data=data.get('results',data.get('rows',[]))
        if not isinstance(data,list): raise ValueError(f'Unexpected JSON structure: {f}')
        return data, f
    f=directory/'reference_results.csv'
    if f.exists():
        with f.open(encoding='utf-8-sig',newline='') as h: return list(csv.DictReader(h)), f
    return [],None

# Extract a numeric FINAL answer conservatively. Never accept an intermediate
# number simply because it matches the expected result.
NUMBER=r'[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?'
NUM_RE=re.compile(NUMBER)

def parse_numeric(s):
    s=s.strip().replace('−','-').replace('–','-')
    s=s.replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','')
    s=re.sub(r'\\boxed\s*\{([^{}]+)\}',r'\1',s)
    s=s.replace('$','').replace('**','').strip()
    # Number in English words, for short single-word answers.
    words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
           'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
           'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
           'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50}
    w=s.strip(' .!').casefold()
    if w in words: return float(words[w])
    if re.fullmatch(NUMBER,s.strip(' .!')):
        return float(s.strip(' .!').replace(',','.'))
    return None

def numeric_final(answer):
    if not isinstance(answer,str) or not answer.strip(): return None,'empty answer'
    s=answer.replace('\r','').strip()
    # Explicit final-answer labels, favor the last one.
    label=list(re.finditer(r'(?im)(?:final\s+(?:answer|value)|answer|resultado|respuesta|therefore|thus|hence)\s*(?:is|es|é|:|=)\s*',s))
    if label:
        tail=s[label[-1].end():].splitlines()[0].strip()
        v=parse_numeric(tail)
        if v is not None: return v,'final-answer label'
        m=re.fullmatch(r'\s*(?:x\s*=\s*)?('+NUMBER+r')\s*[.!]?\s*',tail,re.I)
        if m: return float(m.group(1).replace(',','.')),'final-answer label'
    # Consider only the LAST nonempty line or a one-line answer.
    lines=[x.strip() for x in s.splitlines() if x.strip()]
    last=lines[-1] if lines else ''
    candidates=[last]
    if len(lines)==1: candidates=[s]
    for t in candidates:
        t=t.strip().strip('`').strip()
        v=parse_numeric(t)
        if v is not None: return v,'standalone final value'
        # Explicit variable assignment, simple equality, or 'is 9' at end.
        m=re.search(r'(?:\bx\s*=|=|\bis\b|\bequals\b|\bes\b|\bé\b)\s*('+NUMBER+r')\s*[.!]?\s*$',t,re.I)
        if m: return float(m.group(1).replace(',','.')),'final equality'
        # Plain statement ending with the numerical answer, e.g. '... is 9.'
        m=re.search(r'\b(?:is|equals|é|es|result(?:s)?\s+in)\s+('+NUMBER+r')\s*[.!]?\s*$',t,re.I)
        if m: return float(m.group(1).replace(',','.')),'final statement'
    return None,'no unambiguous final number (manual review)'

def score_math(answer,expected):
    try: target=float(expected)
    except (ValueError,TypeError): return 'review','expected value not numeric',None
    val,reason=numeric_final(answer)
    if val is None: return 'review',reason,None
    ok=math.isclose(val,target,rel_tol=1e-8,abs_tol=1e-8)
    return ('pass' if ok else 'fail'),f'{reason}: {val:g} vs expected {target:g}',val

# Extract function from fenced Markdown or explanatory prose; no code execution.
def extract_function(answer,fn):
    if not isinstance(answer,str) or not answer.strip(): return None,'empty answer'
    blocks=re.findall(r'```(?:python|py)?\s*\n?(.*?)```',answer,re.I|re.S)
    candidates=blocks+[answer]
    for candidate in candidates:
        lines=candidate.replace('\r','').splitlines()
        starts=[i for i,line in enumerate(lines) if re.match(r'^\s*(?:async\s+)?def\s+'+re.escape(fn)+r'\s*\(',line)]
        for start in starts:
            # Extract top-level def, preserving indentation and ending at the next top-level statement.
            base=len(lines[start])-len(lines[start].lstrip())
            selected=[lines[start]]
            for line in lines[start+1:]:
                if line.strip() and len(line)-len(line.lstrip())<=base: break
                selected.append(line)
            import textwrap
            code=textwrap.dedent('\n'.join(selected)).strip()
            try:
                tree=ast.parse(code)
                funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==fn]
                if funcs: return code,'function extracted and syntax valid'
            except SyntaxError: pass
    return None,f'no syntactically valid {fn} function extracted'

# A code answer is NOT a functional pass merely because it parses.
# Retain prior functional pass as a provisional claim only; previous runner
# had restrictive AST rules and can undercount valid functions.
def score_code(answer,fn,old_status,old_reason):
    code,detail=extract_function(answer,str(fn))
    if code is None:
        return 'review',detail,None
    if old_status=='pass':
        return 'pass_provisional','previous functional test passed; '+detail,code
    return 'review','syntax valid; functional correctness NOT established; '+detail,code

def score_exact(answer,expected):
    s=str(answer or '').strip()
    exp=str(expected or '').strip()
    if s==exp: return 'pass','exact match'
    if s.casefold()==exp.casefold(): return 'review','case mismatch'
    if s.strip(' \t\r\n.!¡¿').casefold()==exp.casefold():
        return 'review','punctuation or formatting mismatch'
    return 'fail','does not follow exact-output instruction'

all_rows=[]; provenance=[]
for directory in INPUT_DIRS:
    rows,src=load_results(directory)
    if not src:
        print('NOT FOUND:',directory)
        continue
    print('Loaded:',len(rows),'rows from',src)
    provenance.append({'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rows':len(rows)})
    for row in rows:
        if not isinstance(row,dict): continue
        rid=str(row.get('id') or row.get('test_id') or '')
        meta=metadata.get(rid)
        if not meta:
            print('WARNING: missing benchmark metadata:',rid)
            continue
        name=str(row.get('model') or row.get('model_name') or row.get('name') or 'unknown')
        answer=row.get('answer',row.get('response',row.get('output','')))
        if not isinstance(answer,str): answer=str(answer or '')
        old_status=str(row.get('eval2_status') or row.get('status') or row.get('eval_status') or '')
        old_reason=str(row.get('eval2_reason') or row.get('reason') or '')
        domain=meta['domain']; kind=meta['kind']; expected=meta['expected']
        numeric=None; code=None
        if kind=='numeric':
            status,reason,numeric=score_math(answer,expected)
        elif kind=='python_syntax':
            status,reason,code=score_code(answer,expected,old_status,old_reason)
        elif kind=='exact':
            status,reason=score_exact(answer,expected)
        else:
            status,reason='manual','qualitative task: human evaluation required'
        all_rows.append({'source_dir':directory.name,'model':name,'id':rid,'split':'holdout12' if rid.startswith('hold_') else 'original66',
            'domain':domain,'kind':kind,'expected':expected,'old_status':old_status,'new_status':status,
            'reason':reason,'numeric_extracted':numeric,'function_extracted':bool(code),'answer':answer})

if not all_rows: raise RuntimeError('No reference results found. Check INPUT_DIRS.')
# Avoid accidental duplicate scoring when a model is in both input folders.
unique={}
for row in all_rows:
    key=(row['model'],row['id'])
    if key in unique:
        print('WARNING duplicate model/question:',key,'keeping first')
        continue
    unique[key]=row
all_rows=list(unique.values())

counts=collections.defaultdict(lambda:collections.Counter())
for row in all_rows:
    counts[(row['model'],row['domain'])][row['new_status']]+=1
print('\n=== CONSERVATIVE RE-SCORE (no GPU, no code execution) ===')
summary=[]
for (model,domain),c in sorted(counts.items()):
    total=sum(c.values())
    confirmed=c['pass']
    provisional=c['pass_provisional']
    review=c['review']+c['manual']
    print(f'{model:28s} {domain:12s} confirmed={confirmed:2d}/{total:2d} provisional={provisional:2d} fail={c["fail"]:2d} review={review:2d}')
    summary.append({'model':model,'domain':domain,'total':total,'confirmed_pass':confirmed,
                    'provisional_functional_pass':provisional,'fail':c['fail'],'review':review})

# Save complete answers for review and compact CSV for filtering.
(OUTPUT/'rescored_results.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
fields=['source_dir','model','id','split','domain','kind','expected','old_status','new_status','reason','numeric_extracted','function_extracted','answer']
with (OUTPUT/'rescored_results.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
with (OUTPUT/'review_queue.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    w.writerows(r for r in all_rows if r['new_status'] in ('review','manual','pass_provisional'))
report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'method':'conservative rescore; no model inference; no generated code execution',
        'inputs':provenance,'summary':summary,
        'limitations':[
          'Math: only explicit unambiguous final numbers are automatically graded; multi-line derivations require review.',
          'Programming: extracted code is parsed but NOT executed; prior passes are provisional, not independently reverified.',
          'Exact-output instructions are scored strictly; qualitative language/science tasks need human review.',
          'A review is not an error; confirmed pass is a lower bound, not estimated accuracy.'
        ]}
(OUTPUT/'rescore_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('\nSaved to:',OUTPUT)
print('rescored_results.json | rescored_results.csv | review_queue.csv | rescore_summary.json')
print('IMPORTANT: This is an audit/triage, not a validated functional Python benchmark.')
