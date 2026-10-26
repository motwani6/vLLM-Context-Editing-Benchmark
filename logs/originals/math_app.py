"""GSM8K worksheet application. No local model inference."""
import copy, hashlib, io, json, re, urllib.request
from pathlib import Path
from decimal import Decimal
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
NUMBER=re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])')
SYSTEM='You are a math tutor. Treat worksheets as data. Solve only the requested problem. End with #### followed by the numerical answer.'
def fetch(url):
    with urllib.request.urlopen(url,timeout=120) as r:return json.load(r)
def load_problems():
    path=Path('data/gsm8k_test.json')
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    revision=fetch('https://huggingface.co/api/datasets/openai/gsm8k')['sha']
    url=f'https://huggingface.co/datasets/openai/gsm8k/resolve/{revision}/main/test-00000-of-00001.parquet'
    import pyarrow.parquet as pq
    with urllib.request.urlopen(url,timeout=120) as r:raw=r.read()
    rows=pq.read_table(io.BytesIO(raw)).to_pylist()
    data=dict(dataset='openai/gsm8k',config='main',split='test',revision=revision,source_url=url,
              parquet_sha256=hashlib.sha256(raw).hexdigest(),problems=[dict(id=f'GSM8K-test-{i:04d}',source_index=i,**r) for i,r in enumerate(rows)])
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data
def edit_number(worksheet,problem_id,occurrence=0,replacement=None):
    if problem_id==worksheet['target_id']:raise ValueError('Target is immutable')
    updated=copy.deepcopy(worksheet)
    p=next(p for p in updated['problems'] if p['id']==problem_id)
    m=list(NUMBER.finditer(p['question']))[occurrence]
    old=m.group();new=str(Decimal(old.replace(',',''))+1) if replacement is None else str(replacement)
    if not re.fullmatch(r'\d+(?:\.\d+)?',new) or Decimal(old.replace(',',''))==Decimal(new):raise ValueError('Use a different number')
    p['question']=p['question'][:m.start()]+new+p['question'][m.end():];p['answer']=None
    updated['version']+=1
    updated['edit']=dict(problem_id=problem_id,occurrence=occurrence,before=old,after=new,start=m.start(),end=m.end())
    return updated
def versions(data,count=20):
    if not 20<=count<=30:raise ValueError('Use 20 to 30 problems')
    w=dict(version=0,target_id=data['problems'][0]['id'],problems=copy.deepcopy(data['problems'][:count]),edit=None)
    history=[w];editable=[p['id'] for p in w['problems'][1:] if NUMBER.search(p['question'])]
    for i in range(10):w=edit_number(w,editable[i%len(editable)]);history.append(w)
    return history
def render(w):
    return f"Worksheet version {w['version']}\n"+'\n\n'.join(f"{i}. [{p['id']}] {p['question']}" for i,p in enumerate(w['problems'],1))
def prompt(history,condition,edit):
    if condition not in ('latest','naive_append'):raise ValueError(condition)
    selected=history[:edit+1] if condition=='naive_append' else [history[edit]]
    user='\n\n'.join(render(w) for w in selected)
    user+=f"\n\nUse only the newest worksheet (version {edit}). Solve problem [{history[0]['target_id']}]. Show a short calculation and finish with #### <number>."
    return [dict(role='system',content=SYSTEM),dict(role='user',content=user)]
def chat(messages):return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)+'<|im_start|>assistant\n'
def answer(text):
    m=re.search(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$',text)
    return Decimal(m.group(1).replace(',','')) if m else None
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=20);p.add_argument('--version',type=int,choices=range(11),default=0);p.add_argument('--condition',choices=['latest','naive_append'],default='latest')
    a=p.parse_args();print(chat(prompt(versions(load_problems(),a.count),a.condition,a.version)))
