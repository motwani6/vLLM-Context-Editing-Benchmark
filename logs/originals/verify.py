"""Validate workload invariants, real counts, deliverables and notebook."""
import csv,json
from pathlib import Path
from math_app import answer,load_problems,versions,prompt,edit_number,chat

def main():
    data=load_problems();history=versions(data);target=history[0]['problems'][0]
    assert len(history)==11 and len(history[0]['problems'])==20
    ids=[p['id'] for p in history[0]['problems']]
    for old,new in zip(history,history[1:]):
        assert [p['id'] for p in new['problems']]==ids
        assert new['problems'][0]==target
        changed=[(a,b) for a,b in zip(old['problems'],new['problems']) if a['question']!=b['question']]
        assert len(changed)==1
        a,b=changed[0];e=new['edit']
        assert a['id']==e['problem_id']!=new['target_id']
        assert a['question'][:e['start']]+e['after']+a['question'][e['end']:]==b['question']
        assert b['answer'] is None
    try:edit_number(history[0],target['id'])
    except ValueError:pass
    else:raise AssertionError('Target editing was accepted')
    assert prompt(history,'latest',0)==prompt(history,'naive_append',0)
    assert answer('work 123\n#### 1,234')==1234 and answer('answer 1234') is None
    assert answer(target['answer']) is not None
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file('data/tokenizer.json')
    counts=list(csv.DictReader(open('context_counts.csv',encoding='utf-8')));assert len(counts)==22
    pre=json.loads(Path('preflight.json').read_text())
    for r in counts:
        c,e=r['condition'],int(r['edit_number'])
        n=len(tok.encode(chat(prompt(history,c,e)),add_special_tokens=False).ids)
        base=len(tok.encode(chat(prompt(history,c,0)),add_special_tokens=False).ids)
        assert n==int(r['input_tokens'])
        assert abs(float(r['context_growth_ratio'])-n/base)<1e-10
        assert n+pre['max_output_tokens']<=pre['max_model_len']
    results=list(csv.DictReader(open('results.csv',encoding='utf-8')))
    if results:
        keys=[(r['repetition'],r['condition'],r['edit_number']) for r in results]
        assert len(keys)==len(set(keys)) and all(float(r['latency_s'])>0 for r in results)
    import nbformat
    nb=nbformat.read('math_patchkv_benchmark.ipynb',as_version=4);nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'notebook-cell','exec')
    for path in Path('.').glob('*.py'):compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    for name in ['math_app.py','experiment.py','requirements.txt','README.md','results.csv','report.md','application_proposal.md','math_patchkv_benchmark.ipynb']:
        assert Path(name).is_file()
    assert len(list(Path('plots').glob('*.svg')))==4
    from experiment import run
    from types import SimpleNamespace
    try:run(SimpleNamespace())
    except RuntimeError as e:assert 'Colab' in str(e)
    else:raise AssertionError('Local inference guard failed')
    print(f'PASS: 11 versions, protected target, ten single-span edits, 22 independently recounted prompts, {len(results)} GPU rows, valid notebook, local inference blocked.')
if __name__=='__main__':main()
