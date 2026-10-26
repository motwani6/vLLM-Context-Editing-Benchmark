"""Token audit and Colab-only vLLM runner; no fabricated measurements."""
import argparse,csv,hashlib,importlib.metadata,json,os,platform,random,subprocess,time
from pathlib import Path
from project_paths import ROOT
from math_app import MODEL,answer,fetch,load_problems,prompt,chat,versions
FIELDS=['repetition','condition','edit_number','input_tokens','context_growth_ratio','latency_s','output_tokens','ttft_s','prefill_latency_s','correct','parsed_answer','gpu_memory_used_mib','finish_reason','output_text','prompt_sha256']
def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def prepare(count,max_len,max_output,remote=False):
    data=load_problems();d=Path(str(ROOT / 'data'));d.mkdir(exist_ok=True);revpath=d/'model_revision.json'
    if revpath.exists():meta=json.loads(revpath.read_text())
    else:
        meta=dict(model=MODEL,revision=fetch(f'https://huggingface.co/api/models/{MODEL}')['sha']);save(revpath,meta)
    revision=meta['revision']
    if remote:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
        encode=lambda messages:tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    else:
        from tokenizers import Tokenizer
        from jinja2 import Environment
        import urllib.request
        p=d/'tokenizer.json'
        if not p.exists():
            with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{revision}/tokenizer.json',timeout=120) as r:p.write_bytes(r.read())
        cp=d/'tokenizer_config.json'
        if not cp.exists():save(cp,fetch(f'https://huggingface.co/{MODEL}/resolve/{revision}/tokenizer_config.json'))
        template=Environment().from_string(json.loads(cp.read_text())['chat_template'])
        messages=prompt(versions(data,count),'latest',0)
        assert template.render(messages=messages,add_generation_prompt=True)==chat(messages),'Chat template mismatch'
        tok=Tokenizer.from_file(str(p));encode=lambda m:tok.encode(chat(m),add_special_tokens=False).ids
    for n in range(count,19,-1):
        history=versions(data,n)
        prompts={(c,e):encode(prompt(history,c,e)) for c in ('latest','naive_append') for e in range(11)}
        if max(map(len,prompts.values()))+max_output<=max_len:break
    else:raise ValueError('20 problems do not fit. No truncation performed; increase context budget.')
    save(d/'worksheet_versions.json',history)
    with open(str(ROOT / 'results/context_counts.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['condition','edit_number','input_tokens','context_growth_ratio']);w.writeheader()
        for c in ('latest','naive_append'):
            for e in range(11):w.writerow(dict(condition=c,edit_number=e,input_tokens=len(prompts[c,e]),context_growth_ratio=len(prompts[c,e])/len(prompts[c,0])))
    save(str(ROOT / 'logs/preflight.json'),dict(dataset_revision=data['revision'],model_revision=revision,worksheet_size=n,target_id=history[0]['target_id'],max_model_len=max_len,max_output_tokens=max_output,maximum_input_tokens=max(map(len,prompts.values())),tokenizer_backend='transformers' if remote else 'tokenizers with validated Qwen chat template'))
    return history,prompts,revision

def memory():
    try:return float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    except (OSError,subprocess.SubprocessError,ValueError):return None

def run(a):
    if platform.system()!='Linux' or not (os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION')):raise RuntimeError('Inference is restricted to Google Colab. Use the notebook.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU available')
    # T4: cap cache to the configured window to bound FlexAttention's dense mask.
    if torch.cuda.get_device_capability(0)[0] < 8:
        os.environ.setdefault('VLLM_ATTENTION_BACKEND','FLEX_ATTENTION')
    history,prompts,revision=prepare(a.count,a.max_model_len,a.max_output,True)
    from vllm import LLM,SamplingParams
    config=dict(model=MODEL,revision=revision,tokenizer_revision=revision,dtype='half',max_model_len=a.max_model_len,enable_prefix_caching=False,enable_chunked_prefill=False,max_num_batched_tokens=a.max_model_len,max_num_seqs=1,gpu_memory_utilization=0.85,enforce_eager=True,seed=42)
    if torch.cuda.get_device_capability(0)[0] < 8:
        config['block_size']=16
        config['num_gpu_blocks_override']=(a.max_model_len+15)//16
    llm=LLM(**config);sampling=SamplingParams(temperature=0,top_p=1,max_tokens=a.max_output,seed=42)
    def generate(ids):
        torch.cuda.synchronize();start=time.perf_counter()
        result=llm.generate([{'prompt_token_ids':ids}],sampling,use_tqdm=False)[0]
        torch.cuda.synchronize();return result,time.perf_counter()-start
    smoke_ids=llm.get_tokenizer().apply_chat_template([dict(role='user',content='What is 2 + 2? End with #### 4.')],tokenize=True,add_generation_prompt=True)
    smoke,seconds=generate(smoke_ids)
    import re
    smoke_text=smoke.outputs[0].text.strip()
    smoke_ok=answer(smoke_text)==4 or re.fullmatch(r'2\s*\+\s*2\s*=\s*4\.?',smoke_text) is not None
    save(str(ROOT / 'logs/smoke_test.json'),dict(text=smoke_text,latency_s=seconds,passed=smoke_ok,marker_compliant=answer(smoke_text)==4,validation='Final marker or exact correct 2+2=4 equation; benchmark scoring remains strict'))
    if not smoke_ok:raise RuntimeError('Smoke inference failed; inspect smoke_test.json')
    generate(prompts['latest',0]);generate(prompts['naive_append',10])
    truth=answer(history[0]['problems'][0]['answer']);assert truth is not None
    meta=dict(status='running',inference_executed=True,configuration=config,repetitions=a.repetitions,gpu=torch.cuda.get_device_name(0),sampling=dict(temperature=0,top_p=1,max_tokens=a.max_output,seed=42),ttft='Engine metrics only when populated; otherwise null',prefill_latency='Not directly measurable; null',gpu_memory='Device-wide post-request nvidia-smi snapshot, includes reserved KV cache',packages={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','tokenizers','pyarrow']})
    engine_config=getattr(llm.llm_engine,'vllm_config',None)
    scheduler=getattr(engine_config,'scheduler_config',None)
    cache=getattr(engine_config,'cache_config',None)
    meta['effective_engine_settings']=dict(attention_backend=os.environ.get('VLLM_ATTENTION_BACKEND','auto'),enable_chunked_prefill=getattr(scheduler,'enable_chunked_prefill',None),enable_prefix_caching=getattr(cache,'enable_prefix_caching',None))
    save(str(ROOT / 'logs/run_metadata.json'),meta);rng=random.Random(42)
    with open(str(ROOT / 'results/results.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();f.flush()
        for rep in range(1,a.repetitions+1):
            edits=list(range(11));rng.shuffle(edits)
            for e in edits:
                conditions=['latest','naive_append'];rng.shuffle(conditions)
                for c in conditions:
                    ids=prompts[c,e];out,seconds=generate(ids);completion=out.outputs[0];pred=answer(completion.text)
                    metrics=getattr(out,'metrics',None);first=getattr(metrics,'first_token_time',None);arrival=getattr(metrics,'arrival_time',None)
                    w.writerow(dict(repetition=rep,condition=c,edit_number=e,input_tokens=len(ids),context_growth_ratio=len(ids)/len(prompts[c,0]),latency_s=seconds,output_tokens=len(completion.token_ids),ttft_s=first-arrival if first is not None and arrival is not None else None,prefill_latency_s=None,correct=pred==truth,parsed_answer=str(pred) if pred is not None else '',gpu_memory_used_mib=memory(),finish_reason=completion.finish_reason,output_text=completion.text,prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()));f.flush()
                    print(f'rep={rep} edit={e} {c}: tokens={len(ids)}, seconds={seconds:.3f}, correct={pred==truth}',flush=True)
    meta['status']='complete';save(str(ROOT / 'logs/run_metadata.json'),meta)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['audit','run','analyze'],default='audit');p.add_argument('--count',type=int,default=20);p.add_argument('--max-model-len',type=int,default=32768);p.add_argument('--max-output',type=int,default=256);p.add_argument('--repetitions',type=int,choices=[1,2,3],default=3);a=p.parse_args()
    if a.mode=='run':run(a)
    elif a.mode=='audit':prepare(a.count,a.max_model_len,a.max_output)
    from analyze import analyze
    analyze()
if __name__=='__main__':main()
