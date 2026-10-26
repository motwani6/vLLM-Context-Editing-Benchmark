"""Generate SVG plots, comparisons and report from actual artifacts."""
import csv,json,statistics
from pathlib import Path
from html import escape

def read(path):
    if not Path(path).exists():return []
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def plot(path,title,series,ylabel,missing=False):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="500" viewBox="0 0 880 500">','<rect width="880" height="500" fill="white"/>',f'<text x="440" y="30" text-anchor="middle" font-size="20">{escape(title)}</text>','<path d="M90 65 V410 H830" fill="none" stroke="black"/>','<text x="440" y="475" text-anchor="middle">Edit number</text>',f'<text transform="translate(22 250) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']
    if missing:svg.append('<text x="440" y="240" text-anchor="middle" fill="#555">No GPU inference executed: latency data unavailable</text>')
    else:
        top=max((y for points in series.values() for _,y in points),default=1)*1.08 or 1
        for i in range(6):
            y=410-i*345/5;svg.append(f'<text x="82" y="{y+4}" text-anchor="end">{top*i/5:.1f}</text>')
        for i in range(11):svg.append(f'<text x="{90+i*74}" y="435" text-anchor="middle">{i}</text>')
        for index,(name,points) in enumerate(series.items()):
            color=['#2471a3','#c0392b'][index];coords=' '.join(f'{90+x*74:.2f},{410-y/top*345:.2f}' for x,y in points)
            svg+=[f'<polyline points="{coords}" stroke="{color}" stroke-width="2.5" fill="none"/>',f'<text x="{110+index*280}" y="60" fill="{color}">{name}</text>']
    svg.append('</svg>');Path(path).write_text('\n'.join(svg),encoding='utf-8')
def analyze():
    Path('plots').mkdir(exist_ok=True);counts=read('context_counts.csv');results=read('results.csv')
    if not counts:return
    conditions=['latest','naive_append']
    series={c:[(int(r['edit_number']),float(r['input_tokens'])) for r in counts if r['condition']==c] for c in conditions}
    plot('plots/context_tokens.svg','Measured prompt tokens by edit',series,'Input tokens')
    growth={c:[(int(r['edit_number']),float(r['context_growth_ratio'])) for r in counts if r['condition']==c] for c in conditions}
    plot('plots/token_growth.svg','Token growth relative to version 0',growth,'Context growth ratio')
    latency={c:[] for c in conditions};comparisons=[]
    for e in range(11):
        a=next(r for r in counts if r['condition']=='latest' and int(r['edit_number'])==e);b=next(r for r in counts if r['condition']=='naive_append' and int(r['edit_number'])==e)
        means={}
        for c in conditions:
            values=[float(r['latency_s']) for r in results if r['condition']==c and int(r['edit_number'])==e]
            means[c]=statistics.mean(values) if values else None
            if values:latency[c].append((e,means[c]))
        ma,mb=means['latest'],means['naive_append']
        comparisons.append(dict(edit_number=e,latest_tokens=int(a['input_tokens']),append_tokens=int(b['input_tokens']),extra_tokens=int(b['input_tokens'])-int(a['input_tokens']),token_ratio=float(b['input_tokens'])/float(a['input_tokens']),latest_latency_s=ma,append_latency_s=mb,latency_difference_s=mb-ma if ma is not None and mb is not None else None,latency_ratio=mb/ma if ma and mb is not None else None))
    plot('plots/inference_latency.svg','Mean end-to-end inference latency',latency,'Seconds',not results)
    plot('plots/latency_comparison.svg','Latest worksheet versus naive append',latency,'Mean seconds',not results)
    with open('comparison.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    pre=json.loads(Path('preflight.json').read_text());last=comparisons[-1]
    at=sum(r['latest_tokens'] for r in comparisons);bt=sum(r['append_tokens'] for r in comparisons)
    table='\n'.join(f"| {r['edit_number']} | {r['latest_tokens']} | {r['append_tokens']} | {r['extra_tokens']} | {r['token_ratio']:.3f} |" for r in comparisons)
    status=f'Actual GPU measurements: {len(results)} requests. Inspect run_metadata.json for completion status.' if results else 'GPU inference was not executed. On 2026-10-24 the authenticated Colab CLI 0.7.5 reported no active sessions; usage showed 0.00 compute units and zero active assignments. No new runtime was created. No session was available to stop. results.csv contains a header only. Smoke inference, warm-up and all 66 planned vLLM requests remain unexecuted.'
    report=f'''# Math worksheet context-appending benchmark

## Execution status
{status}

The real CPU tokenizer measurements in context_counts.csv are prompt-size measurements. Latency figures explicitly mark missing data. No latency, output length, correctness, TTFT, prefill or GPU-memory values are fabricated.

## Application and data
The application loads real [GSM8K](https://huggingface.co/datasets/openai/gsm8k) main/test questions at immutable revision `{pre['dataset_revision']}`. It uses the first {pre['worksheet_size']} test problems with stable row-based IDs. GSM8K supplies short numerical reasoning problems and reference final answers. The target `{pre['target_id']}` stays unchanged. Solutions are never included in prompts; edited distractor labels are invalidated. This is a controlled workload, not a representative accuracy evaluation.

## Model and methodology
The planned inference uses vLLM and [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `{pre['model_revision']}`, in FP16. The context budget is {pre['max_model_len']} including an output reservation of {pre['max_output_tokens']} tokens. Every prompt is checked before inference; none is truncated. Local auditing uses the real Qwen tokenizer and validates the system/user rendering against its official chat template.

Eleven worksheet versions comprise the original and ten sequential edits. Each changes exactly one numerical span in a non-target problem. Latest-only includes one worksheet; naive append includes every version through the current edit. Both share system instructions, newest-version question and target. Model responses are not appended.

The remote runner first checks a small real inference, then warms both shortest and longest prompts. The planned experiment has three repetitions per condition (66 serial requests), seeded randomized edit and paired condition order, greedy generation, seed 42, top_p=1, fixed output cap 256. Prefix caching and chunked prefill are explicitly disabled. Eager execution avoids graph compilation, with identical configuration across conditions.

Latency brackets synchronous vLLM generation with CUDA synchronization. It includes prefill and decoding, excluding model loading, tokenizer work, warm-up, network transport and file writes. It is end-to-end engine request latency, not browser-to-server latency. TTFT uses engine metrics only if populated; isolated prefill remains null. GPU memory is a device-wide nvidia-smi snapshot after each request, including reserved KV cache, not peak per-request usage.

## Actual token measurements
| Edit | Latest tokens | Append tokens | Extra tokens | Append/latest |
| --- | ---: | ---: | ---: | ---: |
{table}

At edit 10, naive append has **{last['append_tokens']:,}** input tokens versus **{last['latest_tokens']:,}**, a **{last['token_ratio']:.3f}x** ratio and **{last['extra_tokens']:,}** extra tokens. Across one request for each version, totals are {at:,} latest tokens and {bt:,} append tokens: {bt-at:,} extra, or {bt/at:.3f}x. Three repetitions would multiply these totals by three.

Full-worksheet duplication gives approximately linear context growth with edit number, and approximately quadratic cumulative input tokens across updates when worksheet size is stable. This is an algorithmic interpretation of token counts, not a timing measurement.

![Context tokens](plots/context_tokens.svg)
![Token growth](plots/token_growth.svg)
![Inference latency](plots/inference_latency.svg)
![Latency comparison](plots/latency_comparison.svg)

## Inference comparison
'''
    if results:
        for c in conditions:
            rows=[r for r in results if r['condition']==c]
            report+=f"\n{c}: {len(rows)} requests; mean latency {statistics.mean(float(r['latency_s']) for r in rows):.4f}s; correctness {sum(r['correct']=='True' for r in rows)}/{len(rows)}.\n"
        report+='\nPer-edit latency differences and ratios are in comparison.csv. Check paired repetitions and completion status before interpreting partial runs.\n'
    else:report+='No measured latency differences or speedups can be calculated. GPU inference remains pending because no existing runtime was available.\n'
    report+='''
## Limitations and confounders
A small fixed worksheet and single unchanged target do not establish general math accuracy. Edits may make distractors inconsistent, and only the target is scored. Greedy decoding is not guaranteed identical across hardware or software versions. Variable output lengths and output-cap termination affect latency; examine output tokens and finish reasons. Strict final-marker scoring can reject another correct answer format. GPU clocks, shared load, thermal state, warm-up, reserved cache and request order affect timings. Three repetitions provide descriptive comparisons, not strong statistical inference. Prefix caching is deliberately disabled and is not the default in every deployed service. Device memory snapshots do not measure peak request allocations. Save environment_remote.txt for the eventual remote run.

## Future PatchKV evaluation
Stable IDs and recorded edit spans provide a future PatchKV integration point. A follow-up must implement PatchKV, verify answer equivalence, and compare update cost and cache reuse against both full recomputation and automatic prefix caching under identical hardware and workloads. PatchKV is not implemented or evaluated here; no PatchKV speedup is claimed.

## Sources
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [vLLM engine arguments](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args/)
- [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
'''
    Path('report.md').write_text(report,encoding='utf-8')
    print(f'Analysis: {len(counts)} real token counts, {len(results)} GPU measurements.')
if __name__=='__main__':analyze()
