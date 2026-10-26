"""Install downloaded GPU artifacts and refresh the existing project deliverables."""
import csv,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
root=Path.cwd().resolve();archive=root/'gpu_results.zip';evidence=root/'gpu_artifacts';evidence.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        target=(evidence/item.filename).resolve()
        if not target.is_relative_to(evidence.resolve()):raise ValueError('Unsafe archive path')
    z.extractall(evidence)
remote=json.loads((evidence/'remote_verification.json').read_text());assert remote['passed']
for name in ['results.csv','context_counts.csv','preflight.json','run_metadata.json','smoke_test.json','environment_remote.txt','experiment.log','experiment_exit.json','remote_verification.json']:
    shutil.copyfile(evidence/name,root/name)
for name in ['worksheet_versions.json']:
    shutil.copyfile(evidence/'data'/name,root/'data'/name)
assert hashlib.sha256((root/'results.csv').read_bytes()).hexdigest()==remote['results_sha256']
assert 'No active sessions' in Path('colab_sessions_after_gpu.txt').read_text()
from summarize_gpu import main
main()
# Refresh only the embedded source cell; the notebook remains unexecuted as a notebook.
nb=json.loads(Path('math_patchkv_benchmark.ipynb').read_text())
sources={n:Path(n).read_text(encoding='utf-8-sig') for n in ['math_app.py','experiment.py','analyze.py','requirements.txt']}
for cell in nb['cells']:
    content=''.join(cell['source'])
    if cell['cell_type']=='code' and '\nsources=' in content:
        cell['source']=('from pathlib import Path\nimport json\nsources='+repr(sources)+'\nfor name,text in sources.items():\n    Path(name).write_text(text,encoding="utf-8")\n').splitlines(True)
    if cell['cell_type']=='markdown':
        content=content.replace('GPU execution has not been performed.','The experiment was executed through the Colab CLI; see results.csv and report.md. This notebook itself has not been executed.')
        cell['source']=content.splitlines(True)
Path('math_patchkv_benchmark.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
status=dict(date='2026-10-24',colab_cli_version='0.7.5',authorized_t4_sessions_created=1,initial_compute_units=0.0,billing_changes=False,purchases=False,inference_executed=True,measurement_rows=66,repetitions=3,shutdown='CLI stop succeeded; server reports no active sessions',results_sha256=remote['results_sha256'])
Path('execution_status.json').write_text(json.dumps(status,indent=2))
p=Path('README.md');text=p.read_text()
text=text.replace('vLLM dependency installation and GPU compatibility have not been validated without a live runtime.','The experiment completed on a Tesla T4. See run_metadata.json, environment_remote.txt and gpu_verification.json. The smoke validator accepts the exact correct 2+2=4 equation; measured correctness scoring remains strict.')
text=text.replace('No runtime was available to stop during this assignment.','The single authorized T4 runtime was stopped after artifact retrieval; colab_sessions_after_gpu.txt confirms no active sessions.')
text+='\n## Completed GPU execution\nAll 66 requests completed on one Tesla T4 session using the existing runner, three repetitions per condition and version. gpu_artifacts/ preserves downloaded source, logs, environment and original remote output. The local report and figures are regenerated from that real CSV. GPU shutdown is confirmed in gpu_shutdown.txt and colab_sessions_after_gpu.txt.\n\nT4 compatibility uses FlexAttention and caps KV-cache blocks to the configured context window. vLLM V1 enables chunked prefill despite the requested false setting; effective_engine_settings records this. Prefix caching is disabled. Startup attempts and their failures were excluded from measurements.\n'
p.write_text(text,encoding='utf-8')
summary=list(csv.DictReader(open('latency_summary.csv',encoding='utf-8')))[-1]
Path('application_proposal.md').write_text(f"I completed a GSM8K worksheet benchmark on a Tesla T4 using Qwen2.5-1.5B-Instruct in FP16 through vLLM. It compares latest-only worksheets with naive appending across ten single-number updates, with stable IDs and an unchanged target. All 66 requests completed (three repetitions per condition and version), with prefix caching disabled. At edit 10, naive append uses 16,748 versus 1,591 input tokens (10.53x); mean engine latency is {float(summary['append_mean_s']):.3f}s versus {float(summary['latest_mean_s']):.3f}s ({float(summary['ratio_of_means']):.3f}x). Output lengths differ, so latency includes a decoding confound. Raw CSVs, plots, dependency versions, correctness results and the report are saved. The single GPU session was stopped. PatchKV is not implemented or evaluated, and no PatchKV speedup is claimed.\n",encoding='utf-8')
from verify import main as verify
verify()
verification=json.loads(Path('gpu_verification.json').read_text());verification['session_stopped']=True
Path('verification.json').write_text(json.dumps(verification,indent=2))
print('FINALIZED: real GPU results, plots, report, notebook, proposal, verification and shutdown evidence.')
