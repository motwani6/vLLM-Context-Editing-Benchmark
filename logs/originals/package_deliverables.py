from pathlib import Path
import csv,json
files={
'requirements.txt':'''# Install only in the remote Colab Linux GPU runtime, not Windows.
vllm==0.11.0
transformers==4.57.1
pyarrow>=18,<27
jinja2>=3.1,<4
''',
'README.md':'''# GSM8K worksheet context benchmark

Compares the latest worksheet against naive full-worksheet appending. PatchKV is not implemented. See report.md for actual measurements and execution status. No inference runs locally.

## Local prompt audit
Use Python 3.10+ and `pip install pyarrow tokenizers jinja2`. Run:

```text
python experiment.py --mode audit
python math_app.py --version 10 --condition naive_append
python verify.py
```

On this machine the audit dependencies are in .local_deps. In PowerShell set `$env:PYTHONPATH = "$PWD\\.local_deps"` and use `py` in place of `python`. The local Python is 3.15; Colab must use its supplied Python.

## Existing Colab runtime only
The installed CLI is outside PATH at `C:\\Users\\drish\\AppData\\Local\\Python\\pythoncore-3.15-64\\Scripts\\colab.exe`. Invoke it in PowerShell with `&` followed by its quoted full path.

```text
colab --help
colab sessions
colab usage
colab exec --help
colab upload --help
colab download --help
colab stop --help
```

Select the existing T4 session explicitly with `-s NAME`. Do not create a new session, use `colab run`, buy units or provision extra resources without permission. If sessions is empty, no GPU experiment can execute under the current authorization. The CLI upstream documents Windows as unsupported, although the installed 0.7.5 successfully queried sessions and usage here.

Open math_patchkv_benchmark.ipynb on an existing Colab GPU and run the cells. It embeds the application source and a compact immutable data snapshot. Alternatively upload math_app.py, experiment.py, analyze.py and requirements.txt to one remote directory, install requirements there, and run:

```text
python experiment.py --mode run --repetitions 3
python -m pip freeze
```

Save the freeze output to environment_remote.txt. Restart the Colab kernel if installation replaced already imported torch packages. The runner verifies CUDA and Colab, performs a real 2+2 smoke inference, warms up short and long requests, and checks every context budget before the full experiment. Inspect smoke_test.json. vLLM dependency installation and GPU compatibility have not been validated without a live runtime.

Default settings: first 20 GSM8K test questions, unchanged first question as target, ten single-number distractor edits, 32768 total context tokens, output cap 256, greedy generation, three repetitions, prefix caching off, chunked prefill off, eager mode and serial requests. --count supports 20 through 30 and reduces count to fit if necessary, never below 20. If the minimum size does not fit, it fails instead of truncating. --repetitions 1 can reduce GPU time; use the same repetition count for both conditions.

The run yields 66 measurement rows at three repetitions. Rows flush after every request to preserve partial work; run_metadata.json distinguishes running and complete. Before retrying a partial run, copy the old results because the runner starts a fresh CSV. Analysis supports partial runs; missing latency values stay blank.

## Artifacts and definitions
- math_app.py: real data loading, stable IDs, protected target, numerical editing and prompt rendering.
- experiment.py: audit and Colab-only inference runner; analyze.py: comparisons, plots and report.
- context_counts.csv: real tokenizer counts, separate from actual GPU inference results.csv.
- comparison.csv: append minus latest token and latency differences and append/latest ratios.
- data/: dataset revision/hash, original test data, tokenizer and 11 worksheet versions.
- preflight.json, run_metadata.json, smoke_test.json: budgets and actual remote execution metadata.
- plots/: standalone SVG figures. Latency figures explicitly label absent data.

Context growth is input tokens divided by the same condition's version-0 input tokens. Latency includes engine prefill and decode, excluding initialization, tokenization, network and file writes. TTFT is recorded only from engine metrics if supported; isolated prefill remains blank. Memory is post-request device-wide nvidia-smi usage, not peak request memory. Correctness requires the final `#### <number>` to match the unchanged target reference. Original solutions never enter prompts; edited distractor answers are invalidated.

Download results.csv, context_counts.csv, comparison.csv, report.md, plots, metadata, environment_remote.txt and data provenance before shutdown. The notebook creates a downloadable zip then supplies a runtime-release cell. With the CLI download artifacts first, then run `colab stop -s NAME` and verify `colab sessions`. No runtime was available to stop during this assignment.
''',
'application_proposal.md':'''I built a GSM8K worksheet application as a workload for future PatchKV benchmarking. Stable problem IDs and ten single-number distractor updates let us compare the latest worksheet with repeated full-worksheet appending while keeping one target fixed. The benchmark is configured for Qwen2.5-1.5B-Instruct in FP16 through vLLM, with prefix caching disabled. Real prompt-token measurements, plots and a reproducible Colab notebook are included. GPU timing remains pending: the authenticated CLI found no active runtime and a zero compute-unit balance, so no additional session was created. This study measures naive appending; PatchKV is not implemented and no PatchKV speedup is claimed.
''',
'.gitignore':'.local_deps/\n__pycache__/\n*.zip\n',
'execution_status.json':json.dumps(dict(date='2026-10-24',colab_cli_version='0.7.5',sessions_output='[colab] No active sessions found on server.',compute_units=0.0,usage_rate_per_hour=0.0,active_assignments=0,new_session_created=False,inference_executed=False,shutdown='No existing session was available to reuse or stop.'),indent=2)
}
for name,content in files.items():Path(name).write_text(content,encoding='utf-8')
from experiment import FIELDS
if not Path('results.csv').exists():
    with open('results.csv','w',newline='',encoding='utf-8') as f:csv.writer(f).writerow(FIELDS)
def code(s):return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True))
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
sources={name:Path(name).read_text(encoding='utf-8-sig') for name in ['math_app.py','experiment.py','analyze.py','requirements.txt']}
cells=[md('# GSM8K context-appending benchmark\nUse an existing GPU runtime only. No PatchKV implementation. GPU execution has not been performed. Download artifacts before releasing the runtime.\n'),code("import os, subprocess, sys\nfrom pathlib import Path\nassert os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION'), 'Google Colab required'\nsubprocess.run(['nvidia-smi'], check=True)\nwork=Path('/content/math_patchkv_benchmark')\nwork.mkdir(exist_ok=True)\nos.chdir(work)\n"),code('from pathlib import Path\nimport json\nsources='+repr(sources)+'\nfor name,text in sources.items():\n    Path(name).write_text(text,encoding="utf-8")\n'),code("subprocess.run([sys.executable,'-m','pip','install','-r','requirements.txt'],check=True)\n"),md('Restart the kernel if installation replaced imported torch libraries, then rerun directory setup and source cells before continuing.\n'),code("import os,subprocess,sys\nos.chdir('/content/math_patchkv_benchmark')\nsubprocess.run([sys.executable,'experiment.py','--mode','run','--repetitions','3'],check=True)\nwith open('environment_remote.txt','w') as f:\n    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)\n"),code("from pathlib import Path\nfrom IPython.display import SVG, display\nprint(Path('report.md').read_text())\nfor p in sorted(Path('plots').glob('*.svg')):\n    display(SVG(filename=str(p)))\n"),code("import shutil\nfrom google.colab import files\narchive=shutil.make_archive('/content/math_patchkv_results','zip','/content/math_patchkv_benchmark')\nfiles.download(archive)\n"),md('After confirming the artifact download finished, run this cell to release the runtime. With CLI use `colab stop -s NAME` after downloads.\n'),code('from google.colab import runtime\nruntime.unassign()\n')]
# When the audit has completed, embed the compact worksheet source data and model revision.
p=Path('data/gsm8k_test.json')
if p.exists() and Path('data/model_revision.json').exists():
    data=json.loads(p.read_text());data['problems']=data['problems'][:30]
    snapshots={'data/gsm8k_test.json':data,'data/model_revision.json':json.loads(Path('data/model_revision.json').read_text())}
    cells.insert(3,code('from pathlib import Path\nimport json\nsnapshots='+repr(snapshots)+'\nPath("data").mkdir(exist_ok=True)\nfor name,value in snapshots.items():\n    Path(name).write_text(json.dumps(value,indent=2),encoding="utf-8")\n'))
for i,c in enumerate(cells):c['id']=f'cell-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),accelerator='GPU'),cells=cells)
Path('math_patchkv_benchmark.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
print('Documentation and notebook generated.')
