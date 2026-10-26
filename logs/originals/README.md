# GSM8K worksheet context benchmark

Compares the latest worksheet against naive full-worksheet appending. PatchKV is not implemented. See report.md for actual measurements and execution status. No inference runs locally.

## Local prompt audit
Use Python 3.10+ and `pip install pyarrow tokenizers jinja2`. Run:

```text
python experiment.py --mode audit
python math_app.py --version 10 --condition naive_append
python verify.py
```

On this machine the audit dependencies are in .local_deps. In PowerShell set `$env:PYTHONPATH = "$PWD\.local_deps"` and use `py` in place of `python`. The local Python is 3.15; Colab must use its supplied Python.

## Existing Colab runtime only
The installed CLI is outside PATH at `C:\Users\drish\AppData\Local\Python\pythoncore-3.15-64\Scripts\colab.exe`. Invoke it in PowerShell with `&` followed by its quoted full path.

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

Save the freeze output to environment_remote.txt. Restart the Colab kernel if installation replaced already imported torch packages. The runner verifies CUDA and Colab, performs a real 2+2 smoke inference, warms up short and long requests, and checks every context budget before the full experiment. Inspect smoke_test.json. The experiment completed on a Tesla T4. See run_metadata.json, environment_remote.txt and gpu_verification.json. The smoke validator accepts the exact correct 2+2=4 equation; measured correctness scoring remains strict.

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

Download results.csv, context_counts.csv, comparison.csv, report.md, plots, metadata, environment_remote.txt and data provenance before shutdown. The notebook creates a downloadable zip then supplies a runtime-release cell. With the CLI download artifacts first, then run `colab stop -s NAME` and verify `colab sessions`. The single authorized T4 runtime was stopped after artifact retrieval; colab_sessions_after_gpu.txt confirms no active sessions.

## Completed GPU execution
All 66 requests completed on one Tesla T4 session using the existing runner, three repetitions per condition and version. gpu_artifacts/ preserves downloaded source, logs, environment and original remote output. The local report and figures are regenerated from that real CSV. GPU shutdown is confirmed in gpu_shutdown.txt and colab_sessions_after_gpu.txt.

T4 compatibility uses FlexAttention and caps KV-cache blocks to the configured context window. vLLM V1 enables chunked prefill despite the requested false setting; effective_engine_settings records this. Prefix caching is disabled. Startup attempts and their failures were excluded from measurements.

Regenerate the full measured report and descriptive latency summary with `python summarize_gpu.py`; this also requires the complete 66-row CSV and recorded metadata. `python experiment.py --mode analyze` regenerates the basic report and plots.
