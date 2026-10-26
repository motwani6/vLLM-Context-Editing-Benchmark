# GSM8K worksheet context benchmark

A controlled Qwen2.5-1.5B-Instruct/vLLM benchmark comparing the latest worksheet with naive full-worksheet appending across ten numerical edits. **PatchKV is not implemented or evaluated.** All 66 real requests completed on one Tesla T4: 11 versions ? 2 conditions ? 3 repetitions. The GPU session was stopped; shutdown evidence is preserved.

## Repository structure

```text
src/                  Application, experiment, analysis and verification Python code
data/                 GSM8K source, worksheet versions, tokenizer and pinned revisions
results/              66 GPU measurements, token counts, comparisons and latency summary
figures/              Four standalone SVG graphs
docs/                 Report, application proposal and retired workflow entry points
notebooks/            Self-contained Colab notebook (not executed during organization)
logs/                 Execution, environment, verification and shutdown evidence
  gpu_artifacts/      Unmodified original remote download, including startup failures
  originals/          Original versions of edited files and historical scripts
  archives/           Original transfer ZIPs (preserved locally, ignored by Git)
README.md
requirements-local.txt
requirements.txt      Original pinned Colab GPU dependencies
.gitignore
```

No presentation existed in the supplied files. Historical originals and remote snapshots intentionally retain their original paths and content for provenance; they are not current entry points.

## Setup and local verification

Use Python 3.10+ from the repository root. For a fresh environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
python src/verify.py
python src/validate_artifacts.py
```

No dependency installation is needed on the supplied machine: `nbformat` is already installed and the other audit dependencies are in `.local_deps`. Use:

```powershell
$env:PYTHONPATH = "$PWD\.local_deps"
py src/verify.py
py src/validate_artifacts.py
py src/math_app.py --version 10 --condition naive_append
```

Verification recounts all 22 prompts, checks worksheet invariants, validates notebook structure and Python syntax, and blocks local inference. `validate_artifacts.py` also parses all four SVGs and writes a new local verification record without replacing the historical verification record. Paths resolve from source location, so absolute script invocation also works outside the repository.

## Experiment and analysis instructions

The completed run is in [results/results.csv](results/results.csv). Inspect it without rerunning inference. Optional CPU-only regeneration of derived reports/figures uses `python src/summarize_gpu.py`; this writes derived outputs, not the measurement CSV. `python src/experiment.py --mode audit` recounts prompts and refreshes basic analysis; use the summary command afterwards to restore the full measured report.

For a separately authorized future GPU run only, open [the notebook](notebooks/math_patchkv_benchmark.ipynb) in an existing Colab GPU runtime, or copy this folder structure there, install `requirements.txt`, and run `python src/experiment.py --mode run --repetitions 3`. Back up existing outputs first: the runner starts a fresh results CSV. No GPU runs, Colab sessions, installations or uploads were performed during this organization.

The workload uses 20 GSM8K test problems, a fixed target, ten single-number distractor edits, a 32,768-token context budget and a 256-token output cap. Prefix caching is off. Recorded vLLM V1 settings enable chunked prefill despite the requested setting; see [run metadata](logs/run_metadata.json).

## Key measured results

At edit 10, append uses **16,748 vs 1,591 input tokens (10.527?)** and mean engine latency is **81.573 vs 10.130 seconds (8.053?)**. Across 33 requests per condition, total latency is 1,291.008 vs 326.718 seconds. Strict-marker correctness is 6/33 for append and 0/33 for latest-only. Output lengths differ, so these timings do not isolate prefill cost or establish a reliable math tutor.

See the [full report](docs/report.md), [proposal](docs/application_proposal.md), [latency summary](results/latency_summary.csv) and [graphs](figures/). TTFT/prefill remain blank where unsupported; memory is post-request device-wide usage rather than peak request allocation. No PatchKV speedup is claimed.

## Preservation and Git readiness

[The inventory](logs/reorganization_manifest.json) maps every original project file to its new location and SHA-256. Edited originals are preserved under `logs/originals/`; downloaded remote artifacts are unmodified under `logs/gpu_artifacts/`. Local dependencies and caches remain in place and are ignored. Data, measurements, figures and evidence logs are included by `.gitignore`; transfer ZIPs and large reproducible model artifacts are excluded. The organized project is maintained on the `main` branch at [motwani6/PatchKV-Application](https://github.com/motwani6/PatchKV-Application). Original transfer ZIPs remain local; their extracted experimental evidence is included in the repository.
