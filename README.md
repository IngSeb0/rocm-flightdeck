# ROCm FlightDeck

ROCm FlightDeck is an open-source AI performance portability agent for AMD GPUs.

It analyzes CUDA/NVIDIA-first vLLM and PyTorch inference repositories, detects ROCm compatibility blockers, calculates a deterministic ROCm Readiness Score, generates reviewable ROCm migration artifacts, creates a migrated repository copy, re-scores the result, and packages a downloadable migration bundle.

ROCm FlightDeck is not a generic CUDA-to-ROCm converter. It is an AI performance portability lab that turns AMD adoption from a risky migration project into a measurable, reviewable engineering workflow.

## Why This Matters

LLM inference repositories are often NVIDIA-first by default: hardcoded `.cuda()` calls, `nvidia/cuda` Docker images, CUDA-only dependencies, and missing ROCm benchmark methodology. Teams evaluating AMD MI300X need a repeatable way to understand migration effort before spending engineering time.

ROCm FlightDeck provides that first pass as deterministic static analysis. It does not run untrusted code, does not require internet access, and does not invent benchmark results.

## What The MVP Does

- Scans local repositories using safe static file reads.
- Detects CUDA/NVIDIA assumptions in Python, Dockerfiles, and dependency files.
- Scores readiness from 0 to 100 using a deterministic rubric.
- Generates an ordered migration plan.
- Generates reviewable artifacts:
  - `Dockerfile.rocm`
  - `requirements-rocm.txt`
  - `serve_vllm_rocm.py`
  - `benchmark_rocm.py`
  - `README_AMD_MIGRATION.md`
  - `flightdeck.patch`
- Creates a migrated copy in `outputs/migrated_repos/<repo>_rocm/`.
- Re-runs scanner, detectors, and scoring on the migrated copy.
- Generates `migration_report.md`.
- Builds `outputs/bundles/<repo>_migration_bundle.zip`.

## What It Does Not Do Yet

- Does not execute untrusted repository code.
- Does not run real benchmarks locally.
- Does not fabricate latency, throughput, VRAM, or cold-start numbers.
- Does not require AMD Cloud credentials.
- Does not call Qwen or external LLM APIs yet.
- Does not claim generated patches are production-ready.

Reports and benchmark templates clearly state: `Benchmarks not executed in this MVP.`

## Demo Flow

1. Start the app.
2. Select one of the controlled demo repositories.
3. Run analysis.
4. Review the before ROCm Readiness Score.
5. Inspect critical blockers and warnings.
6. Review generated patch and artifacts.
7. Inspect the migrated repository path.
8. Compare the after ROCm Readiness Score.
9. Download the migration bundle.
10. Use `benchmark_rocm.py` later on AMD Developer Cloud / MI300X to collect real metrics.

## Before / After Example

The default demo is `demo_repos/nvidia_locked_vllm_demo`.

Expected behavior:

- Before score: low, around `20/100`.
- After score: `85+/100` after safe transformations, legacy preservation, ROCm-first defaults, and generated ROCm artifacts.
- Bundle: generated at `outputs/bundles/nvidia_locked_vllm_demo_migration_bundle.zip`.
- Benchmark status: not executed locally.

## Demo Repositories

| Demo Repo | Expected Readiness | Purpose |
|-----------|--------------------|---------|
| `nvidia_locked_vllm_demo` | Low, around 20/100 | CUDA-locked baseline with NVIDIA Docker and incompatible dependencies |
| `partially_portable_pytorch_demo` | Medium, around 55-75/100 | Uses device abstraction but still lacks ROCm artifacts and has one risky dependency |
| `rocm_ready_vllm_demo` | High, around 85-100/100 | Includes ROCm Dockerfile, ROCm requirements, benchmark methodology, and migration notes |

## Generated Artifacts

Artifacts are written to:

```text
outputs/patches/<repo_name>/
```

Migration copies are written to:

```text
outputs/migrated_repos/<repo_name>_rocm/
```

Reports are written to:

```text
outputs/reports/<repo_name>/migration_report.md
```

Bundles are written to:

```text
outputs/bundles/<repo_name>_migration_bundle.zip
```

The bundle includes:

- `Dockerfile.rocm`
- `requirements-rocm.txt`
- `serve_vllm_rocm.py`
- `benchmark_rocm.py`
- `README_AMD_MIGRATION.md`
- `migration_report.md`
- `pull_request_description.md`
- `flightdeck.patch`

## How To Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Gradio app:

```bash
python app/main.py
```

Open:

```text
http://localhost:7860
```

## How To Run Tests

```bash
python -m pytest
python -m compileall .
```

## How To Deploy To Hugging Face Space

1. Create a Hugging Face Space.
2. Choose SDK: `Gradio`.
3. Push this repository to the Space.
4. Keep the root-level `app.py`; it imports the Gradio app from `app/main.py`.
5. Ensure `requirements.txt` is included.
6. Keep the controlled demo repositories in `demo_repos/`.
7. Launch the Space, select a demo repo from the dropdown, and click `Run Analysis`.
8. Open the `Download Bundle` tab to download the generated migration bundle.

The Space does not need AMD Cloud credentials because it performs static analysis only.
Benchmarks are not executed locally or inside the Space; run `benchmark_rocm.py` later on AMD Developer Cloud / MI300X.

## Build In Public Strategy

ROCm FlightDeck is designed to produce judge-friendly, shareable outputs:

- A deterministic score.
- A clear before/after improvement.
- A reviewable patch.
- A migrated repository copy.
- A technical report.
- A public summary snippet.

Example build-in-public summary:

```text
Today we migrated a CUDA-first vLLM inference demo repo toward AMD ROCm.
Before: ROCm Readiness Score 20/100.
After: ROCm Readiness Score X/100.
Generated: Dockerfile.rocm, requirements-rocm.txt, serve_vllm_rocm.py,
benchmark_rocm.py, README_AMD_MIGRATION.md and flightdeck.patch.
Benchmark status: pending real execution on AMD Developer Cloud / MI300X.
```

## AMD Developer Hackathon Positioning

ROCm FlightDeck addresses the adoption gap between NVIDIA-first LLM inference repositories and AMD MI300X deployment. The product does not pretend migration is automatic. It makes migration measurable, reviewable, and easier to validate.

The current MVP is intentionally scoped to vLLM / PyTorch inference repositories because that is a practical, high-value surface for AMD GPU adoption.

## Roadmap

- Qwen integration for report writing and migration explanations.
- GitHub PR creation from generated migration artifacts.
- AMD Developer Cloud benchmark execution.
- Real MI300X validation with tokens/sec, latency, VRAM usage, cold start time, and test pass rate.
- Expanded detector coverage for custom CUDA extensions and HIP build paths.
