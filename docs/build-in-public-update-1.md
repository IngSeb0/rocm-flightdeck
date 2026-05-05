# Build in Public Update 1 - ROCm FlightDeck

## LinkedIn / X Post

Today I upgraded ROCm FlightDeck from a static ROCm scanner into an end-to-end migration demo.

It now takes a CUDA-first vLLM demo repo through:

Detect -> Plan -> Patch -> Create migrated repo -> Re-score -> Report -> Download bundle.

Before: ROCm Readiness Score 20/100.
After: ROCm Readiness Score 100/100 on static checks for the migrated copy.

Detected blockers included `.cuda()`, `model.to("cuda")`, `torch.cuda` assumptions, `nvidia/cuda` Docker base image, `CUDA_HOME`, and NVIDIA-risk dependencies like bitsandbytes, flash-attn, xformers, and triton.

Generated files include Dockerfile.rocm, requirements-rocm.txt, serve_vllm_rocm.py, benchmark_rocm.py, README_AMD_MIGRATION.md, migration_report.md, pull_request_description.md, and flightdeck.patch.

Honest limitation: benchmarks are still pending real execution on AMD Developer Cloud / MI300X. No latency, throughput, or VRAM numbers are fabricated.

Next milestones: Qwen-assisted report writing, GitHub PR creation, AMD Developer Cloud benchmark execution, and real MI300X validation.

## Technical Summary

ROCm FlightDeck is an AI performance portability lab for AMD GPUs. It is not a generic CUDA-to-ROCm converter. The current MVP focuses on vLLM / PyTorch LLM inference repositories and keeps all scoring deterministic.

This update improved the migrated repository structure so the generated copy is ROCm-first:

- Original `Dockerfile` with NVIDIA assumptions is preserved as `legacy/Dockerfile.nvidia.legacy`.
- Active `Dockerfile` is promoted from `Dockerfile.rocm`.
- Original risky requirements are preserved as `legacy/requirements.nvidia.legacy.txt`.
- Active `requirements.txt` is promoted from `requirements-rocm.txt`.
- CUDA benchmark code is preserved as `legacy/benchmark.cuda.legacy.py`.
- ROCm benchmark methodology is added as `benchmark_rocm.py`.

## Before / After Score

- Before: 20/100 - Critical - NVIDIA/CUDA locked.
- After: 100/100 - ROCm Ready - no major static blockers detected.
- Improvement: +80 points.

## Blockers Detected

- Hardcoded `torch.cuda` references.
- Bare `.cuda()` calls.
- `.to("cuda")` model movement.
- NVIDIA CUDA Docker base image.
- `CUDA_HOME` environment variable.
- bitsandbytes dependency.
- flash-attn dependency.
- xformers dependency.
- triton dependency.
- Missing ROCm Dockerfile.
- Missing ROCm benchmark methodology.

## Files Generated

- `Dockerfile.rocm`
- `requirements-rocm.txt`
- `serve_vllm_rocm.py`
- `benchmark_rocm.py`
- `README_AMD_MIGRATION.md`
- `flightdeck.patch`
- `migration_report.md`
- `pull_request_description.md`
- `<repo>_migration_bundle.zip`

## Benchmark Status

Benchmarks not executed in this MVP.

The benchmark template defines what to collect later on AMD Developer Cloud / MI300X:

- tokens/sec
- latency
- VRAM usage
- cold start time
- test pass rate

## Next Milestones

- Add Qwen-assisted migration report narration without replacing deterministic checks.
- Generate GitHub pull requests from migration bundles.
- Execute `benchmark_rocm.py` on AMD Developer Cloud / MI300X.
- Record real MI300X validation results.
- Expand detectors for custom CUDA extensions and HIP build paths.
