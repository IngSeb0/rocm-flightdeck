# ROCm FlightDeck — Agent Rules

## Project Overview
ROCm FlightDeck is an open-source AI performance portability agent for AMD GPUs.
It analyzes CUDA/NVIDIA-first LLM inference repositories, detects ROCm compatibility
blockers, generates a deterministic ROCm Readiness Score, creates a migration plan,
generates reviewable ROCm artifacts, and produces a technical migration report.

## Scope
- **Focused MVP**: vLLM / PyTorch LLM inference repositories migrating to AMD ROCm / MI300X.
- **Not a generic CUDA-to-ROCm converter.**

## Core Agent Rules

### 1. Deterministic checks before LLM reasoning
All scoring, detection, and analysis must be deterministic.
- No random numbers.
- No LLM judgment for scoring or detection.
- All checks must be reproducible given the same input.

### 2. Never execute untrusted repository code
- The agent MUST NOT run any code from a scanned repository.
- Static analysis only: read files, parse text, pattern-match.
- No subprocess calls into scanned code.

### 3. No fake benchmark results
- The MVP does not run real benchmarks.
- `benchmark_rocm.py` contains methodology and placeholder code only.
- Reports must state clearly: "Benchmarks not executed in this MVP."
- Do NOT fabricate latency, throughput, or VRAM numbers.

### 4. Generated patches must be reviewable
- All generated artifacts (Dockerfile.rocm, requirements-rocm.txt, patches) are
  written to `outputs/patches/<repo_name>/`.
- Original files are never overwritten.
- Patches are clearly marked as starter templates requiring human review.
- Every generated file includes a header warning that it is agent-generated.

### 5. MVP focus: vLLM / PyTorch inference on ROCm
Detectors and planner are tuned for:
- PyTorch CUDA device assumptions (`torch.cuda`, `.cuda()`, `.to("cuda")`)
- NVIDIA Docker base images
- vLLM serving infrastructure
- Incompatible dependencies: bitsandbytes, flash-attn, xformers, triton
- ROCm-ready alternatives and workarounds for AMD MI300X

## What the MVP Does NOT Do
- Does not integrate Qwen or other LLMs for reasoning.
- Does not integrate AMD Cloud or any cloud provider.
- Does not run real benchmarks.
- Does not execute untrusted repository code.
- Does not require internet access or AMD credentials.

## Output Contract
Every analysis run produces:
1. `ROCm Readiness Score` (0–100, deterministic)
2. `Blockers` list with severity (critical / warning / info)
3. `Migration Plan` (ordered steps)
4. `Generated Artifacts` (files written to outputs/)
5. `Technical Report` (migration_report.md)
