# ROCm FlightDeck - Migration Report
**Repository:** `nvidia_locked_vllm_demo`  
**Generated:** 2026-05-05 18:41 UTC  
**Tool:** ROCm FlightDeck MVP  

## Executive Summary

ROCm FlightDeck analyzed `nvidia_locked_vllm_demo`, generated reviewable ROCm starter artifacts, created a migrated repository copy, and rescored that copy using deterministic static analysis.

- Before Score: **20/100**
- Before Assessment: **Critical - NVIDIA/CUDA locked**
- After Score: **100/100**
- After Assessment: **ROCm Ready - no major static blockers detected**
- Score Improvement: **80 point(s)**
- Critical Blockers: **8**
- Warnings: **4**

Benchmarks not executed in this MVP.

## Before Score

### 20 / 100

**Assessment:** Critical - NVIDIA/CUDA locked

| Category | Score | Max |
|----------|-------|-----|
| Device abstraction | 0 | 20 |
| Dependency compatibility | 0 | 25 |
| Docker / runtime | 0 | 20 |
| vLLM serving readiness | 20 | 20 |
| Benchmark readiness | 0 | 15 |

Total ROCm Readiness Score: 20/100. Critical - NVIDIA/CUDA locked Category breakdown — Device abstraction: 0/20, Dependency compatibility: 0/25, Docker/runtime: 0/20, vLLM serving: 20/20, Benchmark readiness: 0/15.

Deductions:
- -10 device_abstraction: hardcoded torch.cuda reference (critical) in `app.py`. Recommendation: Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API.
- -8 device_abstraction: hardcoded .cuda() call (critical) in `app.py`. Recommendation: Replace .cuda() with .to(device) where device is resolved at runtime. ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel.
- -4 device_abstraction: .to('cuda') / .to("cuda") call (warning) in `app.py`. Recommendation: .to('cuda') works with ROCm PyTorch wheels on AMD GPUs. However, consider abstracting with a device variable for portability.
- -10 dependency_compatibility: bitsandbytes (NVIDIA-only) (critical) in `requirements.txt`. Recommendation: bitsandbytes is NVIDIA-only. Consider bitsandbytes-rocm (community fork), or use AWQ/GPTQ quantization which has ROCm-compatible implementations. For MI300X with 192 GB HBM, full-precision or FP8 inference may be viable without quantization.
- -7 dependency_compatibility: flash-attn (NVIDIA-optimized) (critical) in `requirements.txt`. Recommendation: flash-attn is built for NVIDIA. Use flash-attention-rocm (ROCm fork) or switch to PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention) which has ROCm support.
- -5 dependency_compatibility: xformers (NVIDIA-optimized) (critical) in `requirements.txt`. Recommendation: xformers is primarily NVIDIA-focused. For ROCm, use PyTorch SDPA or composable_kernel-based attention kernels. vLLM on ROCm has built-in attention backends that do not require xformers.
- -3 dependency_compatibility: triton (limited ROCm support) (warning) in `requirements.txt`. Recommendation: OpenAI Triton has experimental ROCm/HIP support via triton-rocm. Test carefully; some kernels may need porting. AMD provides composable_kernel as a lower-level alternative.
- -12 docker_runtime: nvidia/cuda Docker base image (critical) in `Dockerfile`. Recommendation: Replace with an AMD ROCm base image, e.g. rocm/pytorch:latest or rocm/rocm-terminal. See outputs/patches/<repo>/Dockerfile.rocm for a starter template.
- -8 docker_runtime: missing Dockerfile.rocm (warning). Recommendation: Add a Dockerfile.rocm using an AMD ROCm base image. See the generated template in outputs/patches/<repo>/Dockerfile.rocm.
- -10 benchmark_readiness: missing benchmark_rocm.py (info). Recommendation: Add benchmark_rocm.py to measure latency, tokens/sec, VRAM usage and cold-start time on AMD MI300X. A starter template is generated in outputs/patches/<repo>/benchmark_rocm.py.
- -5 benchmark_readiness: missing README_AMD_MIGRATION.md (info). Recommendation: Add README_AMD_MIGRATION.md documenting ROCm setup, driver requirements, and differences from the NVIDIA path. A starter is generated in outputs/patches/<repo>/README_AMD_MIGRATION.md.

## After Score

### 100 / 100

**Assessment:** ROCm Ready - no major static blockers detected

| Category | Score | Max |
|----------|-------|-----|
| Device abstraction | 20 | 20 |
| Dependency compatibility | 25 | 25 |
| Docker / runtime | 20 | 20 |
| vLLM serving readiness | 20 | 20 |
| Benchmark readiness | 15 | 15 |

Total ROCm Readiness Score: 100/100. ROCm Ready - no major static blockers detected Category breakdown — Device abstraction: 20/20, Dependency compatibility: 25/25, Docker/runtime: 20/20, vLLM serving: 20/20, Benchmark readiness: 15/15.

Deductions:
- No deductions.

## Score Improvement

The migrated copy improved by **80 point(s)**.

## Category Breakdown

| Category | Before | After | Max |
|----------|--------|-------|-----|
| Device abstraction | 0 | 20 | 20 |
| Dependency compatibility | 0 | 25 | 25 |
| Docker / runtime | 0 | 20 | 20 |
| vLLM serving readiness | 20 | 20 | 20 |
| Benchmark readiness | 0 | 15 | 15 |

## Critical Blockers

### [TORCH_CUDA] Hardcoded torch.cuda reference
**File:** `app.py`  
**Evidence:**
```
-> str:
    """Run a simple inference pass."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not av
```
**Recommendation:** Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API.

### [TORCH_CUDA] Hardcoded torch.cuda reference
**File:** `benchmark.py`  
**Evidence:**
```
ce():
    """Run a naive benchmark on CUDA."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skippin
```
**Recommendation:** Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API.

### [DOT_CUDA_CALL] Hardcoded .cuda() call
**File:** `app.py`  
**Evidence:**
```
t, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()   # .cuda() call

    with torch.no_grad():
        outp
```
**Recommendation:** Replace .cuda() with .to(device) where device is resolved at runtime. ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel.

### [DOT_CUDA_CALL] Hardcoded .cuda() call
**File:** `benchmark.py`  
**Evidence:**
```
late some tensor operations
    x = torch.randn(1024, 1024).cuda()
    t0 = time.time()
    for _ in range(100):
        x
```
**Recommendation:** Replace .cuda() with .to(device) where device is resolved at runtime. ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel.

### [NVIDIA_DOCKER] NVIDIA CUDA Docker base image
**File:** `Dockerfile`  
**Evidence:**
```
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="demo-team"
```
**Recommendation:** Replace with an AMD ROCm base image, e.g. rocm/pytorch:latest or rocm/rocm-terminal. See outputs/patches/<repo>/Dockerfile.rocm for a starter template.

### [BITSANDBYTES] bitsandbytes dependency (NVIDIA-only quantization)
**File:** `requirements.txt`  
**Evidence:**
```
torch
vllm
bitsandbytes
flash-attn
xformers
triton
transformers
accelerate
```
**Recommendation:** bitsandbytes is NVIDIA-only. Consider bitsandbytes-rocm (community fork), or use AWQ/GPTQ quantization which has ROCm-compatible implementations. For MI300X with 192 GB HBM, full-precision or FP8 inference may be viable without quantization.

### [FLASH_ATTN] flash-attn dependency (NVIDIA-optimized)
**File:** `requirements.txt`  
**Evidence:**
```
torch
vllm
bitsandbytes
flash-attn
xformers
triton
transformers
accelerate
```
**Recommendation:** flash-attn is built for NVIDIA. Use flash-attention-rocm (ROCm fork) or switch to PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention) which has ROCm support.

### [XFORMERS] xformers dependency (NVIDIA-optimized)
**File:** `requirements.txt`  
**Evidence:**
```
torch
vllm
bitsandbytes
flash-attn
xformers
triton
transformers
accelerate
```
**Recommendation:** xformers is primarily NVIDIA-focused. For ROCm, use PyTorch SDPA or composable_kernel-based attention kernels. vLLM on ROCm has built-in attention backends that do not require xformers.


## Warnings

### [TO_CUDA] .to("cuda") / .to('cuda') call
**File:** `app.py`  
**Evidence:**
```
pretrained(model_name, torch_dtype=torch.float16)
    model.to("cuda")   # hardcoded CUDA move
    return model, tokenizer


d
```
**Recommendation:** .to('cuda') works with ROCm PyTorch wheels on AMD GPUs. However, consider abstracting with a device variable for portability.

### [CUDA_HOME] CUDA_HOME environment variable reference
**File:** `Dockerfile`  
**Evidence:**
```
stall --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_HOME=/usr/local/cuda

CMD ["python", "app.py"]
```
**Recommendation:** Replace CUDA_HOME with ROCM_HOME or ROCM_PATH for AMD environments.

### [TRITON] triton dependency (limited ROCm support)
**File:** `requirements.txt`  
**Evidence:**
```
torch
vllm
bitsandbytes
flash-attn
xformers
triton
transformers
accelerate
```
**Recommendation:** OpenAI Triton has experimental ROCm/HIP support via triton-rocm. Test carefully; some kernels may need porting. AMD provides composable_kernel as a lower-level alternative.

### [MISSING_DOCKERFILE_ROCM] Missing Dockerfile.rocm
**Recommendation:** Add a Dockerfile.rocm using an AMD ROCm base image. See the generated template in outputs/patches/<repo>/Dockerfile.rocm.


## Migration Plan

- 0. Pin ROCm version: decide on ROCm 6.x for MI300X compatibility. Verify all dependencies against the ROCm compatibility matrix.
- 1. Replace hardcoded torch.cuda references with device abstraction. Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and pass `device` throughout the model loading and inference code.
- 2. Replace .cuda() calls with .to(device). Note: ROCm PyTorch exposes AMD GPUs through the CUDA-compatible API, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel — but abstracting the device string improves portability.
- 3. Keep .to('cuda') calls where ROCm PyTorch is used (they work on MI300X). For maximum portability, wrap in a device variable: `device = 'cuda'` and use `.to(device)`.
- 4. Replace the nvidia/cuda Docker base image with an AMD ROCm image. Use `rocm/pytorch:latest` or `rocm/rocm-terminal` as the base. See the generated Dockerfile.rocm in outputs/patches/<repo>/.
- 5. Create Dockerfile.rocm alongside the existing Dockerfile. A starter template has been generated at outputs/patches/<repo>/Dockerfile.rocm.
- 6. Replace CUDA_HOME environment variable references with ROCM_HOME or ROCM_PATH in build scripts, Dockerfiles, and shell scripts.
- 8. Remove or replace bitsandbytes. Options: (a) bitsandbytes-rocm community fork, (b) switch to AWQ or GPTQ quantization (ROCm-compatible), (c) for MI300X with 192 GB HBM consider full-precision or FP8 without quantization.
- 9. Replace flash-attn with a ROCm-compatible attention implementation. Options: (a) flash-attention-rocm fork, (b) torch.nn.functional.scaled_dot_product_attention (SDPA) — has ROCm backend.
- 10. Remove xformers or replace with ROCm-native attention. vLLM on ROCm uses built-in attention backends; xformers is not required. Use PyTorch SDPA or composable_kernel-based kernels for custom attention.
- 11. Audit triton usage. OpenAI Triton has experimental ROCm/HIP support (triton-rocm). Test each Triton kernel individually. AMD composable_kernel is the production alternative.
- 12. Build or install the ROCm-enabled vLLM wheel. Use the generated serve_vllm_rocm.py script as a starting point. Pass `--device rocm` or set HIP_VISIBLE_DEVICES as appropriate.
- 13. Add benchmark_rocm.py to measure performance on AMD MI300X. Measure: latency (ms/token), throughput (tokens/sec), VRAM usage (GB), cold-start time. A methodology template is in outputs/patches/<repo>/benchmark_rocm.py.
- 14. Add README_AMD_MIGRATION.md documenting: ROCm version requirements, driver setup, differences from NVIDIA path, known limitations, and how to run on AMD Developer Cloud / MI300X. A starter is in outputs/patches/<repo>/README_AMD_MIGRATION.md.
- 15. Test the migrated stack on AMD Developer Cloud (MI300X) or a local ROCm GPU. Run benchmark_rocm.py and compare against the NVIDIA baseline.

## Generated Artifacts

- `Dockerfile.rocm` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\Dockerfile.rocm`
- `requirements-rocm.txt` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\requirements-rocm.txt`
- `serve_vllm_rocm.py` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\serve_vllm_rocm.py`
- `benchmark_rocm.py` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\benchmark_rocm.py`
- `README_AMD_MIGRATION.md` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\README_AMD_MIGRATION.md`
- `flightdeck.patch` -> `C:\Users\Acer\AMD\rocm-flightdeck\outputs\patches\nvidia_locked_vllm_demo\flightdeck.patch`

## Migrated Repo Path

`C:\Users\Acer\AMD\rocm-flightdeck\outputs\migrated_repos\nvidia_locked_vllm_demo_rocm`

## Bundle Path

`C:\Users\Acer\AMD\rocm-flightdeck\outputs\bundles\nvidia_locked_vllm_demo_migration_bundle.zip`

## Pull Request Description

`C:\Users\Acer\AMD\rocm-flightdeck\outputs\reports\nvidia_locked_vllm_demo\pull_request_description.md`

## Generated Patch

`flightdeck.patch` is included in the generated artifacts and migration bundle. It contains reviewable starter hunks and must be validated before use.

## Benchmark Status

Benchmark status: not executed locally.

Reason: This MVP does not run GPU workloads on the user machine.

Next step: Run `benchmark_rocm.py` on AMD Developer Cloud / MI300X.

Metrics to collect:
- tokens/sec
- latency
- VRAM usage
- cold start time
- test pass rate

## Remaining Risks

- Benchmarks not executed in this MVP; validate on AMD Developer Cloud / MI300X.

## Next Steps

1. Review every generated artifact and `flightdeck.patch` hunk.
2. Build `Dockerfile.rocm` on an AMD ROCm machine.
3. Install ROCm PyTorch and ROCm-compatible vLLM dependencies.
4. Run functional tests on AMD Developer Cloud / MI300X.
5. Execute `benchmark_rocm.py` on real MI300X hardware and record real metrics.

## Build in Public Summary

Today we migrated a CUDA-first vLLM inference demo repo toward AMD ROCm. Before: ROCm Readiness Score 20/100. After: ROCm Readiness Score 100/100. Generated: Dockerfile.rocm, requirements-rocm.txt, serve_vllm_rocm.py, benchmark_rocm.py, README_AMD_MIGRATION.md and flightdeck.patch. Benchmark status: pending real execution on AMD Developer Cloud / MI300X.

_Generated by ROCm FlightDeck - open-source AI performance portability agent for AMD GPUs._