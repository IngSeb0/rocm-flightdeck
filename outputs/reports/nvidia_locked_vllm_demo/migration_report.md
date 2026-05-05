# ROCm FlightDeck — Migration Report
**Repository:** `nvidia_locked_vllm_demo`  
**Generated:** 2026-05-05 14:03 UTC  
**Tool:** ROCm FlightDeck MVP  

---

## Summary

ROCm FlightDeck analysed `nvidia_locked_vllm_demo` and found **8 critical blocker(s)**, **4 warning(s)**, and **3 informational item(s)**.

The repository requires migration work before it can run reliably on AMD ROCm / MI300X.

---

## ROCm Readiness Score

### **20 / 100**

| Category | Score | Max |
|----------|-------|-----|
| Device abstraction | 0 | 20 |
| Dependency compatibility | 0 | 25 |
| Docker / runtime | 0 | 20 |
| vLLM serving readiness | 20 | 20 |
| Benchmark readiness | 0 | 15 |

**Assessment:** Total ROCm Readiness Score: 20/100. Critical — repository is tightly coupled to NVIDIA/CUDA infrastructure. Category breakdown — Device abstraction: 0/20, Dependency compatibility: 0/25, Docker/runtime: 0/20, vLLM serving: 20/20, Benchmark readiness: 0/15.

### Deductions

- -10 device_abstraction: hardcoded torch.cuda reference
- -8 device_abstraction: hardcoded .cuda() call
- -4 device_abstraction: .to('cuda') / .to("cuda") call
- -10 dependency_compatibility: bitsandbytes (NVIDIA-only)
- -7 dependency_compatibility: flash-attn (NVIDIA-optimized)
- -5 dependency_compatibility: xformers (NVIDIA-optimized)
- -3 dependency_compatibility: triton (limited ROCm support)
- -12 docker_runtime: nvidia/cuda Docker base image
- -8 docker_runtime: missing Dockerfile.rocm
- -10 benchmark_readiness: missing benchmark_rocm.py
- -5 benchmark_readiness: missing README_AMD_MIGRATION.md

---

## Critical Blockers

### ❌ [TORCH_CUDA] Hardcoded torch.cuda reference
**File:** `benchmark.py`  
**Evidence:**
```
ence():
    """Run a naive benchmark on CUDA."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping
```
**Recommendation:** Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API.

### ❌ [TORCH_CUDA] Hardcoded torch.cuda reference
**File:** `app.py`  
**Evidence:**
```
) -> str:
    """Run a simple inference pass."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not ava
```
**Recommendation:** Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API.

### ❌ [DOT_CUDA_CALL] Hardcoded .cuda() call
**File:** `benchmark.py`  
**Evidence:**
```
ulate some tensor operations
    x = torch.randn(1024, 1024).cuda()
    t0 = time.time()
    for _ in range(100):
        x = x
```
**Recommendation:** Replace .cuda() with .to(device) where device is resolved at runtime. ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel.

### ❌ [DOT_CUDA_CALL] Hardcoded .cuda() call
**File:** `app.py`  
**Evidence:**
```
pt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()   # .cuda() call

    with torch.no_grad():
        output_
```
**Recommendation:** Replace .cuda() with .to(device) where device is resolved at runtime. ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X when using the ROCm PyTorch wheel.

### ❌ [NVIDIA_DOCKER] NVIDIA CUDA Docker base image
**File:** `Dockerfile`  
**Evidence:**
```
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="demo-team"
LA
```
**Recommendation:** Replace with an AMD ROCm base image, e.g. rocm/pytorch:latest or rocm/rocm-terminal. See outputs/patches/<repo>/Dockerfile.rocm for a starter template.

### ❌ [BITSANDBYTES] bitsandbytes dependency (NVIDIA-only quantization)
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

### ❌ [FLASH_ATTN] flash-attn dependency (NVIDIA-optimized)
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

### ❌ [XFORMERS] xformers dependency (NVIDIA-optimized)
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

---

## Warnings

### ⚠️ [TO_CUDA] .to("cuda") / .to('cuda') call
**File:** `app.py`  
**Evidence:**
```
_pretrained(model_name, torch_dtype=torch.float16)
    model.to("cuda")   # hardcoded CUDA move
    return model, tokenizer


def r
```
**Recommendation:** .to('cuda') works with ROCm PyTorch wheels on AMD GPUs. However, consider abstracting with a device variable for portability.

### ⚠️ [CUDA_HOME] CUDA_HOME environment variable reference
**File:** `Dockerfile`  
**Evidence:**
```
p install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_HOME=/usr/local/cuda

CMD ["python", "app.py"]
```
**Recommendation:** Replace CUDA_HOME with ROCM_HOME or ROCM_PATH for AMD environments.

### ⚠️ [TRITON] triton dependency (limited ROCm support)
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

### ⚠️ [MISSING_DOCKERFILE_ROCM] Missing Dockerfile.rocm
**Recommendation:** Add a Dockerfile.rocm using an AMD ROCm base image. See the generated template in outputs/patches/<repo>/Dockerfile.rocm.

---

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

---

## Generated Files

The following files were generated in `outputs/patches/`:

- `Dockerfile.rocm` → `outputs/patches/nvidia_locked_vllm_demo/Dockerfile.rocm`
- `requirements-rocm.txt` → `outputs/patches/nvidia_locked_vllm_demo/requirements-rocm.txt`
- `serve_vllm_rocm.py` → `outputs/patches/nvidia_locked_vllm_demo/serve_vllm_rocm.py`
- `benchmark_rocm.py` → `outputs/patches/nvidia_locked_vllm_demo/benchmark_rocm.py`
- `README_AMD_MIGRATION.md` → `outputs/patches/nvidia_locked_vllm_demo/README_AMD_MIGRATION.md`
- `flightdeck.patch` → `outputs/patches/nvidia_locked_vllm_demo/flightdeck.patch`

---

## Benchmark Status

> **Benchmark status: not executed in this local MVP.**  
> Run `benchmark_rocm.py` on AMD Developer Cloud / MI300X for real results.

---

## Remaining Risks

- bitsandbytes quantization may not be fully supported on ROCm without a community fork.
- Triton kernels need individual validation on the HIP backend.
- flash-attn must be replaced with a ROCm-compatible implementation.
- Driver and ROCm version compatibility must be validated end-to-end.
- Performance parity with NVIDIA is not guaranteed and requires real benchmarking.

---

## Next Steps

1. Address all **critical** blockers listed above.
2. Review and apply `flightdeck.patch` (in outputs/patches/).
3. Build and test `Dockerfile.rocm` on an AMD machine.
4. Install ROCm PyTorch wheel and `requirements-rocm.txt`.
5. Run `serve_vllm_rocm.py` and validate the serving endpoint.
6. Run `benchmark_rocm.py` on MI300X and record real performance numbers.
7. Fill in the benchmark table in `README_AMD_MIGRATION.md`.

---

_Generated by ROCm FlightDeck — open-source AI performance portability agent for AMD GPUs._