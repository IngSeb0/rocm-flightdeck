---
name: rocm-repo-scanner
description: Use when implementing or improving static analysis for CUDA/NVIDIA assumptions in Python, PyTorch, vLLM, Dockerfile, or requirements files.
---

You are working on ROCm FlightDeck's deterministic repository scanner.

Focus on static analysis only. Do not execute untrusted repo code.

Detect:
- torch.cuda
- .cuda()
- .to("cuda") and .to('cuda')
- nvidia/cuda Docker base images
- CUDA_HOME
- nvcc
- bitsandbytes
- flash-attn
- xformers
- triton
- missing Dockerfile.rocm
- missing benchmark_rocm.py

Return structured data with:
- files_scanned
- blockers
- warnings
- detected_dependencies
- detected_runtime_assumptions

Keep implementation simple, testable, and deterministic.