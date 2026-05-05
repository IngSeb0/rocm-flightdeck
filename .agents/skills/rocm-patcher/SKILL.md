---
name: rocm-patcher
description: Use when generating reviewable ROCm migration patches, Dockerfile.rocm, requirements-rocm.txt, vLLM serving scripts, or benchmark scripts.
---

You are working on ROCm FlightDeck's patch generation system.

Generate reviewable patches, not unsafe automatic rewrites.

Prefer creating new files over destructively modifying user files.

Expected outputs:
- Dockerfile.rocm
- requirements-rocm.txt
- serve_vllm_rocm.py
- benchmark_rocm.py
- README_AMD_MIGRATION.md
- flightdeck.patch

Always explain:
- what changed
- why it helps ROCm compatibility
- what risks remain

Do not claim a patch is production-ready unless it has been tested.