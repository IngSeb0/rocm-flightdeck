---
name: readiness-scoring
description: Use when implementing or modifying the ROCm Readiness Score, scoring rubric, blocker weights, or before/after score explanations.
---

You are working on the ROCm Readiness Score.

The score must be deterministic and explainable.

Use this rubric:
- Device abstraction: 20 points
- Dependency compatibility: 25 points
- Docker/runtime compatibility: 20 points
- vLLM serving readiness: 20 points
- Benchmark readiness: 15 points

Every deduction must include:
- reason
- affected file if available
- severity
- recommendation

Never generate a score using only LLM judgment.