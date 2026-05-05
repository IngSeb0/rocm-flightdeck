# 🚀 ROCm FlightDeck

**Autonomous performance portability for LLM inference on AMD GPUs.**

ROCm FlightDeck is an open-source AI performance portability agent for AMD GPUs.
It analyzes CUDA/NVIDIA-first LLM inference repositories, detects ROCm compatibility
blockers, generates a deterministic ROCm Readiness Score, creates a migration plan,
generates reviewable ROCm artifacts, and produces a technical migration report.

> **Build in Public** — This project is developed openly as part of the AMD Hackathon.
> All analysis is deterministic, no LLM reasoning is used for scoring.

---

## What ROCm FlightDeck Does

| Feature | Description |
|---------|-------------|
| 📊 ROCm Readiness Score | Deterministic 0–100 score across 5 categories |
| 🚫 Blocker Detection | Finds CUDA/NVIDIA assumptions in Python, Dockerfiles, requirements |
| 🗺️ Migration Plan | Ordered, actionable steps for each detected issue |
| 📁 Artifact Generation | Generates Dockerfile.rocm, requirements-rocm.txt, serve/benchmark scripts |
| 📋 Technical Report | Full migration_report.md with score breakdown, blockers, and next steps |

**Focused on:** vLLM / PyTorch LLM inference repositories migrating to AMD ROCm / MI300X.

---

## What the MVP Does NOT Do Yet

- ❌ No LLM/Qwen integration for reasoning
- ❌ No AMD Cloud or remote execution
- ❌ No real benchmark execution (placeholder methodology only)
- ❌ Does not run any untrusted repository code
- ❌ No internet access required

---

## How to Run Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Gradio App

```bash
python app/main.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

**Default demo:** `demo_repos/nvidia_locked_vllm_demo`

Click **Run Analysis** to see:
- ROCm Readiness Score
- Detected blockers and warnings
- Migration plan
- Generated ROCm artifacts
- Full technical report

---

## How to Run Tests

```bash
python -m pytest
```

Or with verbose output:

```bash
python -m pytest -v
```

Expected: all tests pass. Tests cover scanner, detectors, scoring, and patcher.

---

## Demo Flow

1. Start the app: `python app/main.py`
2. The default repo path points to `demo_repos/nvidia_locked_vllm_demo`
3. Click **Run Analysis**
4. Observe:
   - Score: ~20/100 (intentionally bad demo repo)
   - 7+ critical blockers (torch.cuda, nvidia/cuda Docker, bitsandbytes, etc.)
   - A full migration plan with 15 ordered steps
   - 6 generated artifacts in `outputs/patches/nvidia_locked_vllm_demo/`
   - A migration report at `outputs/reports/nvidia_locked_vllm_demo/migration_report.md`

---

## Expected Output

### ROCm Readiness Score
```
Total: ~20 / 100
Device abstraction:        0 / 20  (hardcoded CUDA)
Dependency compatibility:  0 / 25  (bitsandbytes, flash-attn, xformers, triton)
Docker/runtime:            0 / 20  (nvidia/cuda base image, no Dockerfile.rocm)
vLLM serving readiness:   20 / 20  (vllm detected)
Benchmark readiness:       0 / 15  (no benchmark_rocm.py or README_AMD_MIGRATION.md)
```

### Generated Files
```
outputs/patches/nvidia_locked_vllm_demo/
├── Dockerfile.rocm
├── requirements-rocm.txt
├── serve_vllm_rocm.py
├── benchmark_rocm.py
├── README_AMD_MIGRATION.md
└── flightdeck.patch

outputs/reports/nvidia_locked_vllm_demo/
└── migration_report.md
```

---

## Project Structure

```
rocm-flightdeck/
├── AGENTS.md                         # Agent rules and project scope
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   └── main.py                       # Gradio app
├── core/
│   ├── __init__.py
│   ├── scanner.py                    # File scanner
│   ├── detectors.py                  # CUDA/NVIDIA pattern detectors
│   ├── scoring.py                    # ROCm Readiness Score
│   ├── planner.py                    # Migration plan generator
│   ├── patcher.py                    # ROCm artifact generator
│   ├── reporter.py                   # Migration report generator
│   └── models.py                     # Data models
├── demo_repos/
│   └── nvidia_locked_vllm_demo/      # Intentionally NVIDIA-locked demo
│       ├── app.py
│       ├── requirements.txt
│       ├── Dockerfile
│       └── benchmark.py
├── outputs/
│   ├── reports/                      # Generated migration reports
│   └── patches/                      # Generated ROCm artifacts
└── tests/
    ├── __init__.py
    ├── test_scanner.py
    ├── test_scoring.py
    └── test_patcher.py
```

---

## Hackathon Positioning

ROCm FlightDeck addresses a real pain point: developers who want to run LLM inference
on AMD MI300X face significant migration friction from NVIDIA-first repositories.

This MVP demonstrates:
- **Deterministic analysis** — no hallucinated scores or fabricated benchmarks
- **Actionable output** — generates runnable starter files, not just a checklist
- **Focused scope** — vLLM + PyTorch, the most common LLM inference stack
- **Build in Public** — fully open-source, hackathon-developed

---

## Build in Public

This project is built openly during the AMD Hackathon.
Follow the development on GitHub: [IngSeb0/rocm-flightdeck](https://github.com/IngSeb0/rocm-flightdeck)
