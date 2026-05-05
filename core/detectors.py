"""Deterministic detectors for ROCm compatibility blockers."""

from __future__ import annotations

import re
from typing import List

from core.models import Detection, ScanResult


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_evidence(text: str, pattern: str, context: int = 60) -> str:
    """Return a short snippet around the first regex match in *text*."""
    m = re.search(pattern, text)
    if not m:
        return ""
    start = max(0, m.start() - context)
    end = min(len(text), m.end() + context)
    return text[start:end].strip()


def _iter_relevant_files(scan: ScanResult, suffixes: tuple[str, ...]):
    """Yield files whose normalized lowercase path ends with any suffix."""
    for path, content in scan.file_contents.items():
        normalized = path.replace("\\", "/").lower()
        if _is_legacy_path(normalized):
            continue
        if normalized.endswith(suffixes):
            yield path, content


def _is_legacy_path(normalized_path: str) -> bool:
    """Return True for preserved NVIDIA/CUDA legacy files."""
    return (
        normalized_path.startswith("legacy/")
        or ".legacy." in normalized_path
        or normalized_path.endswith(".legacy")
        or normalized_path.endswith(".nvidia.legacy")
    )


def _active_lines(text: str) -> str:
    """Return non-empty, non-comment lines to avoid dependency false positives in docs."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(line)
    return "\n".join(lines)


def _without_portable_device_abstraction(text: str) -> str:
    pattern = (
        r"\w+\s*=\s*torch\.device\(\s*[\"']cuda[\"']\s+if\s+"
        r"torch\.cuda\.is_available\(\)\s+else\s+[\"']cpu[\"']\s*\)"
    )
    return re.sub(pattern, "", text)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_torch_cuda(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"torch\.cuda"
    for path, content in _iter_relevant_files(scan, (".py", ".sh", "dockerfile")):
        normalized = path.replace("\\", "/").lower()
        if normalized.endswith("benchmark_rocm.py"):
            continue
        active_content = _without_portable_device_abstraction(_active_lines(content))
        if re.search(pattern, active_content):
            detections.append(Detection(
                id="TORCH_CUDA",
                title="Hardcoded torch.cuda reference",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "Abstract device selection with torch.device('cuda' if torch.cuda.is_available() else 'cpu') "
                    "or use the ROCm-compatible HIP backend. ROCm PyTorch exposes GPUs through the CUDA-compatible API."
                ),
            ))
    return detections


def detect_cuda_call(scan: ScanResult) -> List[Detection]:
    """Detect bare .cuda() tensor/model calls."""
    detections: List[Detection] = []
    pattern = r"\.cuda\(\)"
    for path, content in _iter_relevant_files(scan, (".py",)):
        if re.search(pattern, content):
            detections.append(Detection(
                id="DOT_CUDA_CALL",
                title="Hardcoded .cuda() call",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "Replace .cuda() with .to(device) where device is resolved at runtime. "
                    "ROCm PyTorch supports the CUDA device string, so .to('cuda') works on MI300X "
                    "when using the ROCm PyTorch wheel."
                ),
            ))
    return detections


def detect_to_cuda(scan: ScanResult) -> List[Detection]:
    """Detect .to('cuda') or .to(\"cuda\") calls."""
    detections: List[Detection] = []
    pattern = r'\.to\(["\']cuda["\']\)'
    for path, content in _iter_relevant_files(scan, (".py",)):
        if re.search(pattern, content):
            detections.append(Detection(
                id="TO_CUDA",
                title='.to("cuda") / .to(\'cuda\') call',
                severity="warning",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    ".to('cuda') works with ROCm PyTorch wheels on AMD GPUs. "
                    "However, consider abstracting with a device variable for portability."
                ),
            ))
    return detections


def detect_nvidia_docker(scan: ScanResult) -> List[Detection]:
    """Detect nvidia/cuda base image in Dockerfile."""
    detections: List[Detection] = []
    pattern = r"nvidia/cuda"
    for path, content in _iter_relevant_files(scan, ("dockerfile",)):
        if re.search(pattern, _active_lines(content)):
            detections.append(Detection(
                id="NVIDIA_DOCKER",
                title="NVIDIA CUDA Docker base image",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "Replace with an AMD ROCm base image, e.g. "
                    "rocm/pytorch:latest or rocm/rocm-terminal. "
                    "See outputs/patches/<repo>/Dockerfile.rocm for a starter template."
                ),
            ))
    return detections


def detect_cuda_home(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"CUDA_HOME"
    for path, content in _iter_relevant_files(scan, (".py", ".sh", "dockerfile")):
        if re.search(pattern, content):
            detections.append(Detection(
                id="CUDA_HOME",
                title="CUDA_HOME environment variable reference",
                severity="warning",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "Replace CUDA_HOME with ROCM_HOME or ROCM_PATH for AMD environments."
                ),
            ))
    return detections


def detect_nvcc(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"\bnvcc\b"
    for path, content in _iter_relevant_files(scan, (".py", ".sh", "dockerfile", ".toml", ".txt")):
        if re.search(pattern, content):
            detections.append(Detection(
                id="NVCC",
                title="nvcc compiler reference",
                severity="warning",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "Replace nvcc with hipcc for AMD GPU compilation. "
                    "For PyTorch extensions use the HIP-compatible build path."
                ),
            ))
    return detections


def detect_bitsandbytes(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"bitsandbytes"
    for path, content in _iter_relevant_files(scan, ("requirements.txt", "pyproject.toml", "setup.py", ".py")):
        if re.search(pattern, _active_lines(content)):
            detections.append(Detection(
                id="BITSANDBYTES",
                title="bitsandbytes dependency (NVIDIA-only quantization)",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "bitsandbytes is NVIDIA-only. "
                    "Consider bitsandbytes-rocm (community fork), or use AWQ/GPTQ quantization "
                    "which has ROCm-compatible implementations. For MI300X with 192 GB HBM, "
                    "full-precision or FP8 inference may be viable without quantization."
                ),
            ))
    return detections


def detect_flash_attn(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"flash[-_]attn|flash_attention"
    for path, content in _iter_relevant_files(scan, ("requirements.txt", "pyproject.toml", "setup.py", ".py")):
        if re.search(pattern, _active_lines(content), re.IGNORECASE):
            detections.append(Detection(
                id="FLASH_ATTN",
                title="flash-attn dependency (NVIDIA-optimized)",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "flash-attn is built for NVIDIA. "
                    "Use flash-attention-rocm (ROCm fork) or switch to PyTorch SDPA "
                    "(torch.nn.functional.scaled_dot_product_attention) which has ROCm support."
                ),
            ))
    return detections


def detect_xformers(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"xformers"
    for path, content in _iter_relevant_files(scan, ("requirements.txt", "pyproject.toml", "setup.py", ".py")):
        if re.search(pattern, _active_lines(content)):
            detections.append(Detection(
                id="XFORMERS",
                title="xformers dependency (NVIDIA-optimized)",
                severity="critical",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "xformers is primarily NVIDIA-focused. "
                    "For ROCm, use PyTorch SDPA or composable_kernel-based attention kernels. "
                    "vLLM on ROCm has built-in attention backends that do not require xformers."
                ),
            ))
    return detections


def detect_triton(scan: ScanResult) -> List[Detection]:
    detections: List[Detection] = []
    pattern = r"\btriton\b"
    for path, content in _iter_relevant_files(scan, ("requirements.txt", "pyproject.toml", "setup.py", ".py")):
        if re.search(pattern, _active_lines(content)):
            detections.append(Detection(
                id="TRITON",
                title="triton dependency (limited ROCm support)",
                severity="warning",
                file_path=path,
                evidence=_find_evidence(content, pattern),
                recommendation=(
                    "OpenAI Triton has experimental ROCm/HIP support via triton-rocm. "
                    "Test carefully; some kernels may need porting. "
                    "AMD provides composable_kernel as a lower-level alternative."
                ),
            ))
    return detections


def detect_missing_dockerfile_rocm(scan: ScanResult) -> List[Detection]:
    has_rocm_dockerfile = any(
        "Dockerfile.rocm" in p for p in scan.files_scanned
    )
    if not has_rocm_dockerfile:
        return [Detection(
            id="MISSING_DOCKERFILE_ROCM",
            title="Missing Dockerfile.rocm",
            severity="warning",
            recommendation=(
                "Add a Dockerfile.rocm using an AMD ROCm base image. "
                "See the generated template in outputs/patches/<repo>/Dockerfile.rocm."
            ),
        )]
    return []


def detect_missing_benchmark_rocm(scan: ScanResult) -> List[Detection]:
    has_benchmark = any(
        "benchmark_rocm" in p for p in scan.files_scanned
    )
    if not has_benchmark:
        return [Detection(
            id="MISSING_BENCHMARK_ROCM",
            title="Missing benchmark_rocm.py",
            severity="info",
            recommendation=(
                "Add benchmark_rocm.py to measure latency, tokens/sec, VRAM usage "
                "and cold-start time on AMD MI300X. "
                "A starter template is generated in outputs/patches/<repo>/benchmark_rocm.py."
            ),
        )]
    return []


def detect_missing_readme_amd(scan: ScanResult) -> List[Detection]:
    has_readme = any(
        "README_AMD_MIGRATION" in p for p in scan.files_scanned
    )
    if not has_readme:
        return [Detection(
            id="MISSING_README_AMD",
            title="Missing README_AMD_MIGRATION.md",
            severity="info",
            recommendation=(
                "Add README_AMD_MIGRATION.md documenting ROCm setup, driver requirements, "
                "and differences from the NVIDIA path. "
                "A starter is generated in outputs/patches/<repo>/README_AMD_MIGRATION.md."
            ),
        )]
    return []


def detect_vllm(scan: ScanResult) -> List[Detection]:
    """Detect presence or absence of vLLM dependency (informational)."""
    has_vllm = any(
        re.search(r"\bvllm\b", content)
        for content in scan.file_contents.values()
    )
    if has_vllm:
        return [Detection(
            id="VLLM_PRESENT",
            title="vLLM dependency detected",
            severity="info",
            recommendation=(
                "vLLM has official ROCm support. "
                "Use the ROCm-enabled vLLM build: pip install vllm (ROCm wheel) "
                "or build from source with ROCm. Verify the serving script uses "
                "the correct device arguments."
            ),
        )]
    return [Detection(
        id="VLLM_MISSING",
        title="vLLM dependency not detected",
        severity="info",
        recommendation=(
            "If vLLM serving is intended, add vllm to requirements and "
            "use the ROCm-compatible vLLM wheel."
        ),
    )]


# ---------------------------------------------------------------------------
# Run all detectors
# ---------------------------------------------------------------------------

def run_all_detectors(scan: ScanResult) -> List[Detection]:
    """Run every detector and return the combined list of detections."""
    detections: List[Detection] = []
    detections.extend(detect_torch_cuda(scan))
    detections.extend(detect_cuda_call(scan))
    detections.extend(detect_to_cuda(scan))
    detections.extend(detect_nvidia_docker(scan))
    detections.extend(detect_cuda_home(scan))
    detections.extend(detect_nvcc(scan))
    detections.extend(detect_bitsandbytes(scan))
    detections.extend(detect_flash_attn(scan))
    detections.extend(detect_xformers(scan))
    detections.extend(detect_triton(scan))
    detections.extend(detect_missing_dockerfile_rocm(scan))
    detections.extend(detect_missing_benchmark_rocm(scan))
    detections.extend(detect_missing_readme_amd(scan))
    detections.extend(detect_vllm(scan))
    return detections
