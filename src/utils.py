"""Small shared helpers: seeding, JSONL I/O, disk size and hardware info."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def path_size_mb(path: Path) -> float:
    """Exact on-disk size (MB = 1024^2 bytes) of a file or a whole directory."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 ** 2)


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10)
            if out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _ram_gb() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                kb = int(f.readline().split()[1])
            return f"{kb / 1024 ** 2:.1f} GB"
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            return f"{int(out.stdout.strip()) / 1024 ** 3:.1f} GB"
        if platform.system() == "Windows":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMSTAT()
            stat.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / 1024 ** 3:.1f} GB"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def get_hardware_info() -> dict:
    """Hardware / software description to report next to latency numbers."""
    info = {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["quantized_engine"] = torch.backends.quantized.engine
    except ImportError:
        pass
    return info


def configure_quantized_engine() -> str:
    """Pick a quantization backend that exists on this CPU.

    x86 (Intel/AMD) -> fbgemm / x86, ARM (Apple Silicon, Raspberry Pi) -> qnnpack.
    Must be called before quantizing AND before loading a quantized model.
    """
    import torch

    supported = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    return torch.backends.quantized.engine


def resolve_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
