#!/usr/bin/env python3
"""Write hardware / OS / package versions to results/env.json."""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGS = ["tensorflow", "keras", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib",
        "pyyaml", "astroquery", "astropy", "pytest", "nbconvert", "jupyter"]


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e!r}"


env = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
    "logical_cpus": os.cpu_count(),
    "memory_bytes": sh(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else "n/a",
    "gpu_used": False,
    "packages": {p: metadata.version(p) for p in PKGS},
    "determinism": {"TF_DETERMINISTIC_OPS": "1", "tf.config.experimental.enable_op_determinism": True,
                    "seeds": "python random, numpy, tf via keras.utils.set_random_seed (per run)"},
    "lockfile": "uv.lock",
    "git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
}
out = ROOT / "results" / "env.json"
out.write_text(json.dumps(env, indent=1) + "\n")
print(json.dumps(env, indent=1))
