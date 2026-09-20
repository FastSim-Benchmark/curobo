# SPDX-License-Identifier: Apache-2.0

"""Bounded CoACD partition proposal using the already-installed public API."""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def propose_parts(vertices, faces, max_parts, seed, timeout=30):
    """Isolate native decomposition so a difficult mesh cannot block the batch."""
    if importlib.util.find_spec("coacd") is None:
        raise ImportError("FAST sphere fitting requires the coacd dependency")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="curobo-coacd-") as directory:
        directory = Path(directory)
        source = directory / "input.npz"
        target = directory / "parts.npz"
        np.savez_compressed(source, vertices=vertices, faces=faces)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input",
            str(source),
            "--output",
            str(target),
            "--max-parts",
            str(max_parts),
            "--seed",
            str(seed),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return None, {
                "status": "timeout",
                "timeout_s": timeout,
                "wall_time_s": time.perf_counter() - started,
            }
        if completed.returncode != 0:
            return None, {
                "status": "failed",
                "exit_code": completed.returncode,
                "diagnostic": completed.stderr[-2500:],
                "wall_time_s": time.perf_counter() - started,
            }
        with np.load(target) as result:
            parts = [result[f"v{i}"] for i in range(len(result.files))]
        metadata = json.loads(target.with_suffix(".json").read_text())
        metadata.update({"status": "completed", "wall_time_s": time.perf_counter() - started})
        return parts, metadata


def main():
    import coacd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    with np.load(args.input) as source:
        vertices, faces = source["vertices"], source["faces"]
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = np.ptp(vertices, axis=0).max()
    coacd.set_log_level("error")
    settings = {
        "threshold": 0.15,
        "max_convex_hull": args.max_parts,
        "resolution": 1000,
        "mcts_nodes": 10,
        "mcts_iterations": 25,
        "mcts_max_depth": 2,
        "preprocess_resolution": 30,
        "seed": args.seed,
    }
    parts = coacd.run_coacd(coacd.Mesh((vertices - center) / scale, faces), **settings)
    if not parts:
        raise ValueError("CoACD returned no convex parts")
    np.savez_compressed(
        args.output, **{f"v{i}": p[0] * scale + center for i, p in enumerate(parts)}
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": importlib.metadata.version("coacd"),
                "settings": settings,
                "num_parts": len(parts),
                "normalization": "AABB max extent = 1",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
