#!/usr/bin/env python3
"""Validate the ZGCM-1 pretrain and midtrain stage configurations.

Each stage env file is sourced in a clean bash process, the same way
``launch_train.sh`` sources it, and the resulting variables are checked for
internal consistency: parallel geometry, batch sizes, sample budgets and
resume bookkeeping. The script exits non-zero if any stage fails a check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIRS = ("pretrain", "midtrain")

# Values that launch_train.sh requires from the scheduler. They only need to be
# present so that the env files can be sourced; nothing here is checked.
PLACEHOLDERS = {
    "SHARED_STORAGE_ROOT": "/nonexistent/storage",
    "ZGCM_REPO_ROOT": "/nonexistent/repo",
    "NODE_RANK": "0",
    "MASTER_ADDR": "127.0.0.1",
}

REQUIRED = (
    "EXP_NAME",
    "TRAIN_PROGRAM",
    "OPTIMIZER",
    "LR",
    "NUM_LAYERS",
    "NUM_ATTENTION_HEADS",
    "NUM_QUERY_GROUPS",
    "TP_SIZE",
    "PP_SIZE",
    "CP_SIZE",
    "MICRO_BATCH_SIZE",
    "GLOBAL_BATCH_SIZE",
    "SEQ_LENGTH",
    "MAX_POSITION_EMBEDDINGS",
)

FORBIDDEN_PATH_FRAGMENTS = ("/home/", "/mnt/", "/root/", "/data/")


def load_env(path: Path) -> dict[str, str]:
    """Source a stage file exactly as launch_train.sh does and return its exports."""
    proc = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(path)],
        env={**PLACEHOLDERS, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    result: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if not item:
            continue
        key, _, value = item.decode(errors="replace").partition("=")
        result[key] = value
    return result


def check_stage(env: dict[str, str]) -> tuple[list[str], dict[str, object]]:
    """Return (errors, summary) for one resolved stage configuration."""
    errors = [f"missing or empty {name}" for name in REQUIRED if not env.get(name)]
    if errors:
        return errors, {}

    tp, pp, cp = int(env["TP_SIZE"]), int(env["PP_SIZE"]), int(env["CP_SIZE"])
    mbs, gbs = int(env["MICRO_BATCH_SIZE"]), int(env["GLOBAL_BATCH_SIZE"])
    seq, max_pos = int(env["SEQ_LENGTH"]), int(env["MAX_POSITION_EMBEDDINGS"])
    layers = int(env["NUM_LAYERS"])
    heads, groups = int(env["NUM_ATTENTION_HEADS"]), int(env["NUM_QUERY_GROUPS"])

    world = int(env["NUM_NODES"]) * int(env["GPUS_PER_NODE"])
    model_parallel = tp * pp * cp
    summary: dict[str, object] = {"world": world, "tp": tp, "pp": pp, "cp": cp}

    if world % model_parallel:
        errors.append(f"world size {world} is not divisible by TP*PP*CP={model_parallel}")
        return errors, summary
    dp = world // model_parallel
    summary["dp"] = dp

    if gbs % (mbs * dp):
        errors.append(f"GLOBAL_BATCH_SIZE {gbs} is not divisible by MICRO_BATCH_SIZE*DP={mbs * dp}")
    else:
        summary["accum"] = gbs // (mbs * dp)

    if layers % pp:
        errors.append(f"NUM_LAYERS {layers} is not divisible by PP_SIZE {pp}")
    if heads % tp or groups % tp:
        errors.append(f"attention heads {heads} / query groups {groups} are not divisible by TP_SIZE {tp}")
    if cp > 1 and seq % (2 * cp):
        errors.append(f"SEQ_LENGTH {seq} must be divisible by 2*CP_SIZE={2 * cp}")
    if seq != max_pos:
        errors.append(f"SEQ_LENGTH {seq} differs from MAX_POSITION_EMBEDDINGS {max_pos}")

    if env.get("TRAIN_ITERS"):
        samples = int(env["TRAIN_ITERS"]) * gbs
    elif env.get("TRAIN_SAMPLES"):
        samples = int(env["TRAIN_SAMPLES"])
        if samples % gbs:
            errors.append(f"TRAIN_SAMPLES {samples} is not a multiple of GLOBAL_BATCH_SIZE {gbs}")
    else:
        samples = 0
        errors.append("neither TRAIN_ITERS nor TRAIN_SAMPLES is set")
    summary["tokens_b"] = samples * seq / 1e9

    if env.get("RESUME_ITER"):
        expected = int(env["RESUME_ITER"]) * gbs
        consumed = env.get("RESUME_CONSUMED_SAMPLES")
        if consumed is None:
            errors.append("RESUME_ITER is set but RESUME_CONSUMED_SAMPLES is not")
        elif int(consumed) != expected:
            errors.append(f"RESUME_CONSUMED_SAMPLES {consumed} != RESUME_ITER*GBS={expected}")
        offset = env.get("MEGATRON_DATASET_SAMPLE_OFFSET")
        if offset is not None and consumed is not None and int(offset) != int(consumed):
            errors.append(f"MEGATRON_DATASET_SAMPLE_OFFSET {offset} != RESUME_CONSUMED_SAMPLES {consumed}")

    return errors, summary


def check_directory(stage_dir: Path) -> int:
    failures = 0
    names: dict[str, Path] = {}
    print(f"== {stage_dir.name}")
    for path in sorted((stage_dir / "configs" / "stages").glob("*.env")):
        rel = path.relative_to(ROOT)
        try:
            env = load_env(path)
        except RuntimeError as exc:
            print(f"  FAIL {rel}: cannot source: {exc}")
            failures += 1
            continue

        errors, summary = check_stage(env)
        exp_name = env.get("EXP_NAME", "")
        if exp_name in names:
            errors.append(f"EXP_NAME {exp_name} is also used by {names[exp_name].name}")
        names[exp_name] = path

        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in path.read_text(encoding="utf-8"):
                errors.append(f"hard-coded path fragment {fragment!r} in file")

        status = "FAIL" if errors else "ok  "
        detail = ", ".join(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}" for key, value in summary.items())
        print(f"  {status} {path.name}: {detail}")
        for error in errors:
            print(f"       - {error}")
        failures += bool(errors)
    return failures


def main() -> int:
    failures = sum(check_directory(ROOT / name) for name in STAGE_DIRS)
    print(f"\n{failures} stage(s) failed" if failures else "\nall stage configs passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
