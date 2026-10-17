#!/usr/bin/env python3
"""Launch one ZGCM-7B stage with torchrun.

The stage env file is sourced by bash, as before, to resolve defaults and
the Megatron virtual environment. The torchrun arguments are then assembled
here, so the command can be printed with --dry-run and reviewed without
starting training. Exit codes and emitted arguments match the previous
launch_train.sh.

usage: launch_stage.py [--dry-run] configs/stages/<stage>.env
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_USAGE = 2
EXIT_MISSING_ASSET = 3
EXIT_CHECKPOINT_MISMATCH = 4


class LaunchError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def resolve_env(env_file: Path) -> dict[str, str]:
    """Source the stage file and the venv activate script, as launch_train.sh did."""
    script = (
        'set -a; source "$1"; '
        'if [[ -f "${MEGATRON_VENV:-}/bin/activate" ]]; then source "${MEGATRON_VENV}/bin/activate"; fi; '
        "env -0"
    )
    proc = subprocess.run(
        ["bash", "-c", script, "bash", str(env_file)],
        env={**os.environ, "ZGCM_REPO_ROOT": str(REPO_ROOT)},
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LaunchError(proc.stderr.decode(errors="replace").strip() or f"cannot source {env_file}", EXIT_USAGE)
    resolved: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if item:
            key, _, value = item.decode(errors="replace").partition("=")
            resolved[key] = value
    return resolved


def need(env: dict[str, str], name: str) -> str:
    """Mirror ``${NAME}`` under ``set -u``: the variable must exist, it may be empty."""
    if name not in env:
        raise LaunchError(f"required variable not set: {name}", EXIT_USAGE)
    return env[name]


def opt(env: dict[str, str], name: str, default: str = "") -> str:
    """Mirror ``${NAME:-default}``: unset or empty falls back to the default."""
    return env.get(name) or default


def on(env: dict[str, str], name: str, default: str = "0") -> bool:
    return opt(env, name, default) == "1"


def output_paths(env: dict[str, str]) -> dict[str, str]:
    train_root = need(env, "TRAIN_ROOT")
    exp = need(env, "EXP_NAME")
    save = opt(env, "SAVE_CHECKPOINT_DIR", f"{train_root}/results/{exp}/checkpoints")
    return {
        "save": save,
        "load": opt(env, "LOAD_CHECKPOINT_DIR", save),
        "tensorboard": opt(env, "TENSORBOARD_DIR", f"{train_root}/results/{exp}/tensorboard"),
        "data_cache": opt(env, "DATA_CACHE_DIR", f"{train_root}/results/{exp}/data-cache"),
        "log": opt(env, "LOG_DIR", f"{train_root}/logs"),
    }


def check_assets(env: dict[str, str], paths: dict[str, str]) -> None:
    venv = Path(need(env, "MEGATRON_VENV"))
    if not (venv / "bin" / "activate").is_file():
        raise LaunchError(f"virtual environment not found: {venv}", EXIT_MISSING_ASSET)
    for path in (need(env, "MEGATRON_LM_ROOT"), need(env, "TOKENIZER_PATH")):
        if not Path(path).exists():
            raise LaunchError(f"required path not found: {path}", EXIT_MISSING_ASSET)
    pretrained = opt(env, "PRETRAINED_CHECKPOINT")
    if pretrained and not Path(pretrained).is_dir():
        raise LaunchError(f"pretrained checkpoint not found: {pretrained}", EXIT_MISSING_ASSET)
    data_args = opt(env, "DATA_ARGS_PATH")
    if data_args:
        if not Path(data_args).is_file():
            raise LaunchError(f"data args not found: {data_args}", EXIT_MISSING_ASSET)
    else:
        prefix = need(env, "REAL_DATA_PREFIX")
        for suffix in (".idx", ".bin"):
            if not Path(prefix + suffix).is_file():
                raise LaunchError(f"indexed dataset not found: {prefix}.{{idx,bin}}", EXIT_MISSING_ASSET)


def check_resume(env: dict[str, str], paths: dict[str, str]) -> None:
    if not on(env, "REQUIRE_LOAD_CHECKPOINT"):
        return
    marker = Path(paths["load"]) / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise LaunchError(f"checkpoint marker not found: {marker}", EXIT_CHECKPOINT_MISMATCH)
    resume_iter = opt(env, "RESUME_ITER")
    if resume_iter:
        found = "".join(ch for ch in marker.read_text(encoding="utf-8") if ch.isdigit())
        if found != resume_iter:
            raise LaunchError(
                f"checkpoint iteration {found} != expected {resume_iter}", EXIT_CHECKPOINT_MISMATCH
            )


def runtime_environment(env: dict[str, str]) -> dict[str, str]:
    """Exports that launch_train.sh added on top of the sourced stage file."""
    megatron = need(env, "MEGATRON_LM_ROOT")
    parts = [opt(env, "TRANSFORMER_ENGINE_HOME"), megatron, opt(env, "FLASH_ATTN_V3_EGG"), opt(env, "PYTHONPATH")]
    updated = dict(env)
    updated["MEGATRON_HOME"] = megatron
    updated["PYTHONPATH"] = ":".join(part for part in parts if part)
    updated["NVTE_FRAMEWORK"] = "pytorch"
    updated["TOKENIZERS_PARALLELISM"] = "false"
    updated["PYTHONUNBUFFERED"] = "1"
    if on(env, "NO_DATA_SHUFFLE", "0"):
        updated["MEGATRON_GPT_DATASET_SEQUENTIAL"] = "1"
    else:
        updated.pop("MEGATRON_GPT_DATASET_SEQUENTIAL", None)
    return updated


def build_command(env: dict[str, str], paths: dict[str, str]) -> list[str]:
    distributed = [
        "--nproc_per_node", need(env, "GPUS_PER_NODE"),
        "--nnodes", need(env, "NUM_NODES"),
        "--node_rank", need(env, "NODE_RANK"),
        "--master_addr", need(env, "MASTER_ADDR"),
        "--master_port", need(env, "MASTER_PORT"),
    ]

    model = [
        "--use-mcore-models",
        "--num-layers", need(env, "NUM_LAYERS"),
        "--hidden-size", need(env, "HIDDEN_SIZE"),
        "--ffn-hidden-size", need(env, "FFN_HIDDEN_SIZE"),
        "--num-attention-heads", need(env, "NUM_ATTENTION_HEADS"),
        "--group-query-attention",
        "--num-query-groups", need(env, "NUM_QUERY_GROUPS"),
        "--kv-channels", need(env, "KV_CHANNELS"),
        "--seq-length", need(env, "SEQ_LENGTH"),
        "--max-position-embeddings", need(env, "MAX_POSITION_EMBEDDINGS"),
        "--position-embedding-type", need(env, "POSITION_EMBEDDING_TYPE"),
        "--rotary-base", need(env, "ROTARY_BASE"),
        "--rotary-percent", need(env, "ROTARY_PERCENT"),
        "--window-size", need(env, "WINDOW_SIZE"),
        "--window-attn-skip-freq", need(env, "WINDOW_ATTN_SKIP_FREQ"),
        "--attention-backend", need(env, "ATTENTION_BACKEND"),
        "--norm-epsilon", need(env, "NORM_EPSILON"),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--swiglu",
        "--normalization", "RMSNorm",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--make-vocab-size-divisible-by", "128",
        "--transformer-impl", need(env, "TRANSFORMER_IMPL"),
        "--fp8-format", need(env, "FP8_FORMAT"),
        "--fp8-recipe", need(env, "FP8_RECIPE"),
        "--fp8-amax-compute-algo", need(env, "FP8_AMAX_COMPUTE_ALGO"),
        "--fp8-amax-history-len", need(env, "FP8_AMAX_HISTORY_LEN"),
    ]
    if on(env, "ATTENTION_OUTPUT_GATE"):
        model.append("--attention-output-gate")
    if on(env, "ATTENTION_OUTPUT_GATE_ONLY_SWA"):
        model.append("--attention-output-gate-only-swa")
    if on(env, "QK_LAYERNORM"):
        model.append("--qk-layernorm")
    if on(env, "NO_ROPE_FUSION"):
        model.append("--no-rope-fusion")
    if on(env, "FP8_PARAM_GATHER"):
        model.append("--fp8-param-gather")

    training = [
        "--micro-batch-size", need(env, "MICRO_BATCH_SIZE"),
        "--global-batch-size", need(env, "GLOBAL_BATCH_SIZE"),
        "--optimizer", need(env, "OPTIMIZER"),
        "--lr", need(env, "LR"),
        "--min-lr", need(env, "MIN_LR"),
        "--lr-decay-style", need(env, "LR_DECAY_STYLE"),
        "--weight-decay", need(env, "WEIGHT_DECAY"),
        "--adam-beta1", need(env, "ADAM_BETA1"),
        "--adam-beta2", need(env, "ADAM_BETA2"),
        "--clip-grad", need(env, "CLIP_GRAD"),
        "--seed", need(env, "SEED"),
        "--bf16",
        "--calculate-per-token-loss",
        "--muon-momentum", need(env, "MUON_MOMENTUM"),
        "--muon-scale-mode", need(env, "MUON_SCALE_MODE"),
        "--muon-fp32-matmul-prec", need(env, "MUON_FP32_MATMUL_PREC"),
        "--muon-coefficient-type", need(env, "MUON_COEFFIC_TYPE"),
        "--muon-num-ns-steps", need(env, "MUON_NUM_NS_STEPS"),
        "--muon-tp-mode", need(env, "MUON_TP_MODE"),
        "--muon-extra-scale-factor", need(env, "MUON_EXTRA_SCALE_FACTOR"),
        "--muon-scalar-optimizer", need(env, "MUON_SCALAR_OPTIMIZER"),
    ]
    if opt(env, "TRAIN_ITERS"):
        training += ["--train-iters", env["TRAIN_ITERS"], "--lr-warmup-iters", opt(env, "LR_WARMUP_ITERS", "0")]
    else:
        training += [
            "--train-samples", need(env, "TRAIN_SAMPLES"),
            "--lr-warmup-samples", need(env, "LR_WARMUP_SAMPLES"),
            "--lr-decay-samples", need(env, "LR_DECAY_SAMPLES"),
        ]
    if on(env, "USE_DISTRIBUTED_OPTIMIZER", "1"):
        training.append("--use-distributed-optimizer")
    if on(env, "ENABLE_OVERLAP_GRAD_REDUCE", "1"):
        training.append("--overlap-grad-reduce")
    if on(env, "ENABLE_OVERLAP_PARAM_GATHER"):
        training.append("--overlap-param-gather")
    if on(env, "ENABLE_CROSS_ENTROPY_LOSS_FUSION", "1"):
        training.append("--cross-entropy-loss-fusion")
        if opt(env, "CROSS_ENTROPY_FUSION_IMPL"):
            training += ["--cross-entropy-fusion-impl", env["CROSS_ENTROPY_FUSION_IMPL"]]
    if on(env, "OVERRIDE_OPT_PARAM_SCHEDULER"):
        training.append("--override-opt-param-scheduler")
    if on(env, "ENABLE_RECOMPUTE"):
        granularity = opt(env, "RECOMPUTE_GRANULARITY", "selective")
        if granularity == "selective":
            training += ["--recompute-activations", "--recompute-granularity", "selective"]
        else:
            training += ["--recompute-granularity", granularity]
            if opt(env, "RECOMPUTE_METHOD"):
                training += ["--recompute-method", env["RECOMPUTE_METHOD"]]
            if opt(env, "RECOMPUTE_NUM_LAYERS"):
                training += ["--recompute-num-layers", env["RECOMPUTE_NUM_LAYERS"]]

    tensor_parallel = need(env, "TP_SIZE")
    parallel = [
        "--tensor-model-parallel-size", tensor_parallel,
        "--pipeline-model-parallel-size", need(env, "PP_SIZE"),
        "--context-parallel-size", need(env, "CP_SIZE"),
    ]
    if tensor_parallel != "1":
        parallel.append("--sequence-parallel")
    if opt(env, "CP_COMM_TYPE"):
        parallel += ["--cp-comm-type", env["CP_COMM_TYPE"]]

    data = [
        "--vocab-size", need(env, "VOCAB_SIZE"),
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", need(env, "TOKENIZER_PATH"),
        "--data-cache-path", paths["data_cache"],
        "--split", opt(env, "DATA_SPLIT", "100,0,0"),
        "--no-create-attention-mask-in-dataloader",
        "--num-workers", opt(env, "DATA_NUM_WORKERS", "4"),
        "--num-dataset-builder-threads", opt(env, "NUM_DATASET_BUILDER_THREADS", "8"),
    ]
    if opt(env, "DATA_ARGS_PATH"):
        data += ["--data-args-path", env["DATA_ARGS_PATH"]]
    else:
        data += ["--data-path", need(env, "REAL_DATA_PREFIX")]

    logging = [
        "--log-interval", opt(env, "LOG_INTERVAL", "10"),
        "--save-interval", opt(env, "SAVE_INTERVAL", "1000"),
        "--eval-interval", opt(env, "EVAL_INTERVAL", "100000"),
        "--eval-iters", opt(env, "EVAL_ITERS", "0"),
        "--log-throughput",
        "--ckpt-format", "torch_dist",
        "--distributed-timeout-minutes", "120",
        "--tensorboard-dir", paths["tensorboard"],
        "--save", paths["save"],
        "--load", paths["load"],
        "--rerun-mode", opt(env, "RERUN_MODE", "disabled"),
    ]
    if opt(env, "PRETRAINED_CHECKPOINT"):
        logging += ["--pretrained-checkpoint", env["PRETRAINED_CHECKPOINT"]]

    return [
        need(env, "TORCHRUN_BIN"),
        *distributed,
        need(env, "TRAIN_PROGRAM"),
        *model,
        *training,
        *parallel,
        *data,
        *logging,
    ]


def ensure_directories(paths: dict[str, str]) -> None:
    for key in ("save", "tensorboard", "data_cache", "log"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def run(command: list[str], env: dict[str, str], cwd: str, log_path: Path) -> int:
    """Run torchrun, teeing its output to stdout and the run log, and forward signals."""
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def forward(signum: int, _frame: object) -> None:
        proc.send_signal(signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        assert proc.stdout is not None
        with log_path.open("ab") as log:
            for line in iter(proc.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log.write(line)
                log.flush()
        return proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch one ZGCM-7B pretrain stage.")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the torchrun command only")
    parser.add_argument("env_file", type=Path, help="configs/stages/<stage>.env")
    args = parser.parse_args(argv)

    try:
        if not args.env_file.is_file():
            raise LaunchError(f"stage file not found: {args.env_file}", EXIT_USAGE)
        env = resolve_env(args.env_file.resolve())
        for name in ("MEGATRON_LM_ROOT", "MEGATRON_VENV", "TORCHRUN_BIN", "TOKENIZER_PATH", "TRAIN_ROOT",
                     "NUM_NODES", "GPUS_PER_NODE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
            if not env.get(name):
                raise LaunchError(f"required variable is empty: {name}", EXIT_USAGE)

        paths = output_paths(env)
        check_assets(env, paths)
        check_resume(env, paths)
        env = runtime_environment(env)
        command = build_command(env, paths)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code

    run_tag = f"{env['EXP_NAME']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"run_tag={run_tag} env={args.env_file.resolve()} save={paths['save']} load={paths['load']}")
    if args.dry_run:
        print(shlex.join(command))
        return 0

    ensure_directories(paths)
    log_path = Path(paths["log"]) / f"{run_tag}.log"
    return run(command, env, env["MEGATRON_LM_ROOT"], log_path)


if __name__ == "__main__":
    sys.exit(main())
