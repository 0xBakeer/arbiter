#!/usr/bin/env bash
# arbiter -- a Jev-compatible System One server over the Laya decision checkpoints.
#
#   ./run.sh setup     create .venv, install torch (cu130 on Linux, PyPI on macOS), the deps,
#                      and fetch the checkpoints. ARBITER_ENGINE=laya_mlx adds the Apple-silicon
#                      MLX engine: laya-mlx, mlx, and the three converted checkpoints
#   ./run.sh serve     start the server detached on $PORT, with a pid file and a log
#   ./run.sh stop      stop it
#   ./run.sh status    /healthz, /readyz and /v1/models
#   ./run.sh smoke     a handful of real requests across all three checkpoints
#   ./run.sh examples  run examples/support_triage.py against a running server
#   ./run.sh bench     latency and throughput against a running server
#   ./run.sh equivalence   compare every dtype/mode path against the SDK reference
#   ./run.sh test      both pytest suites: tests/ and examples/tests/
#
# Every setting is an environment variable; see the table in README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VERSION="$(cat "$HERE/VERSION")"

UNAME_S="$(uname -s)"

PORT="${PORT:-8010}"
HOST="${HOST:-0.0.0.0}"
# `auto` is resolved once by the engine: cuda if there is a CUDA device, else mps on Apple
# silicon, else cpu. Set it to pin one; /readyz reports which one was taken.
ARBITER_DEVICE="${ARBITER_DEVICE:-auto}"
ARBITER_MODELS="${ARBITER_MODELS:-english,multilingual,typed-decisions}"
ARBITER_MODE="${ARBITER_MODE:-eager}"
# Which package under engines/ answers. `laya` is torch and runs everywhere; `laya_mlx` is the
# MLX port and is Apple silicon only, opt-in, and has its own weights and its own dtype default.
ARBITER_ENGINE="${ARBITER_ENGINE:-laya}"
# `autocast` is a CUDA mode and the MLX engine has no equivalent of it; fp32 is what that lane
# ships, for the reason its equivalence table gives.
if [ "$ARBITER_ENGINE" = laya_mlx ]; then
  ARBITER_DTYPE="${ARBITER_DTYPE:-fp32}"
else
  ARBITER_DTYPE="${ARBITER_DTYPE:-autocast}"
fi
ARBITER_API_KEY="${ARBITER_API_KEY:-}"
ARBITER_BATCH_WAIT_MS="${ARBITER_BATCH_WAIT_MS:-2}"
ARBITER_MAX_BATCH="${ARBITER_MAX_BATCH:-64}"
ARBITER_MAX_QUEUE="${ARBITER_MAX_QUEUE:-256}"
ARBITER_GRAPH_MAX_MARKERS="${ARBITER_GRAPH_MAX_MARKERS:-32}"
# laya_mlx only: how much freed GPU memory MLX may keep for reuse. Unbounded is its own default
# and the wrong one for a server; 0 restores it.
ARBITER_MLX_CACHE_MB="${ARBITER_MLX_CACHE_MB:-1024}"
MODELS_DIR="${MODELS_DIR:-$HERE/models}"
if [ "$ARBITER_ENGINE" = laya_mlx ]; then
  ARBITER_MODELS_DIR="${ARBITER_MODELS_DIR:-$MODELS_DIR/laya-mlx}"
else
  ARBITER_MODELS_DIR="${ARBITER_MODELS_DIR:-$MODELS_DIR/laya}"
fi
# The CUDA wheels are Linux and Windows only. On macOS the plain PyPI wheels are the MPS-enabled
# arm64 builds, so the index is left empty there and pip takes its default.
if [ "$UNAME_S" = Darwin ]; then
  TORCH_INDEX="${TORCH_INDEX-}"
else
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
fi
WORKERS="${WORKERS:-1}"

VENV="$HERE/.venv"
PY="$VENV/bin/python"
LOGS="$HERE/logs"
PIDFILE="$LOGS/arbiter.pid"
BASE="http://127.0.0.1:$PORT"

export PORT HOST ARBITER_DEVICE ARBITER_MODELS ARBITER_MODE ARBITER_DTYPE ARBITER_API_KEY ARBITER_BATCH_WAIT_MS ARBITER_MAX_BATCH \
       ARBITER_MAX_QUEUE ARBITER_GRAPH_MAX_MARKERS ARBITER_MODELS_DIR ARBITER_ENGINE \
       ARBITER_MLX_CACHE_MB
export ARBITER_VERSION="$VERSION"

banner() { echo "arbiter $VERSION -- $1"; }

need_venv() {
  [ -x "$PY" ] || { echo "no .venv yet; run ./run.sh setup" >&2; exit 1; }
}

# torch routes a handful of eager ops (ModernBERT's RoPE among them) through Triton, and Triton
# JIT-compiles a small C extension the first time one runs. That needs the interpreter's
# development headers on disk. A distribution python without python3-dev does not have them, and
# the failure lands on the first forward pass as "Python.h: No such file or directory".
HEADER_PROBE='import os,sys,sysconfig; p=sysconfig.get_paths(scheme=sysconfig.get_default_scheme())["include"]; sys.exit(0 if os.path.exists(os.path.join(p,"Python.h")) else 1)'

# Candidates in preference order. A uv-managed CPython ships its own headers, which a
# distribution python3.12 without python3-dev does not, so it is tried first.
uv_pythons() {
  ls -d "$HOME"/.local/share/uv/python/cpython-3.1[2-9]*/bin/python3.1[2-9] 2>/dev/null | sort -r
}

pick_python() {
  local fallback=""
  for c in ${PYTHON:-} $(uv_pythons) python3.12 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    [ -n "$fallback" ] || fallback="$c"
    if "$c" -c "$HEADER_PROBE" 2>/dev/null; then echo "$c"; return 0; fi
  done
  [ -n "$fallback" ] && { echo "$fallback"; return 0; }
  echo "no python3 found" >&2
  return 1
}

native_jit_guard() {
  # If the headers are missing anyway, take torch's plain eager kernels instead of dying.
  if [ -z "${TORCH_DISABLE_NATIVE_JIT:-}" ] && ! "$PY" -c "$HEADER_PROBE" 2>/dev/null; then
    export TORCH_DISABLE_NATIVE_JIT=1
    echo "note: no CPython headers for this interpreter, so torch's Triton eager kernels cannot"
    echo "      build here; running with TORCH_DISABLE_NATIVE_JIT=1"
  fi
}

mac_graphs_guard() {
  # CUDA graphs are CUDA graphs. On a Mac the flag can only be a mistake, and it is worth saying
  # so here rather than as a load failure inside uvicorn thirty seconds later.
  if [ "$UNAME_S" = Darwin ] && [ "$ARBITER_MODE" = graphs ]; then
    echo "ARBITER_MODE=graphs is CUDA graph capture and there is no CUDA here; use eager" >&2
    exit 2
  fi
}

mlx_guard() {
  # The mlx wheels are arm64 macOS only and there is nothing to fall back to. The engine refuses
  # the same machine with the same sentence at startup; saying it here saves a setup that would
  # install nothing usable.
  if [ "$ARBITER_ENGINE" = laya_mlx ] && { [ "$UNAME_S" != Darwin ] || [ "$(uname -m)" != arm64 ]; }; then
    echo "ARBITER_ENGINE=laya_mlx needs MLX, which is Apple silicon only; this is $UNAME_S/$(uname -m)" >&2
    exit 2
  fi
}

cmd_setup() {
  banner "setup"
  mlx_guard
  mkdir -p "$LOGS" "$MODELS_DIR"

  if [ ! -x "$PY" ]; then
    local base
    base="$(pick_python)"
    echo "creating .venv with $(command -v "$base") ($("$base" --version 2>&1))"
    "$base" -c "$HEADER_PROBE" 2>/dev/null || echo "  (this interpreter has no development headers; see the note at serve time)"
    "$base" -m venv "$VENV"
  fi

  native_jit_guard
  "$PY" -m pip install --upgrade pip wheel >/dev/null
  # torch first and from the CUDA index, so the rest resolve against the build that is staying.
  # torchvision and torchaudio are not installed: nothing here uses them.
  if [ -n "$TORCH_INDEX" ]; then
    "$PY" -m pip install torch --index-url "$TORCH_INDEX"
  else
    "$PY" -m pip install torch
  fi
  # mcp is not needed to serve, but `./run.sh test` covers integrations/mcp over stdio, and a
  # fresh setup should be able to run both suites.
  "$PY" -m pip install \
      "transformers>=5" safetensors huggingface_hub numpy \
      fastapi "uvicorn[standard]" "laya==0.3.4" "mcp>=2" pytest httpx

  echo
  "$PY" - <<'PYCHECK'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
elif torch.backends.mps.is_available():
    print("device mps, built", torch.backends.mps.is_built())
PYCHECK

  echo
  # On the MLX lane `ARBITER_MODELS_DIR` is the converted tree, but the upstream one is still
  # wanted: it is what `ARBITER_ENGINE=laya` serves on the same checkout and what the
  # equivalence gate's reference -- `laya.Agent` under torch -- loads.
  local upstream_dir="$ARBITER_MODELS_DIR"
  [ "$ARBITER_ENGINE" = laya_mlx ] && upstream_dir="$MODELS_DIR/laya"
  echo "fetching the three checkpoints into $upstream_dir (~2.4 GB)"
  # The hub CLI is called as a module, not as $VENV/bin/hf: that script carries the absolute
  # shebang pip wrote, so a moved or copied .venv dies on it with "bad interpreter".
  # One --exclude per pattern: the flag takes a single value, and extra patterns after it are
  # read as the positional FILENAMES list, which downloads exactly the files you meant to skip.
  "$PY" -m huggingface_hub.cli.hf download convaiinnovations/laya \
      --local-dir "$upstream_dir" \
      --exclude "assets/*" --exclude "eval/*" --exclude "*.png" --exclude "*.jpg"
  du -sh "$upstream_dir" 2>/dev/null || true

  [ "$ARBITER_ENGINE" = laya_mlx ] && setup_mlx
  banner "setup done"
}

setup_mlx() {
  # laya-mlx pulls mlx, tokenizers, huggingface_hub and numpy. Every one of those is either
  # already installed here at the same version or has nothing to do with torch, so this goes in
  # the same .venv -- no sibling .venv-mlx, and `pip check` stays clean either way.
  echo
  echo "installing the MLX engine into the same .venv"
  "$PY" -m pip install "laya-mlx>=0.1"
  "$PY" -m pip check || true
  "$PY" - <<'PYCHECK'
import mlx.core as mx

import laya_mlx

print("laya-mlx", laya_mlx.__version__, "mlx", mx.__version__, "device", mx.default_device())
PYCHECK

  echo
  echo "fetching the three converted checkpoints into $ARBITER_MODELS_DIR (~2.1 GB)"
  # One repo per checkpoint, unlike the upstream bundle, and named here by the checkpoint name
  # the server uses so that `models/laya-mlx/<name>` is what the engine looks for. A here-doc
  # rather than an associative array: macOS ships bash 3.2, which has none.
  while read -r name repo; do
    [ -n "$name" ] || continue
    "$PY" -m huggingface_hub.cli.hf download "$repo" --local-dir "$ARBITER_MODELS_DIR/$name"
  done <<'CHECKPOINTS'
english aac6fef/laya-mlx
multilingual aac6fef/laya-multilingual-mlx
typed-decisions aac6fef/laya-typed-decisions-mlx
CHECKPOINTS
  du -sh "$ARBITER_MODELS_DIR" 2>/dev/null || true
}

cmd_serve() {
  need_venv
  native_jit_guard
  mac_graphs_guard
  mlx_guard
  banner "serve on $HOST:$PORT (engine=$ARBITER_ENGINE mode=$ARBITER_MODE dtype=$ARBITER_DTYPE device=$ARBITER_DEVICE models=$ARBITER_MODELS)"
  mkdir -p "$LOGS"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running as pid $(cat "$PIDFILE")" >&2
    exit 1
  fi
  # setsid is util-linux and macOS does not ship it; nohup and a background job are enough to
  # survive this shell there, and `stop` works off the pid file either way.
  SETSID="$(command -v setsid || true)"
  $SETSID nohup "$PY" -m uvicorn server.app:app \
      --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level info \
      > "$LOGS/serve.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "pid $(cat "$PIDFILE"), log $LOGS/serve.log"
  echo "waiting for /readyz (checkpoints load from disk, ~20-60 s cold)"
  for _ in $(seq 1 180); do
    if curl -fsS "$BASE/readyz" >/dev/null 2>&1; then
      curl -s "$BASE/readyz"; echo; return 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server exited; see $LOGS/serve.log" >&2; tail -20 "$LOGS/serve.log" >&2; exit 1; }
    sleep 1
  done
  echo "timed out waiting for /readyz; see $LOGS/serve.log" >&2
  exit 1
}

cmd_serve_foreground() {
  need_venv
  native_jit_guard
  mac_graphs_guard
  mlx_guard
  banner "serve (foreground) on $HOST:$PORT (engine=$ARBITER_ENGINE)"
  exec "$PY" -m uvicorn server.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
}

cmd_stop() {
  if [ ! -f "$PIDFILE" ]; then echo "not running (no $PIDFILE)"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
    echo "stopped $pid"
  else
    echo "pid $pid is gone"
  fi
  rm -f "$PIDFILE"
}

cmd_status() {
  banner "status"
  echo "-- /healthz";   curl -s "$BASE/healthz"   || true; echo
  echo "-- /readyz";    curl -s "$BASE/readyz"    || true; echo
  echo "-- /v1/models"; curl -s "$BASE/v1/models" || true; echo
}

cmd_smoke() {
  need_venv
  native_jit_guard
  banner "smoke against $BASE"
  exec "$PY" "$HERE/tools/smoke.py" --base "$BASE"
}

cmd_examples() {
  need_venv
  banner "examples against $BASE"
  ARBITER_URL="$BASE" exec "$PY" "$HERE/examples/support_triage.py" "$@"
}

cmd_bench() {
  need_venv
  native_jit_guard
  banner "bench against $BASE"
  exec "$PY" "$HERE/bench/bench.py" --base "$BASE" "$@"
}

cmd_equivalence() {
  need_venv
  native_jit_guard
  banner "equivalence ($ARBITER_ENGINE)"
  if [ "$ARBITER_ENGINE" = laya_mlx ]; then
    mlx_guard
    exec "$PY" "$HERE/tools/equivalence_mlx.py" "$@"
  fi
  exec "$PY" "$HERE/tools/equivalence.py" "$@"
}

cmd_test() {
  need_venv
  native_jit_guard
  # The server suite and the examples suite; neither needs a GPU or a running server.
  exec "$PY" -m pytest "$HERE/tests" "$HERE/examples/tests" -q "$@"
}

usage() {
  # the header comment of this file, up to the first line that is not one
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$HERE/run.sh"
}

case "${1:-}" in
  setup)             shift; cmd_setup "$@" ;;
  serve)             shift; cmd_serve "$@" ;;
  serve-foreground)  shift; cmd_serve_foreground "$@" ;;
  stop)              shift; cmd_stop "$@" ;;
  restart)           shift; cmd_stop; cmd_serve "$@" ;;
  status)            shift; cmd_status "$@" ;;
  smoke)             shift; cmd_smoke "$@" ;;
  examples)          shift; cmd_examples "$@" ;;
  bench)             shift; cmd_bench "$@" ;;
  equivalence)       shift; cmd_equivalence "$@" ;;
  test)              shift; cmd_test "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $1" >&2; echo; usage; exit 2 ;;
esac
