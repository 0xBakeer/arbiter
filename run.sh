#!/usr/bin/env bash
# arbiter -- a Jev-compatible System One server over the Laya decision checkpoints.
#
#   ./run.sh setup     create .venv, install torch (cu130) and the deps, fetch the checkpoints
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

PORT="${PORT:-8010}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
ARBITER_MODELS="${ARBITER_MODELS:-english,multilingual,typed-decisions}"
ARBITER_MODE="${ARBITER_MODE:-eager}"
ARBITER_DTYPE="${ARBITER_DTYPE:-autocast}"
ARBITER_API_KEY="${ARBITER_API_KEY:-}"
ARBITER_BATCH_WAIT_MS="${ARBITER_BATCH_WAIT_MS:-2}"
ARBITER_MAX_BATCH="${ARBITER_MAX_BATCH:-64}"
ARBITER_MAX_QUEUE="${ARBITER_MAX_QUEUE:-256}"
ARBITER_GRAPH_MAX_MARKERS="${ARBITER_GRAPH_MAX_MARKERS:-32}"
MODELS_DIR="${MODELS_DIR:-$HERE/models}"
ARBITER_MODELS_DIR="${ARBITER_MODELS_DIR:-$MODELS_DIR/laya}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
WORKERS="${WORKERS:-1}"

VENV="$HERE/.venv"
PY="$VENV/bin/python"
LOGS="$HERE/logs"
PIDFILE="$LOGS/arbiter.pid"
BASE="http://127.0.0.1:$PORT"

export PORT HOST DEVICE ARBITER_MODELS ARBITER_MODE ARBITER_DTYPE ARBITER_API_KEY ARBITER_BATCH_WAIT_MS ARBITER_MAX_BATCH \
       ARBITER_MAX_QUEUE ARBITER_GRAPH_MAX_MARKERS ARBITER_MODELS_DIR
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

cmd_setup() {
  banner "setup"
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
  "$PY" -m pip install torch --index-url "$TORCH_INDEX"
  "$PY" -m pip install \
      "transformers>=5" safetensors huggingface_hub numpy \
      fastapi "uvicorn[standard]" "laya==0.3.4" pytest httpx

  echo
  "$PY" - <<'PYCHECK'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PYCHECK

  echo
  echo "fetching the three checkpoints into $ARBITER_MODELS_DIR (~2.4 GB)"
  local hf="$VENV/bin/hf"
  [ -x "$hf" ] || hf="$(command -v hf)"
  # One --exclude per pattern: the flag takes a single value, and extra patterns after it are
  # read as the positional FILENAMES list, which downloads exactly the files you meant to skip.
  "$hf" download convaiinnovations/laya \
      --local-dir "$ARBITER_MODELS_DIR" \
      --exclude "assets/*" --exclude "eval/*" --exclude "*.png" --exclude "*.jpg"
  du -sh "$ARBITER_MODELS_DIR" 2>/dev/null || true
  banner "setup done"
}

cmd_serve() {
  need_venv
  native_jit_guard
  banner "serve on $HOST:$PORT (mode=$ARBITER_MODE dtype=$ARBITER_DTYPE device=$DEVICE models=$ARBITER_MODELS)"
  mkdir -p "$LOGS"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running as pid $(cat "$PIDFILE")" >&2
    exit 1
  fi
  setsid nohup "$PY" -m uvicorn server.app:app \
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
  banner "serve (foreground) on $HOST:$PORT"
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
  banner "equivalence"
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
