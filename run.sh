#!/usr/bin/env bash
# laya-spark -- a Jev-compatible System One server over the Laya decision checkpoints.
#
#   ./run.sh setup     create .venv, install torch (cu130) and the deps, fetch the checkpoints
#   ./run.sh serve     start the server detached on $PORT, with a pid file and a log
#   ./run.sh stop      stop it
#   ./run.sh status    /healthz, /readyz and /v1/models
#   ./run.sh smoke     a handful of real requests across all three checkpoints
#   ./run.sh bench     latency and throughput against a running server
#
# Every setting is an environment variable; see the table in README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VERSION="$(cat "$HERE/VERSION")"

PORT="${PORT:-8010}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
LAYA_MODELS="${LAYA_MODELS:-english,multilingual,typed-decisions}"
LAYA_MODE="${LAYA_MODE:-eager}"
LAYA_API_KEY="${LAYA_API_KEY:-}"
LAYA_BATCH_WAIT_MS="${LAYA_BATCH_WAIT_MS:-2}"
LAYA_MAX_BATCH="${LAYA_MAX_BATCH:-64}"
LAYA_MAX_QUEUE="${LAYA_MAX_QUEUE:-256}"
LAYA_GRAPH_MAX_MARKERS="${LAYA_GRAPH_MAX_MARKERS:-32}"
MODELS_DIR="${MODELS_DIR:-$HERE/models}"
LAYA_MODELS_DIR="${LAYA_MODELS_DIR:-$MODELS_DIR/laya}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
WORKERS="${WORKERS:-1}"

VENV="$HERE/.venv"
PY="$VENV/bin/python"
LOGS="$HERE/logs"
PIDFILE="$LOGS/laya.pid"
BASE="http://127.0.0.1:$PORT"

export PORT HOST DEVICE LAYA_MODELS LAYA_MODE LAYA_API_KEY LAYA_BATCH_WAIT_MS LAYA_MAX_BATCH \
       LAYA_MAX_QUEUE LAYA_GRAPH_MAX_MARKERS LAYA_MODELS_DIR
export LAYA_SPARK_VERSION="$VERSION"

banner() { echo "laya-spark $VERSION -- $1"; }

need_venv() {
  [ -x "$PY" ] || { echo "no .venv yet; run ./run.sh setup" >&2; exit 1; }
}

cmd_setup() {
  banner "setup"
  mkdir -p "$LOGS" "$MODELS_DIR"

  if [ ! -x "$PY" ]; then
    local base
    base="$(command -v python3.12 || command -v python3)"
    echo "creating .venv with $base ($("$base" --version 2>&1))"
    "$base" -m venv "$VENV"
  fi

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
  echo "fetching the three checkpoints into $LAYA_MODELS_DIR (~2.4 GB)"
  local hf="$VENV/bin/hf"
  [ -x "$hf" ] || hf="$(command -v hf)"
  "$hf" download convaiinnovations/laya \
      --local-dir "$LAYA_MODELS_DIR" \
      --exclude "assets/*" "eval/*" "*.png" "*.jpg"
  du -sh "$LAYA_MODELS_DIR" 2>/dev/null || true
  banner "setup done"
}

cmd_serve() {
  need_venv
  banner "serve on $HOST:$PORT (mode=$LAYA_MODE device=$DEVICE models=$LAYA_MODELS)"
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
  banner "smoke against $BASE"
  exec "$PY" "$HERE/tools/smoke.py" --base "$BASE"
}

cmd_bench() {
  need_venv
  banner "bench against $BASE"
  exec "$PY" "$HERE/bench/bench.py" --base "$BASE" "$@"
}

cmd_equivalence() {
  need_venv
  banner "equivalence"
  exec "$PY" "$HERE/tools/equivalence.py" "$@"
}

cmd_test() {
  need_venv
  exec "$PY" -m pytest "$HERE/tests" -q "$@"
}

usage() {
  sed -n '2,12p' "$HERE/run.sh" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  setup)             shift; cmd_setup "$@" ;;
  serve)             shift; cmd_serve "$@" ;;
  serve-foreground)  shift; cmd_serve_foreground "$@" ;;
  stop)              shift; cmd_stop "$@" ;;
  restart)           shift; cmd_stop; cmd_serve "$@" ;;
  status)            shift; cmd_status "$@" ;;
  smoke)             shift; cmd_smoke "$@" ;;
  bench)             shift; cmd_bench "$@" ;;
  equivalence)       shift; cmd_equivalence "$@" ;;
  test)              shift; cmd_test "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command: $1" >&2; echo; usage; exit 2 ;;
esac
