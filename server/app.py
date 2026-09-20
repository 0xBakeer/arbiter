"""HTTP surface: the Jev `/v1/systemone` contract, served by local Laya checkpoints.

The request and response shapes are TypeSafe's Jev API, so a client written against Jev works
against this by changing the base URL and the key. Everything this adds is an extra key
(`routing`, `latency_ms`) or an extra field inside an answer (`confidence`, `action`), which a
Jev client ignores.
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import metrics
from .errors import OptionBudgetError, OverloadedError

VERSION = (os.environ.get("LAYA_SPARK_VERSION") or "").strip() or "0.0.0-dev"

# What a client may put in `model`. "jev-latest" is what a migrating Jev client already sends,
# and it means the same thing here as "let the router choose".
AUTO_NAMES = {"", "auto", "laya", "jev-latest", "jev", "default"}
EXPLICIT_NAMES = {
    "english": "english", "laya-english": "english", "en": "english",
    "multilingual": "multilingual", "laya-multilingual": "multilingual", "multi": "multilingual",
    "typed-decisions": "typed-decisions", "laya-typed-decisions": "typed-decisions",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
}

# The playground is one self-contained page, served from the same origin as the API so that
# it needs no configuration and the server needs no CORS headers.
PLAYGROUND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "playground", "index.html")

QTYPES_ALLOWED = ("noul", "choice", "score")
MAX_CHOICES = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


class ValidationError(ValueError):
    """A request that does not satisfy the Jev contract. Answered with 422."""


class Question(BaseModel):
    type: str
    instructions: Union[str, Dict[str, Any], List[Any]]
    criteria: Optional[Union[Dict[str, Any], List[Any]]] = None


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Question]
    model: Optional[str] = None
    # Not part of Jev; the router's two escape hatches, for callers that already know.
    task: Optional[str] = None
    lang: Optional[str] = None


def resolve_model(name: Optional[str]) -> Optional[str]:
    """Map the `model` field to a checkpoint, or None for 'let the router decide'."""
    key = (name or "").strip().lower()
    if key in AUTO_NAMES:
        return None
    if key in EXPLICIT_NAMES:
        return EXPLICIT_NAMES[key]
    raise ValidationError(
        "unknown model %r; use one of %s, or omit it for automatic routing"
        % (name, sorted(set(EXPLICIT_NAMES) | {"auto", "jev-latest"})))


def validate_questions(questions: Dict[str, Question]) -> None:
    """Enforce the Jev constraints, so a malformed question fails before any model runs."""
    if not questions:
        raise ValidationError("questions must contain at least one question")
    for qid, q in questions.items():
        if q.type not in QTYPES_ALLOWED:
            raise ValidationError("question %r has unknown type %r; expected one of %s"
                                  % (qid, q.type, list(QTYPES_ALLOWED)))
        if q.instructions is None or (isinstance(q.instructions, str) and not q.instructions.strip()):
            raise ValidationError("question %r has empty instructions" % qid)
        crit = q.criteria
        if q.type == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise ValidationError("question %r: noul criteria must be an object with "
                                      "'true' and/or 'false' keys" % qid)
            if isinstance(crit, dict):
                extra = sorted(set(crit) - {"true", "false"})
                if extra:
                    raise ValidationError("question %r: noul criteria may only contain 'true' and "
                                          "'false'; found %s" % (qid, extra))
        elif q.type == "choice":
            if crit is None:
                raise ValidationError("question %r: choice requires criteria" % qid)
            n = len(crit)
            if n < 2:
                raise ValidationError("question %r: choice needs at least 2 options, got %d" % (qid, n))
            if n > MAX_CHOICES:
                raise ValidationError("question %r: choice allows at most %d options, got %d"
                                      % (qid, MAX_CHOICES, n))
        else:
            if not isinstance(crit, list):
                raise ValidationError("question %r: score criteria must be an array of level "
                                      "descriptions" % qid)
            n = len(crit)
            if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
                raise ValidationError("question %r: score needs between %d and %d levels, got %d"
                                      % (qid, MIN_SCORE_LEVELS, MAX_SCORE_LEVELS, n))


def error_body(kind: str, message: str) -> Dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def create_app(engine=None) -> FastAPI:
    """Build the app. Pass `engine` to serve an already-built (or stubbed) engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loading and warming happens off the event loop, so /healthz answers immediately while
        # /readyz stays 503. That is the difference the two probes exist for.
        if app.state.engine is None:
            from .engine import engine_from_env

            def build():
                eng = engine_from_env()
                eng.warm()
                return eng

            app.state.engine = await asyncio.get_running_loop().run_in_executor(None, build)
        app.state.ready = True
        yield
        if app.state.engine is not None:
            app.state.engine.close()

    app = FastAPI(title="laya-spark", version=VERSION, docs_url="/docs", lifespan=lifespan)
    app.state.engine = engine
    app.state.ready = engine is not None
    app.state.api_key = os.environ.get("LAYA_API_KEY") or None
    app.state.metrics = metrics.Registry()

    # -- helpers -----------------------------------------------------------------
    def check_auth(authorization: Optional[str]) -> Optional[JSONResponse]:
        key = app.state.api_key
        if not key:
            return None
        if authorization != "Bearer %s" % key:
            return JSONResponse(status_code=401, content=error_body(
                "authentication_error", "missing or invalid Authorization: Bearer <key>"))
        return None

    # -- endpoints ---------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def playground():
        return FileResponse(PLAYGROUND, media_type="text/html")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": VERSION}

    @app.get("/readyz")
    async def readyz():
        eng = app.state.engine
        if not app.state.ready or eng is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {"status": "ready", "models": eng.loaded, "mode": eng.mode,
                "dtype": eng.dtype_mode, "version": VERSION}

    @app.get("/v1/models")
    async def models():
        eng = app.state.engine
        loaded = eng.loaded if eng is not None else []
        created = int(time.time())
        data = [{"id": "laya-%s" % n, "object": "model", "created": created,
                 "owned_by": "convaiinnovations",
                 "aliases": sorted(a for a, t in EXPLICIT_NAMES.items() if t == n)}
                for n in loaded]
        data.append({"id": "jev-latest", "object": "model", "created": created,
                     "owned_by": "laya-spark",
                     "aliases": sorted(AUTO_NAMES - {""}),
                     "description": "automatic routing across the loaded checkpoints"})
        return {"object": "list", "data": data}

    @app.get("/metrics")
    async def metrics_endpoint():
        eng = app.state.engine
        depth = eng.queue_depth if eng is not None else 0
        return PlainTextResponse(app.state.metrics.render(depth), media_type="text/plain; version=0.0.4")

    async def _systemone(req: Request, authorization: Optional[str]):
        denied = check_auth(authorization)
        if denied is not None:
            app.state.metrics.observe_request("none", 401, 0, 0.0)
            return denied

        started = time.perf_counter()
        eng = app.state.engine
        if eng is None:
            return JSONResponse(status_code=503, content=error_body(
                "overloaded_error", "models are still loading"))

        try:
            payload = await req.json()
        except Exception:
            return JSONResponse(status_code=422, content=error_body(
                "invalid_request_error", "request body is not valid JSON"))

        try:
            parsed = SystemOneRequest(**payload) if isinstance(payload, dict) else None
            if parsed is None:
                raise ValidationError("request body must be a JSON object")
            validate_questions(parsed.questions)
            target = resolve_model(parsed.model)
        except ValidationError as exc:
            app.state.metrics.observe_request("none", 422, 0, time.perf_counter() - started)
            return JSONResponse(status_code=422, content=error_body("invalid_request_error", str(exc)))
        except Exception as exc:                          # pydantic and anything else structural
            app.state.metrics.observe_request("none", 422, 0, time.perf_counter() - started)
            return JSONResponse(status_code=422, content=error_body("invalid_request_error", str(exc)))

        questions = {qid: q.model_dump(exclude_none=False) for qid, q in parsed.questions.items()}
        routing = eng.route(parsed.state, questions, model=target, task=parsed.task, lang=parsed.lang)
        name = routing["model"]

        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(None, eng.infer, name, parsed.state, questions)
        except OptionBudgetError as exc:
            app.state.metrics.observe_request(name, 422, len(questions), time.perf_counter() - started)
            return JSONResponse(status_code=422, content=error_body("invalid_request_error", str(exc)))
        except OverloadedError as exc:
            app.state.metrics.observe_request(name, 529, len(questions), time.perf_counter() - started)
            return JSONResponse(status_code=529, content=error_body("overloaded_error", str(exc)))
        except KeyError:
            app.state.metrics.observe_request(name, 422, len(questions), time.perf_counter() - started)
            return JSONResponse(status_code=422, content=error_body(
                "invalid_request_error",
                "checkpoint %r is not loaded on this server; loaded: %s" % (name, eng.loaded)))

        elapsed = time.perf_counter() - started
        app.state.metrics.observe_request(name, 200, len(questions), elapsed)
        app.state.metrics.observe_batch(out.get("batch_rows", len(questions)))
        return {
            "model": "laya-%s" % name,
            "answers": out["answers"],
            "usage": {"input_tokens": out["input_tokens"], "output_tokens": 0},
            "routing": routing,
            "latency_ms": round(elapsed * 1000, 2),
        }

    @app.post("/v1/systemone")
    async def systemone(req: Request, authorization: Optional[str] = Header(default=None)):
        return await _systemone(req, authorization)

    @app.post("/v1/predict")
    async def predict(req: Request, authorization: Optional[str] = Header(default=None)):
        return await _systemone(req, authorization)

    return app


app = create_app()
