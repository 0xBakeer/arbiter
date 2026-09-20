# Mounting the playground on the server

`playground/index.html` is one self-contained file. The server serves it at `GET /`, same origin
as the API, so the page needs no configuration at all.

Two lines inside `create_app()` in `server/app.py`, next to the other endpoints:

```python
from fastapi.responses import FileResponse  # with the other imports

@app.get("/", include_in_schema=False)
async def playground():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "playground", "index.html"),
                        media_type="text/html")
```

The container image must contain `playground/` next to `server/` (add it to the `COPY` in the
Dockerfile if the image copies directories one by one).

## Same-origin is the supported mode

The server sets no CORS headers (`server/app.py` has no `CORSMiddleware`), so a copy of the page
opened from disk cannot call the API: the browser blocks the request before it leaves. That is
deliberate; adding permissive CORS to an inference server that may carry an API key is the wrong
default for a public recipe.

For work on the page itself without the server, or without a GPU:

```
python3 playground/serve_stub.py                     # stub answers on http://localhost:8011
python3 playground/serve_stub.py --proxy http://localhost:8010
                                                     # real answers, page still same-origin
```

The page detects how it was opened. Served over HTTP it defaults the base URL to its own origin;
opened from disk it defaults to `http://localhost:8010` and says in the server settings panel
that cross-origin serving is required. The base URL is remembered in `localStorage`; the API
key only if the "remember" box is ticked.

## What the page uses

- `POST /v1/systemone` with `state`, `questions` and an optional `model`.
- `GET /readyz` every 10 s for the header dot (`status: ready` plus `models[]` and `version`).
- The `routing.model` / `routing.reason` / `routing.detection` and `latency_ms` extras in the
  response, and `usage.input_tokens`. Everything degrades gracefully if they are absent.
- `/docs` is linked from the footer; it is FastAPI's own reference and needs no work.
