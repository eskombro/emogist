"""emogist API and local demo page.

    uvicorn app:app --app-dir demo --host 127.0.0.1 --port 8000

Sentences are never logged or stored.
"""

import json
import logging
import os
import sys
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from emoji_eval.validate import check_validity, check_validity_v2  # noqa: E402
from model_manager import DEFAULT_MODEL, MODELS, Busy, ModelFailed, ModelManager  # noqa: E402

# Each version is held to the format rules it was trained for. The untrained model has none:
# its answer is shown as it comes, flagged when it isn't a valid emoji sentence.
FORMAT_CHECKS = {"v1": check_validity, "v2": check_validity_v2, "base": None}
assert FORMAT_CHECKS.keys() == MODELS.keys()

MAX_CHARS = int(os.environ.get("EMOGIST_MAX_CHARS", 200))
MAX_BODY_BYTES = 2048
SENTENCES_PER_MINUTE = int(os.environ.get("EMOGIST_SENTENCES_PER_MINUTE", 12))
WARMUPS_PER_MINUTE = int(os.environ.get("EMOGIST_WARMUPS_PER_MINUTE", 6))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("EMOGIST_ALLOWED_ORIGINS", "").split(",") if o.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


class RateLimiter:
    """A small in-memory limit per client, enough for one demo process."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str) -> None:
        now = time.monotonic()
        if len(self.hits) > 10_000:
            self.hits.clear()
        hits = self.hits[client]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            raise ApiError(429, "rate_limited", "Too many requests, please wait a moment.",
                           int(60 - (now - hits[0])) + 1)
        hits.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelManager()
    await app.state.model.start()
    yield
    await app.state.model.close()


app = FastAPI(title="emogist", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])
sentence_limit = RateLimiter(SENTENCES_PER_MINUTE)
warmup_limit = RateLimiter(WARMUPS_PER_MINUTE)


@app.exception_handler(ApiError)
async def api_error(_: Request, error: ApiError) -> JSONResponse:
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else None
    return JSONResponse({"code": error.code, "message": error.message}, status_code=error.status, headers=headers)


def client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def read_sentence(request: Request) -> tuple[str, str]:
    """Return the cleaned sentence and the model version asked for."""
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise ApiError(413, "too_long", f"Please use at most {MAX_CHARS} characters.")
    try:
        data = json.loads(body)
        sentence = data["sentence"]
        model = data.get("model", DEFAULT_MODEL)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ApiError(400, "bad_request", 'Send JSON like {"sentence": "..."}.') from None
    if model not in MODELS:
        raise ApiError(400, "bad_model", f"The model must be one of: {', '.join(MODELS)}.")
    if not isinstance(sentence, str):
        raise ApiError(400, "bad_request", "The sentence must be text.")
    sentence = " ".join(sentence.split())
    if not sentence:
        raise ApiError(400, "empty", "Please type a sentence.")
    if len(sentence) > MAX_CHARS:
        raise ApiError(413, "too_long", f"Please use at most {MAX_CHARS} characters.")
    # "<|" could otherwise be read as one of the model's control tokens.
    if "<|" in sentence or any(unicodedata.category(c) == "Cc" for c in sentence):
        raise ApiError(400, "bad_characters", "The sentence contains unsupported characters.")
    return sentence, model


def result(output: str, model: str, usage: dict) -> dict | None:
    """The answer to send back, or None when a trained version broke its format rules."""
    check = FORMAT_CHECKS[model]
    if check is None:
        valid = check_validity_v2(output) == "ok"
    elif check(output) != "ok":
        return None
    else:
        valid = True
    tokens = {"input": usage.get("input"), "output": usage.get("output")}
    return {"model": model, "emojis": output, "valid": valid, "tokens": tokens, "seconds": usage.get("seconds")}


@app.get("/status")
async def status(request: Request) -> dict:
    return {"state": request.app.state.model.state}


@app.post("/warmup", status_code=202)
async def warmup(request: Request) -> dict:
    warmup_limit.check(client_id(request))
    return {"state": request.app.state.model.warmup()}


@app.post("/emoji")
async def to_emoji(request: Request) -> dict:
    sentence, model = await read_sentence(request)
    sentence_limit.check(client_id(request))
    usage: dict = {}
    try:
        output = await request.app.state.model.generate(sentence, usage, model)
    except Busy:
        raise ApiError(503, "busy", "The demo is busy, please try again later.", 30) from None
    except ModelFailed as error:
        logging.getLogger("emogist").error("generation_failed reason=%s", error)
        raise ApiError(503, "unavailable", "The model is unavailable, please try again.", 10) from None
    answer = result(output, model, usage)
    if answer is None:
        raise ApiError(422, "no_valid_output", "The model couldn't turn this one into emojis. Try rephrasing it.")
    return answer


def event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def emoji_events(request: Request, sentence: str, model: str) -> AsyncIterator[str]:
    pieces = []
    usage: dict = {}
    try:
        async with aclosing(request.app.state.model.stream(sentence, usage, model)) as stream:
            async for piece in stream:
                if await request.is_disconnected():
                    return
                pieces.append(piece)
                yield event({"type": "piece", "text": piece})
    except Busy:
        yield event({"type": "error", "code": "busy"})
        return
    except ModelFailed as error:
        logging.getLogger("emogist").error("generation_failed reason=%s", error)
        yield event({"type": "error", "code": "unavailable"})
        return
    answer = result("".join(pieces).strip(), model, usage)
    if answer is None:
        yield event({"type": "error", "code": "no_valid_output"})
    else:
        yield event({"type": "done", **answer})


@app.post("/emoji/stream")
async def to_emoji_stream(request: Request) -> StreamingResponse:
    sentence, model = await read_sentence(request)
    sentence_limit.check(client_id(request))
    if request.app.state.model.queue_full():
        raise ApiError(503, "busy", "The demo is busy, please try again later.", 30)
    return StreamingResponse(emoji_events(request, sentence, model), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.get("/", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse(HERE / "web" / "index.html")


@app.get("/about", include_in_schema=False)
async def about_page() -> FileResponse:
    return FileResponse(HERE / "web" / "about.html")


@app.get("/random.json", include_in_schema=False)
async def random_sentences() -> FileResponse:
    """Sentences for the page's random button, by language (11_demo_v2/make_random_list.py)."""
    return FileResponse(HERE / "web" / "random.json", media_type="application/json")
