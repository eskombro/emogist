# emogist demo

A small web demo that turns an everyday English or French sentence into emojis, live at [emogist.esklab.com](https://emogist.esklab.com).

# Main features and functionalities

- **English and French page,** with clickable examples and a guide to reading the emojis.
- **On-demand model:** loads when a visitor starts typing and unloads when idle.
- **Output validation:** invalid model outputs are replaced by a clear message.
- **Limits for a shared server:** memory, CPU, queue, sentence length and per-visitor rate limits.
- **Privacy:** sentences are not logged or stored.

# Main parts of the project

| Path | Purpose |
|---|---|
| `app.py` | FastAPI routes, input checks and rate limits |
| `model_manager.py` | Starts, warms up and stops the `llama-server` process |
| `web/index.html` | The demo page |
| `Dockerfile`, `docker-compose.prod.yml` | Production image and container limits |
| `caddy/Caddyfile.snippet` | Reverse proxy site |
| `models/` | The GGUF model (not in git) |

The export and evaluation of the model are documented in [Step 6](../06_serving/). Pushing a change in this folder to `main` deploys it through [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).

## API

| Route | Purpose |
|---|---|
| `GET /status` | Returns `unloaded`, `loading`, `unloading` or `ready`, without loading the model |
| `POST /warmup` | Starts loading and returns immediately |
| `POST /emoji` | Takes `{"sentence": "..."}` and returns `{"emojis": "...", "tokens": {"input": 0, "output": 0}}` |
| `POST /emoji/stream` | Same input, streamed as server-sent events: `piece`, then `done` (with `tokens`) or `error` |

Input tokens include the fixed instruction and chat template, not only the visitor's sentence.

## Run locally

You need `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) and the model at `demo/models/emogist-qwen2b-Q8_0.gguf`. From the repository root:

```bash
python -m venv demo/.venv
demo/.venv/bin/pip install -r demo/requirements.txt
EMOGIST_LLAMA_SERVER=/path/to/llama-server \
  demo/.venv/bin/uvicorn app:app --app-dir demo --host 127.0.0.1 --port 8000
```

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `EMOGIST_THREADS` | `3` | CPU threads for the model |
| `EMOGIST_WARMUP_HOLD` | `90` | Seconds a warm-up keeps the model loaded |
| `EMOGIST_SENTENCE_HOLD` | `180` | Seconds the model stays loaded after a sentence |
| `EMOGIST_MIN_AVAILABLE_GB` | `3` | Don't load the model below this available RAM |
| `EMOGIST_MAX_PENDING` | `8` | Sentences allowed in the queue |
| `EMOGIST_SENTENCES_PER_MINUTE` | `12` | Sentences per visitor per minute |
| `EMOGIST_WARMUPS_PER_MINUTE` | `6` | Warm-ups per visitor per minute |
| `EMOGIST_PORT` | `8010` | Loopback port published for the reverse proxy |

## Deployment

The workflow needs the repository secrets `GHCR_TOKEN`, `VPS_SSH_KEY`, `VPS_HOST` and `VPS_USERNAME`, plus the repository variable `EMOGIST_DEPLOY=true`. The VPS needs a clone in `~/emogist` and the model in `~/emogist/demo/models/`.
