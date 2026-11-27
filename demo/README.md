# emogist demo

A small web demo that turns an everyday English, French or Spanish sentence into emojis, live at [emogist.esklab.com](https://emogist.esklab.com).

# Main features and functionalities

- **English, French and Spanish page,** with a 🎲 random-sentence button and a guide to reading the emojis.
- **One base, three versions:** v2 by default, and an opt-in compare view with the untrained model (before and after training). v1 stays available in the API (`"model": "v1"`) but isn't shown on the page.
- **Signs explained in place:** hovering or tapping a sign such as ❓ or 📍 in a result shows its meaning.
- **Share links:** only the sentence travels, after the `#` of the link, so nothing is stored on the server. Opening the link runs the model on it; answers are deterministic, so the visitor sees the same emojis.
- **About page** on how the model was made and scored.
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
| `web/about.html` | The About page |
| `Dockerfile`, `docker-compose.prod.yml` | Production image and container limits |
| `caddy/Caddyfile.snippet` | Reverse proxy site |
| `models/` | The base model and the two adapters as GGUF files (not in git) |

The model files are built by [`11_demo_v2/export_models.py`](../11_demo_v2/export_models.py): `qwen2b-base-Q8_0.gguf` (Qwen3.5 2B, 8-bit), `emogist-v1-adapter-f16.gguf` ([step 4](../04_fine_tuning/)) and `emogist-v2-adapter-f16.gguf` ([step 10](../10_v2_fine_tuning/)). llama-server loads the base with both adapters switched off, and each request turns on the one its version needs. The v1 export as a single merged model is documented in [Step 6](../06_serving/). Pushing a change in this folder to `main` deploys it through [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).

## API

| Route | Purpose |
|---|---|
| `GET /status` | Returns `unloaded`, `loading`, `unloading` or `ready`, without loading the model |
| `POST /warmup` | Starts loading and returns immediately |
| `POST /emoji` | Takes `{"sentence": "...", "model": "v2"}` and returns `{"model": "v2", "emojis": "...", "valid": true, "tokens": {"input": 0, "output": 0}, "seconds": 0.0}` |
| `POST /emoji/stream` | Same input, streamed as server-sent events: `piece`, then `done` (with the same fields) or `error` |
| `GET /about` | The About page |
| `GET /random.json` | Sentences for the page's random button, by language (built by [`11_demo_v2/make_random_list.py`](../11_demo_v2/make_random_list.py)) |

`model` is optional: `v2` (default), `v1` or `base` (the untrained model). v1 and v2 answers that break their format rules return the `no_valid_output` error. The untrained model has no format rules: its answer comes back as it is, with `valid` telling whether it is a valid emoji sentence. It may write at most 40 tokens instead of 60: it often repeats one emoji until the limit, and the repetition shows just as well in 40. Each model run counts towards the per-visitor limit.

Input tokens include the fixed instruction and chat template, not only the visitor's sentence. `seconds` is the generation time on the server, without queueing or loading.

## Run locally

You need `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) and the three model files in `demo/models/`. From the repository root:

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

The workflow needs the repository secrets `GHCR_TOKEN`, `VPS_SSH_KEY`, `VPS_HOST` and `VPS_USERNAME`, plus the repository variable `EMOGIST_DEPLOY=true`. The VPS needs a clone in `~/emogist` and the three model files in `~/emogist/demo/models/`.
