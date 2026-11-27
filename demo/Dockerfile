# Build from the repository root, because the API reuses emoji_eval/validate.py:
#   docker build -f demo/Dockerfile -t emogist .
# The model is not in the image. Compose mounts it read-only from the VPS.

FROM python:3.12-slim AS llama
ARG LLAMA_BUILD=b12359
ARG LLAMA_SHA256=0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /tmp/llama.tar.gz \
       "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" \
    && echo "${LLAMA_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/llama \
    EMOGIST_LLAMA_SERVER=/opt/llama/llama-server \
    EMOGIST_MODEL=/models/qwen2b-base-Q8_0.gguf \
    EMOGIST_V1_ADAPTER=/models/emogist-v1-adapter-f16.gguf \
    EMOGIST_V2_ADAPTER=/models/emogist-v2-adapter-f16.gguf
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /opt/llama /opt/llama
WORKDIR /app
COPY demo/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock
COPY emoji_eval/__init__.py emoji_eval/validate.py emoji_eval/
COPY demo/app.py demo/model_manager.py demo/
COPY demo/web demo/web
RUN useradd --system --uid 10001 --no-create-home emogist
USER emogist
EXPOSE 8000
CMD ["uvicorn", "app:app", "--app-dir", "/app/demo", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*", "--no-access-log", "--no-server-header"]
