# Step 6: Serving

Package the fine-tuned Qwen as a small file that runs without a GPU, and serve it through a public demo.

**Status:** done. The 8-bit GGUF scores the same as the full model on dev at about half the size, and it runs on a shared 4-core VPS. The demo itself now evolves in [`demo/`](../demo/). The git tag `step-6` shows it as it was when this step ended.

# Main features and functionalities

- **Model export:** merges the LoRA adapter into Qwen and converts it to GGUF.
- **Compression comparison:** full precision, 8-bit and 4-bit, on the same 100 dev sentences.
- **On-demand serving:** the model loads when a visitor starts typing and unloads when idle.
- **Protected deployment:** memory, CPU, queue and rate limits keep the demo from disturbing other services.

# Main parts of the project

| Path | Purpose |
|---|---|
| `merge_adapter.py` | Merges the adapter and checks the merged model against the Colab outputs |
| `compare_gguf.py` | Runs each GGUF version on the dev sentences |
| `score_gguf.py` | Scores those outputs with the round-trip scorer |
| `requirements-export.txt` | Exact packages used for the export |
| `results/` | Outputs and scores on dev |
| [`../demo/`](../demo/) | The live API, web page and deployment files |

## Choosing the model file

Gist score on the 100 real dev sentences (0–2, invalid outputs count as 0), run through llama.cpp on an Apple M1 Pro GPU:

| Version | Size | Same output as Colab | Valid | French | English | **Gist** | vs. full model | Time per sentence |
|---|---|---|---|---|---|---|---|---|
| Full precision (16-bit) | 3.78 GB | 100/100 | 98 | 0.88 | 0.62 | **0.75** | | 0.45 s |
| **8-bit (`Q8_0`, chosen)** | **2.01 GB** | **88/100** | **99** | **0.86** | **0.64** | **0.75** | **+0.00 [−0.05, +0.06]** | **0.31 s** |
| 4-bit (`Q4_K_M`) | 1.29 GB | 30/100 | 95 | 0.80 | 0.56 | **0.68** | −0.07 [−0.18, +0.03] | 0.27 s |

- The conversion is faithful: the full-precision GGUF reproduces all 100 Colab outputs.
- 8-bit changes some wording without a measurable loss of gist.
- 4-bit saves another 0.7 GB, but it rewrites most outputs, produces more invalid ones and probably scores lower, although that drop is not proven.

The merge must load Qwen with its full image-text class. With the text-only class, the adapter's layer names don't match and it silently stays inactive, while the model still writes plausible emojis.

## Serving it on demand

The demo shares a VPS with other services, so the model should only use memory while people are using it:

1. Clicking into the text box asks the API to load the model.
2. A sentence is answered at once if the model is ready. Otherwise the page shows a loader until it is.
3. The model unloads 90 seconds after a warm-up with no sentence, or 3 minutes after the last sentence.

FastAPI starts and stops `llama-server` as a separate process, because stopping it is what reliably gives the memory back. A hidden sentence runs during loading, so the first real sentence doesn't pay for reading the model from disk.

Measured on CPU, without weight repacking:

| Where | Threads | Model load | Sentence (median) | Slowest | Gist on dev |
|---|---|---|---|---|---|
| Apple M1 Pro | 2 | about 1.5 s (file cached) | about 1 s | 1.9 s | 0.72, vs. full model −0.03 [−0.10, +0.04] |
| VPS, Intel Haswell | 2 | 5.9 s | 4.57 s | 7.11 s | not measured |
| **VPS, Intel Haswell (chosen)** | **3** | **5.7 s** | **3.67 s** | **4.95 s** | **0.75, vs. full model +0.00 [−0.05, +0.06]** |

The VPS timings come from 12 requests per setting, so they are a guide rather than a precise benchmark. A third thread made sentences about 20% faster. Weight repacking made no clear difference on this CPU, so it stays off.

## Lessons and limitations

- **A silent failure looks like success.** Loaded with the text-only class, the merged model ignored the adapter and still wrote plausible emojis. Checking the merged outputs against Colab is what caught it.
- **8-bit is free, 4-bit isn't.** Halving the size cost nothing measurable; going further rewrote most outputs.
- **A small model doesn't need a GPU to be served,** but on an older CPU each sentence takes seconds, not milliseconds.
- **The VPS is slow for this model.** Its older CPU generates about 6.5 tokens per second, so a sentence takes about 3–5 seconds.
- **Hardware changes some outputs.** On the VPS, 86 of 100 dev outputs match Colab exactly, with 99 valid and the same gist score as the full model.
- **Dev has informed earlier choices,** so these scores are useful to compare versions, not as a final quality estimate.
- **The model is not published.** Qwen3.5 is Apache 2.0, so the fine-tuned model could be published on the Hugging Face Hub, but this is skipped for now. The model runs only in the live demo.

## Decision

**Serve the 8-bit GGUF on our own VPS, loaded on demand.** The 8-bit file is half the size of the full model with the same gist (0.75, +0.00 [−0.05, +0.06]). The 4-bit file saves another 0.7 GB but scores lower (0.68) and produces more invalid outputs. Our own VPS was preferred over free hosting because it has no quota and no long cold start. Since it's shared with other services, the model loads only when someone starts typing and unloads when idle, so it uses no memory while the demo isn't in use.

## Next

v2 starts in [step 7](../07_v2_rules_and_scoring/). Publishing the model on the Hugging Face Hub is possible later, but skipped for now.

## Reproduce

The export used [llama.cpp](https://github.com/ggml-org/llama.cpp) v1.0.0 (build `b12359`).

```bash
pip install -r 06_serving/requirements-export.txt
python 06_serving/merge_adapter.py --out /tmp/emogist-merged

python /path/to/llama.cpp/convert_hf_to_gguf.py /tmp/emogist-merged \
  --outtype f16 --outfile /tmp/gguf/emogist-f16.gguf
llama-quantize /tmp/gguf/emogist-f16.gguf /tmp/gguf/emogist-Q8_0.gguf Q8_0
llama-quantize /tmp/gguf/emogist-f16.gguf /tmp/gguf/emogist-Q4_K_M.gguf Q4_K_M

python 06_serving/compare_gguf.py --llama-server "$(which llama-server)" --gguf-dir /tmp/gguf
python 06_serving/score_gguf.py
```
