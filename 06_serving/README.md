# Step 6: Serving

Package the fine-tuned Qwen as a small file that runs without a GPU, and serve it through a public demo.

<!-- Your summary: what this step is about, in your own words. -->

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

Measured with the served settings (2 CPU threads, no weight repacking):

| Where | Model load | Sentence | Gist on dev |
|---|---|---|---|
| Apple M1 Pro CPU | about 1.5 s (file cached) | median about 1 s | 0.72, vs. full model −0.03 [−0.10, +0.04] |
| VPS, 4 vCores, x86 | 7.4 s (first measured load) | about 3–7 s | not measured |

## Limitations

- **The VPS is slower than expected.** Sentences take several seconds. Weight repacking and thread settings have not been benchmarked there yet.
- **Quality on the VPS has not been scored.** The dev score above comes from the Mac CPU, and hardware can change some outputs.
- **Dev has informed earlier choices,** so these scores are useful to compare versions, not as a final quality estimate.
- **The model is not published yet.** The Qwen licence needs checking first.

## Decision

<!-- Your decision and why. Facts to weigh:
     - 8-bit: 2.01 GB, gist 0.75, same as full precision (+0.00 [-0.05, +0.06]).
     - 4-bit: 1.29 GB, gist 0.68, more invalid outputs.
     - Your VPS over free hosting: no quota or long cold start, shared with other services.
     - On-demand loading keeps the memory free when nobody uses the demo. -->

## Next

Benchmark the VPS settings, score the dev sentences there, then check the Qwen licence before publishing the model.

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
