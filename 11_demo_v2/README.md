# Step 11: v2 demo

Update the live demo to the v2 model, with a compare view against the untrained model, a Spanish page, explained signs, share links and an About page.

**Status:** ✅ done. Deployed on 2026-11-27 at [emogist.esklab.com](https://emogist.esklab.com). The demo code lives in [`demo/`](../demo/); this folder holds the export, the checks and the page sentences.

# What was decided

| Decision | Choice | Who / where |
|---|---|---|
| Model size | 2B. The 4B scored +0.05 on dev, but at twice the memory and about half the speed on a CPU-only server that wasn't worth it | Author, step 10 |
| Where | The live demo in `demo/` is updated in place (no separate v2 demo) | Author |
| Default model | v2 main (step 10) | Step 10 |
| Compare view | Opt-in, the untrained model next to v2: before and after training. v1 was planned as a third column, then dropped from the page so a comparison takes two runs and reads as before / after. v1 stays available in the API (`"model": "v1"`) | Author |
| Page sentences | A 🎲 random button with a short reviewed list per language, instead of example buttons | Author |
| About page | English, French and Spanish, with the same information in each | Author |
| Branch | All work on a branch: pushing `demo/` to `main` deploys the live site | Repository workflow |
| v1 re-score | Not done: the parity check below shows the adapter version of v1 is close to the merged one | Author |

# What was built

## Models: one base, two adapters

The server loads **one** model file and two small adapter files instead of one merged model:

| File | Size | Use |
|---|---|---|
| `qwen2b-base-Q8_0.gguf` | 2.1 GB | Qwen3.5 2B, untouched, 8-bit |
| `emogist-v1-adapter-f16.gguf` | 22 MB | v1 adapter (step 4) |
| `emogist-v2-adapter-f16.gguf` | 22 MB | v2 main adapter (step 10) |

llama-server starts with both adapters loaded and switched off (`--lora … --lora … --lora-init-without-apply`). Each request turns on the one it needs (`"lora": [{"id": 0, "scale": 1}, {"id": 1, "scale": 0}]`), and the untrained model has both switched off. Switching per request leaves no trace (checked in the game exploration: 20 of 20).

Each model keeps its own instruction and format rules:

| Model | Instruction | Format check | Max tokens |
|---|---|---|---|
| untrained | v2 instruction, so only the adapter differs from v2 | none: shown as it comes, even plain text | 40, since it often loops on one emoji until the limit |
| v1 | "…a **short** emoji sentence…" (step 4) | `check_validity` (3 to 10 emojis) | 60 |
| v2 | "…an emoji sentence…" (step 10) | `check_validity_v2` (1 to 10 emojis) | 60 |

Memory stays about the same as with the merged v1 model (2.1 GB instead of 2.0 GB), so the container limit (3 GB) didn't change.

## Backend

- `model_manager.py`:
  - Starts llama-server with the base and both adapters.
  - `MODELS = {"v2": …, "v1": …, "base": …}` holds each model's instruction, adapter scales and token limit.
  - The one-sentence-at-a-time lock stays, so a comparison goes through the queue as two sentences.
- `app.py`:
  - The `/emoji` and `/emoji/stream` routes take an optional `"model"` (`v2` by default); anything else is refused.
  - The untrained model returns its raw text with `"valid"` telling whether it is a valid emoji sentence, instead of an error, so the compare view can show it.
  - The per-visitor limit counts each model run, so one comparison uses 2 of the 12 sentences a minute.
  - `/about` and `/random.json` serve the About page and the random-button sentences.
- `Dockerfile`, `docker-compose.prod.yml` and `deploy.yml` point at the three files. The deploy check `test -f models/…` lists them.

## Page (`web/index.html`)

1. **Three languages:** English, French and Spanish, with the guide to reading the emojis in each.
2. **v2 sign guide:** "How to read the emojis" shows the signs v2 uses: ❓ question, 💭 think, 🤔 maybe, ❔ unknown or asked about, 📍 place, 🔁 again or often, ➡️ then or so, 🙋 I or me, 🫵 you, 👥 we.
3. **Signs explained in place:** in a v2 result, each of those signs has a dotted underline. Hovering or tapping it shows its meaning in the page language.
4. **Compare view:** a "Compare with the untrained AI model" switch.
   - When it's on, a sentence runs through v2 first, then the untrained model, and each box fills in as it finishes.
   - Switching it on after a translation runs only the untrained model on the same sentence.
   - A short banner explains that both answers come from the same model, before and after training.
   - When the untrained answer isn't a valid emoji sentence, it's shown as raw text.
5. **🎲 Random button:** types a sentence from a reviewed list in the page language (`random.json`) and translates it. The list goes through in a shuffled order without repeats.
6. **Share links:**
   - A "Share" button copies a link such as `…/#s=<sentence>`: only the sentence, not the emojis.
   - The part after `#` never reaches the server, so nothing is stored or logged.
   - Opening the link puts the sentence in the box and runs v2 on it. Answers are deterministic (temperature 0), so the visitor sees the same emojis, and a link can't show emojis the model never wrote.
   - Text from a link is always shown as plain text, never as HTML.
7. **Result details:** model name, generation time on the server and input → output tokens.
8. **Polish:** emojis pop in as they stream, clearer loading and busy states, a copy button.

## About page (`web/about.html`)

In English, French and Spanish: how it was made (pipeline from hand-made examples to serving), how the student and teacher were chosen, the scores, limits, privacy, AI disclosure and a link to this repository. The opening paragraph is still to be written by the author (a hidden comment marks its place).

## Files

| Path | Purpose |
|---|---|
| `export_models.py` | Builds the base GGUF and the two adapter GGUFs from the step 4 and step 10 adapters (llama.cpp converters) |
| `check_served.py` | The checks below: v2 parity, v1 parity, CPU timing |
| `run_examples.py`, `examples/candidates.csv`, `examples/outputs.csv`, `examples/picked.csv` | 100 candidate sentences per language written by Claude, their outputs from the three versions, and a first pick of examples. Superseded by the random button, kept for the record |
| `examples/random_shortlist.csv` | The random-button sentences: v2 good, the untrained model clearly worse (column `untrained_problem`), v1 valid. Reviewed with the author: EN 20, FR 19, ES 18 |
| `make_random_list.py` | Builds `demo/web/random.json` from the rows marked `keep` in `random_shortlist.csv` |
| `results/` | `served_parity.csv`, `cpu_timing.csv` |

The page sentences are for the demo only, never for training.

## Rollout

1. Work on branch `step-11-demo-v2`, previewed locally after each part.
2. The author copied the three model files to the VPS (`~/emogist/demo/models/`) before merging, keeping the old file until the new version ran.
3. Merge into `main`, which deployed it. The live site and its timing were checked afterwards.

# Results

## Model files

`export_models.py` (llama.cpp b12359 converters, step 6 export packages):

| File | Size | sha256 |
|---|---|---|
| `qwen2b-base-Q8_0.gguf` | 2,077 MB | `63c2cb5f…aec4523` |
| `emogist-v1-adapter-f16.gguf` | 22 MB | `a732bb91…791ef948` |
| `emogist-v2-adapter-f16.gguf` | 22 MB | `40b9cdfd…b096cd52` |

## Checks (`check_served.py`, local, no API calls)

| Check | Result |
|---|---|
| v2: base + v2 adapter vs the step 10 Colab outputs (300 dev sentences, Mac GPU) | 266/300 identical; 294 valid vs 295 in Colab |
| v1: base + v1 adapter vs the merged v1 GGUF (same sentences) | 247/300 identical; 291 valid for both |

The differences come from the 8-bit base: the adapter is applied on top of 8-bit weights instead of being merged before quantizing. Validity doesn't change, so the served v1 is the same model in practice.

**CPU speed** (3 threads as on the server, Mac CPU, 30 dev sentences, seconds per sentence):

| Model | Median | Mean |
|---|---|---|
| v1, merged (before this step) | 1.13 | 1.13 |
| v1, base + adapter | 1.18 | 1.19 |
| v2, base + adapter | 1.07 | 1.10 |
| untrained | 0.93 | 1.03 |

Switching adapters per request costs about 4% on v1. v2 is a bit faster than v1 because its answers are shorter.

## Live timing (VPS)

- **Generation:** v2 about 2.6 to 4.1 s per sentence, v1 about 2.8 to 3.7 s. The untrained model took 8.3 s when it looped for 60 tokens, which led to its 40-token cap.
- **Cold start:** about 6 s to load the model.
- **Unexplained:** some requests take 2 to 5 s longer in the browser than the server's own time, probably network or load from other services on the VPS.

## Page check

- A smoke test in jsdom against the local server: share link, sign meanings in each language, compare mode, no script errors.
- Checked by the author in a browser, on desktop and phone. Two fixes came out of it: the compare switch now fills in the untrained answer for a sentence already translated, and the sign guide no longer overlaps on narrow phones.

## Decision

**The live demo runs v2 by default, with v1 kept in the API, and compares v2 with the untrained model.**
- **One base and two adapters:** v1 and v2 share a single 2B model, so adding v2 cost 22 MB and no extra memory. Outputs stay valid and the speed is unchanged.
- **Before / after instead of v1 / v2:** the untrained model shows what training changed at a glance. v1 and v2 tie on the score, so a v1 column would mostly confuse visitors.
- **Random sentences instead of example buttons:** a reviewed list where v2 does well and the untrained model clearly doesn't, so the comparison makes its point quickly. It is a showcase, not a measure of quality.
- **Share links without storage:** only the sentence goes in the link, and answers are deterministic, so nothing is stored and a link can't show fake emojis.

This step ends v2.

## Lessons and limitations

- **Adapters make versions cheap.** Switching per request costs about 4% in speed, and adding a model costs 22 MB instead of 2 GB.
- **Show the difference, not the score.** A before / after view explains fine-tuning better than a number visitors can't check.
- **Test on a real phone.** The author's browser check found two problems the automated smoke test missed.
- **Limitations:** the served models run on an 8-bit base, so 11% of v2 outputs differ from Colab (validity unchanged). The random sentences are hand-picked, so the demo looks better than average. Some live requests take 2 to 5 s longer than the server's own time, for reasons not yet found.
