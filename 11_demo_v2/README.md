# Step 11: v2 demo

Update the live demo to the v2 model, with v1 and the untrained model available for comparison, a Spanish page, explained signs, share links and an About page.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** 🔄 built on branch `step-11-demo-v2` and previewed locally; not deployed yet.

# Plan

## Decisions already taken

| Decision | Choice | Who / where |
|---|---|---|
| Model size | 2B. The 4B scored +0.05 on dev, but at twice the memory and about half the speed on a CPU-only server that wasn't worth it | Author, step 10 |
| Where | The live demo in `demo/` is updated in place (no separate v2 demo) | Author, this step |
| Default model | v2 main (step 10) | Step 10 |
| Features | Switchable models, Spanish page, v2 sign guide, compare view, signs explained in place, share links, About page | Author, this step |
| Compare view | Untrained, v1 and v2 side by side, opt-in | Author, this step |
| Branch | All work on a branch: pushing `demo/` to `main` deploys the live site | Repository workflow |

## Models: one base, two adapters

The server loads **one** model file and two small adapter files instead of one merged model:

| File | Size | Use |
|---|---|---|
| `qwen2b-base-Q8_0.gguf` | 2.1 GB | Qwen3.5 2B, untouched, 8-bit |
| `emogist-v1-adapter-f16.gguf` | 22 MB | v1 adapter (step 4) |
| `emogist-v2-adapter-f16.gguf` | 22 MB | v2 main adapter (step 10) |

llama-server starts with both adapters loaded and switched off (`--lora … --lora … --lora-init-without-apply`). Each request turns on the one it needs (`"lora": [{"id": 0, "scale": 1}, {"id": 1, "scale": 0}]`), and the untrained model is both switched off. Switching per request leaves no trace (checked in the game exploration: 20 of 20).

Each model keeps its own instruction and format rules:

| Model | Instruction | Format check |
|---|---|---|
| untrained | v2 instruction, so only the adapter differs from v2 | none: shown as it comes, even plain text. Capped at 40 tokens instead of 60 (author), since it often loops on one emoji until the limit |
| v1 | "…a **short** emoji sentence…" (step 4) | `check_validity` (3 to 10 emojis) |
| v2 | "…an emoji sentence…" (step 10) | `check_validity_v2` (1 to 10 emojis) |

Memory stays about the same as today (2.1 GB instead of 2.0 GB), so the container limit (3 GB) doesn't change.

**Checks before switching** (local, no API calls), in `check_served.py`:
1. **v2:** the served setup against the step 10 Colab outputs on the 300 dev sentences. Spot check so far: 25 of 30 identical for 2B, and the rest are small differences of the 8-bit base.
2. **v1:** the adapter setup against today's merged v1 model on the same 300 sentences. The game exploration found 87 of 100 identical, with all outputs valid. If the share is similar, v1 stays comparable. Re-scoring it with the judge is possible but costs API credits, so only if you want it.
3. **Speed:** each model on the CPU with 3 threads, as on the server. An unmerged adapter adds some work per token, so v1 and v2 may be a little slower than today's merged model.

## Backend changes

- `model_manager.py`:
  - Starts llama-server with the base and both adapters.
  - `MODELS = {"base": …, "v1": …, "v2": …}` holds each model's instruction, adapter scales and format check.
  - `stream(sentence, model)` sends the matching `lora` field.
  - The one-sentence-at-a-time lock stays, so a compare run goes through the queue as three sentences.
- `app.py`:
  - The `/emoji` and `/emoji/stream` routes take an optional `"model"` (`v2` by default), and anything else is refused.
  - The untrained model returns its raw text with `"valid": false` instead of an error, so the compare view can show it.
  - The per-visitor limit counts each model run, so one comparison uses 3 of the 12 sentences a minute.
- `Dockerfile`, `docker-compose.prod.yml` and `deploy.yml` point at the three new files. The deploy check `test -f models/…` lists them.
- `demo/README.md`: API, settings and files updated.

## Page changes (`web/index.html`)

1. **Spanish:** a third page language (EN / FR / ES), with the guide and examples in Spanish. I draft the Spanish texts and you check them.
2. **v2 sign guide:** the "How to read the emojis" section gets the signs v2 uses: ❓ question, 💭 think, 🤔 maybe, ❔ unknown or asked about, 📍 place, 🔁 again or often, ➡️ then or so, 🙋 I or me, 🫵 you, 👥 we.
3. **Signs explained in place:** in a v2 result, each of those signs gets a dotted underline. Tapping or hovering shows its meaning in the page language. The guide stays for the full picture.
4. **Compare view:** a "Compare with v1 and the untrained model" switch under the box.
   - When it's on, the sentence runs through v2, then v1, then untrained, and each column fills in as it finishes.
   - Each column has a one-line label, for example "Untrained: the same model before fine-tuning".
   - When the untrained answer isn't emojis, it's shown greyed out with "not emojis".
   - A note says that comparing takes about three times as long.
5. **Share links:**
   - A "Share" button copies a link such as `…/#s=<sentence>`: only the sentence, not the emojis (author's change).
   - The part after `#` never reaches the server, so nothing is stored or logged.
   - Opening the link puts the sentence in the box and runs v2 on it. Answers are deterministic (temperature 0), so the visitor sees the same emojis, and a link can't show emojis the model never wrote.
   - Text from a link is always shown as plain text, never as HTML.
6. **Result details:** the token line grows into "v2 · 2B model on a CPU · 1.9 s · 46 tokens".
7. **Small polish:** an emoji pop-in as each piece arrives, and clearer loading and busy states.

## About page

A separate page (`/about`) linked from the header and footer. I can draft the structure and fill in facts and numbers. The text in your voice is yours.

1. **What it is:** one paragraph (yours).
2. **How it was made:** a small diagram of the pipeline:
   - sentences
   - teacher labels (DeepSeek Flash)
   - LoRA fine-tuning of Qwen3.5 2B on Colab
   - an 8-bit GGUF served by llama.cpp on a small CPU server
3. **What the numbers say:**
   - v1: on the step 5 test set, fine-tuning raised the score from 0.81 to 1.13 (out of 2). That closed 64% of the gap to the teacher.
   - v2: round-trip score 0.47, the same as v1. The blind Sonnet review leans towards v2 (+0.14).
   - 4B: +0.05, and why it wasn't used.
   - The teacher ceiling is 0.57.
4. **How the score works:** a reader model guesses the sentence back from the emojis, and a judge compares the guess with the original.
5. **Limits:** idioms, abstract ideas, long sentences, numbers. Sentences are not logged.
6. **AI disclosure:** what was done with AI help, in the project's usual wording.
7. **Links:** the GitHub repository and the step READMEs.

## Files (planned)

| Path | Purpose |
|---|---|
| `11_demo_v2/README.md` | This plan, then results and decision |
| `11_demo_v2/export_models.py` | Builds the base GGUF and the two adapter GGUFs from the step 4 and step 10 adapters (llama.cpp converters) |
| `11_demo_v2/check_served.py` | The three checks above: v2 parity, v1 parity, CPU timing |
| `11_demo_v2/run_examples.py`, `examples/` | Candidate page examples and their outputs from the three versions |
| `11_demo_v2/make_random_list.py` | Builds `demo/web/random.json` for the page's random button from the rows marked `keep` in `examples/random_shortlist.csv` |
| `11_demo_v2/examples/random_shortlist.csv` | The random-button sentences: v2 good, the untrained model clearly worse (column `untrained_problem`), v1 valid. Reviewed with the author: EN 20, FR 19, ES 18 |
| `11_demo_v2/results/` | `served_parity.csv`, `cpu_timing.csv` |
| `demo/…` | The updated demo (code above) |

## Rollout

1. Branch `step-11-demo-v2`.
2. Export the three model files and run `check_served.py`. Results go in this README.
3. Backend, then the page, previewed locally after each part.
4. **You** copy the three model files to the VPS (`~/emogist/demo/models/`) before merging, and keep the old file until the new version runs.
5. Merge into `main`, which deploys. Then check the live site and its timing.
6. Root README and portfolio write-up afterwards.

## Answers to the plan's questions

- **About page:** English only for now (author).
- **Page examples:** new ones in all three languages, funny, and showing what training changed (author). Claude wrote 100 candidates per language (`examples/candidates.csv`) and ran them through the three versions (`run_examples.py`, output in `examples/outputs.csv`). It then picked six per language where the untrained model clearly fails (endless repeats, the sentence copied as text) and v1 and v2 give a good answer (`examples/picked.csv`). They are for the page only, never for training, and may change later.
- **Random button instead of examples** (author): a 🎲 button next to Translate runs a sentence from a short curated list in the page language; the example buttons were removed. The list keeps only sentences where v2 is good and the untrained model is clearly worse (it loops, copies the sentence as text, uses a wrong emoji or misses the key point), so a comparison shows what training changed.
- **Compare view without v1** (author): the page compares only the untrained model and v2, so a comparison takes two runs instead of three and reads as before / after training. v1 stays loaded and available in the API (`"model": "v1"`, 22 MB); its scores stay on the About page.
- **Compare order:** first untrained, then v1, then v2 (author); after v1 was dropped from the page, v2 first, then the untrained model (author), so v2's answer comes first.
- **v1 re-score:** not done. The parity check below shows the adapter version of v1 is close to the merged one.

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
| v1: base + v1 adapter vs today's merged v1 GGUF (same sentences) | 247/300 identical; 291 valid for both |

The differences come from the 8-bit base: the adapter is applied on top of 8-bit weights instead of being merged before quantizing. Validity doesn't change, so the served v1 is the same model in practice.

**CPU speed** (3 threads as on the server, Mac CPU, 30 dev sentences, seconds per sentence):

| Model | Median | Mean |
|---|---|---|
| v1, merged (today) | 1.13 | 1.13 |
| v1, base + adapter | 1.18 | 1.19 |
| v2, base + adapter | 1.07 | 1.10 |
| untrained | 0.93 | 1.03 |

Switching adapters per request costs about 4% on v1. v2 is a bit faster than today's v1 because its answers are shorter. The VPS was about 4 times slower than the Mac CPU for v1, so expect about 4 seconds per sentence there and about 12 for a comparison.

## Page check

Smoke test in jsdom against the local server: share link, sign meanings in each language, compare mode with the three versions, no script errors. Still to check by eye in a real browser, on desktop and phone.

## Decision

<!-- Your decision and why. -->
