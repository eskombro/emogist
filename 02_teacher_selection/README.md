# Step 2: Teacher selection

<!-- Your summary: what this step is about, in your own words. -->

**Goal:** choose the large model that will label the ~3,000 training sentences. A student rarely ends up better than its labels, so the teacher sets the ceiling for the whole project.

## Candidates

All candidates run on **Ollama Cloud** through the local Ollama, with the same API, prompt, temperature (0.3) and test set as the students in [step 1](../01_student_selection/). One account and one credit balance cover every provider.

| Teacher | Ollama tag | Licence (weights) |
|---|---|---|
| Mistral Large 4 | `mistral-large-4:cloud` | Mistral Ecosystem Open-Weight License (modified MIT) |
| DeepSeek V4 Pro (with and without thinking) | `deepseek-v4-pro:0813-cloud` | MIT |
| DeepSeek V4.1 Flash | `deepseek-v4.1-flash:cloud` | MIT |
| gpt-oss 120B (OpenAI, open weights) | `gpt-oss:120b-cloud` | Apache 2.0 |
| gpt-oss 20B (OpenAI, open weights) | `gpt-oss:20b-cloud` | Apache 2.0 |

**Thinking:** should a teacher reason before answering? Only its final emoji sentence goes into the dataset, so thinking can improve the labels without the student ever seeing it. The cost is time and output tokens. To measure it instead of guessing, DeepSeek V4 Pro runs twice: without thinking (`deepseek-v4-pro`) and with thinking (`deepseek-v4-pro-think`, 3,000-token budget). Mistral Large 4 can't be compared that way: Mistral's docs say it always reasons (lowest setting `reasoning_effort: "low"`; "none" is rejected), and through Ollama `"think": false` is ignored. It runs with a 2,000-token budget and spends 300–600 tokens reasoning per sentence. gpt-oss also always reasons, so it runs at its lowest level (`"think": "low"`) with a 2,000-token budget.

gpt-oss was added after the first run. The bake-off skips models already in the results file, so the other teachers were not called again, and the scorer's caches meant only the new outputs were decoded and judged.

## Can we train on their outputs?

Checked on 2026-11-21 (a reading of the terms, not legal advice). Through Ollama, three layers apply:

1. **Ollama's terms:** "You retain ownership of content you create… We claim no ownership over your inputs or outputs." The only restriction, "develop competing products", is about competing with Ollama's service.
2. **The model licence** (through Ollama we use the open weights, not the provider's API):
   - **DeepSeek V4 Pro / V4.1 Flash: MIT.** No restriction on outputs or on training other models.
   - **gpt-oss 120B / 20B: Apache 2.0.** No restriction on training other models from its outputs. This is OpenAI's open-weight model, not the ChatGPT API, whose terms are different.
   - **Mistral Large 4:** outputs are not restricted, and there is no clause on training other models. Its one condition, "Model-as-a-Service", requires a commercial licence to give third parties access to *Mistral Large 4 or its derivative works*. We don't serve Mistral Large 4 (our API serves a fine-tuned Qwen), and calling it through Ollama is "mere relaying of requests to models hosted by others", which is exempt. Grey area: "derivative works" isn't defined. The usual reading is that a different model trained on outputs is not a derivative of the weights, but this is untested.
3. **Our student:** Qwen3.5 is Apache 2.0, so the fine-tuned model can be published and served.

Policy: prefer an MIT or Apache 2.0 model on equal results (no grey area). If Mistral Large 4 clearly wins, labelling through Mistral's own API instead of Ollama removes the doubt for about $1 more. Labels are disclosed as AI-generated in the dataset card either way.

Pricing per million tokens (input / output), for reference: Mistral Large 4 $0.70 / $2.10 on Ollama ($0.50 / $1.50 on Mistral directly); DeepSeek V4 Pro $1.32 / $3.96 peak, half off-peak; DeepSeek V4.1 Flash $0.30 / $1.20 peak, half off-peak (same on Ollama and DeepSeek directly). At this project's volumes the differences are around $1.

## What we measure

- **Validity and convention warnings**, as for the students.
- **Gist score** (round trip: Haiku decodes, Sonnet judges 0/1/2). Reference points: students ≈ 0.8–1.0, hand-made examples 1.75.
- **Judgement rules**, using the `time` and `speech` tags in the [test set](../data/): ⏪/⏩ where a time marker matters (and *not* elsewhere), and the 💬 speech pattern. Prompting didn't teach these to the students, so the teacher must apply them consistently:

| Student (step 1) | ⏪/⏩ where needed (6) | ⏪/⏩ elsewhere (94) | 💬 on speech (8) | 💬 elsewhere (92) |
|---|---|---|---|---|
| Ministral 3B | 83% | 21% | 50% | 1% |
| Qwen3.5 2B | 0% | 1% | 12% | 0% |
| Qwen3.5 0.8B | 0% | 1% | 38% | 1% |
| SmolLM3 3B | 0% | 21% | 50% | 8% |

- **French quality:** scores per language, plus a read-through of the French outputs.

## Results

Run on 2026-11-21. Gist score on all 100 sentences (invalid outputs count as 0):

| Teacher | Valid | Gist FR | Gist EN | Gist overall | Gist kept FR / EN |
|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash** | 98/100 | 1.42 | 1.26 | **1.34** | 50% / 46% |
| gpt-oss 120B | 99/100 | 1.40 | 1.24 | 1.32 | 46% / 40% |
| Mistral Large 4 | 99/100 | 1.40 | 1.22 | 1.31 | 54% / 44% |
| DeepSeek V4 Pro | 99/100 | 1.34 | 1.26 | 1.30 | 48% / 42% |
| DeepSeek V4 Pro + thinking | 99/100 | 1.30 | 1.30 | 1.30 | 36% / 48% |
| gpt-oss 20B | 97/100 | 1.18 | 1.16 | 1.17 | 42% / 40% |
| *Best students, prompted (step 1)* | | | | *0.89–0.97* | *≈25%* |
| *Hand-made few-shot examples* | | | | *1.75* | |

Paired differences on the same sentences (bootstrap, 95% interval): **the five large teachers are tied on gist** (every interval between them includes 0). For example, Flash − Pro +0.04 [−0.10, +0.18], Flash − Mistral +0.03 [−0.13, +0.19], gpt-oss 120B − Flash −0.02 [−0.16, +0.12]. **gpt-oss 20B is probably weaker:** 20B − Flash = −0.17 [−0.34, +0.00], right at the edge of significance. **Thinking adds nothing:** Pro with thinking − Pro without = +0.00 [−0.15, +0.15]. Teachers are about **+0.4 above the students**, the gap fine-tuning should transfer.

Judgement rules and format:

| Teacher | ⏪/⏩ where needed (6) | ⏪/⏩ elsewhere (94) | 💬 on speech (8) | Keycap numbers (per 100) | Other warnings | Clean labels (valid, no warning) |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 83% | 3% | **88%** | 8 | 1 | 90 |
| Mistral Large 4 | 83% | 2% | 50% | 4 | 5 | 90 |
| DeepSeek V4 Pro | 83% | 3% | 62% | 8 | 1 | 90 |
| DeepSeek V4 Pro + thinking | 83% | 2% | 50% | 2 | 5 | 92 |
| gpt-oss 120B | 67% | 4% | 62% | 2 | 0 | **97** |
| gpt-oss 20B | 67% | 9% | 62% | 4 | 6 | 88 |

- All teachers apply the time rule better than the students (Ministral: 21% elsewhere; Qwen: 0% where needed). With only 6 time and 8 speech sentences, one sentence moves these rates by 12–17 points, so small gaps are hints only.
- gpt-oss 120B has the cleanest format (97 labels with no warning), while Flash is best on speech (7/8 vs 5/8).
- Invalid outputs are rare and mostly copied text (`30°`, `ok`, `10`, `8h`, `2h`); the thinking variant once returned an empty answer.
- **Keycap numbers** (`4️⃣0️⃣`, `🔟`) appear in 2–8% of teacher outputs and never in the students'. The warning columns of all models were recomputed from the stored outputs after this check was added. They pass validity but break the "numbers are drawn" rule and would teach the student to write numbers, so a `number written as keycap` warning was added. For the dataset: tighten the prompt rule and keep only labels with no warnings.

Practical differences: Flash ≈0.5 s per sentence and ≈$0.30–0.60 for 3,000 labels; Pro ≈1 s and ≈$1–2; Mistral Large 4 3–6 s (always reasons) and ≈$3.

**Ceiling:** five different large models land at 1.30–1.34, below the hand-made 1.75, which suggests the limit now comes more from the prompt or the scorer than from the choice of teacher. Since a student rarely beats its labels, a realistic target after fine-tuning is about 1.1–1.25 (from 0.9–1.0). Raising the ceiling means better labels from the same teacher: a tighter prompt, more examples of hard cases, or best-of-n generation with a non-Claude selector (selecting training data with Claude's judgements would be close to training on Claude outputs).

## Decision

<!-- Your decision and why. Facts to weigh:
     - Gist: the five large teachers are tied (1.30-1.34); thinking adds nothing.
     - gpt-oss 120B: tied on gist, cleanest format (97 clean labels), Apache 2.0, always
       reasons, 5/8 on speech. A credible backup. gpt-oss 20B: probably weaker (1.17).
     - DeepSeek V4.1 Flash: top or tied everywhere, best on speech (88%), MIT licence,
       fastest and cheapest.
     - Mistral Large 4: licence grey area, always reasons (slow), no score advantage. -->

## Reproduce

```bash
ollama signin
python 02_teacher_selection/bakeoff.py
python 02_teacher_selection/score.py
```
