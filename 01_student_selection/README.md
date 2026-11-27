# Step 1: Student selection

**Goal:** choose the small open model to fine-tune, by comparing the candidates *before* any training (prompted only).

## Method

- **Candidates** (local, via Ollama): Qwen3.5 2B, Qwen3.5 0.8B, Ministral 3B, plus SmolLM3 3B as a negative control (kept to show what a weak student looks like; not trained).
- **Prompt:** the system rules from [CONVENTIONS.md](../CONVENTIONS.md) and 12 hand-made few-shot examples (6 EN, 6 FR), one sentence per request, temperature 0.3.
- **Test set:** 100 sentences, 50 FR and 50 EN ([data/](../data/)).
- **Validity:** format rules (3–10 emojis, emoji only, single line) and convention warnings ([`emoji_eval/validate.py`](../emoji_eval/validate.py)).
- **Round-trip score** ([`emoji_eval/roundtrip.py`](../emoji_eval/roundtrip.py)): Claude Haiku 5.5 decodes each emoji sentence back into a sentence without seeing the original; Claude Sonnet 5.5 judges whether the decode keeps the gist of the original: **2** = gist kept, **1** = partly, **0** = wrong story. Invalid outputs count as 0.

Scorer sanity check on the hand-made examples: matched pairs **1.75**, shuffled pairs **0.00**.

## Results

| Model | Valid (FR / EN) | Gist FR (0–2) | Gist EN (0–2) | Gist kept FR / EN |
|---|---|---|---|---|
| Ministral 3B | 50 / 50 | **0.98** | **0.96** | 26% / 24% |
| Qwen3.5 2B | 50 / 50 | **0.98** | 0.80 | 26% / 18% |
| Qwen3.5 0.8B | 50 / 49 | 0.68 | 0.50 | 14% / 8% |
| SmolLM3 3B | 48 / 45 | 0.46 | 0.46 | 8% / 10% |

Paired differences on the same sentences (bootstrap, 95% interval):

| Comparison | FR | EN |
|---|---|---|
| Qwen 2B − Ministral 3B | +0.00 [−0.22, +0.22] (tie) | −0.16 [−0.38, +0.04] (tie, slight edge to Ministral) |
| Qwen 2B − Qwen 0.8B | +0.30 [+0.08, +0.54] | +0.30 [+0.08, +0.52] |
| Qwen 0.8B − SmolLM3 | +0.22 [−0.04, +0.48] | +0.04 [−0.18, +0.26] |

Other observations:
- Format is nearly solved by prompting once the examples cover the weak spots (numbers drawn as ⏳ instead of copied digits, ❌ placement).
- Judgement rules are not learned from the prompt (measured with the `time`/`speech` tags of the test set): Ministral uses ⏪/⏩ where needed (83%) but also on 21% of other sentences; the Qwen models never use them where needed; 💬 appears on at most half of the speech sentences.
- SmolLM3 falls into repetition loops (the same emoji 30 times).
- Only about a quarter of sentences keep the full gist even for the best students, against 1.75 for the hand-made examples: plenty of room for fine-tuning.

## Follow-up: after fine-tuning

The tie was tested again in the [fine-tuning pilot](../04_fine_tuning/), on 100 real dev sentences with the same 400 training examples. Qwen 2B reached **0.68** and Ministral 3B **0.56**. The difference (−0.12 [−0.25, +0.02] for Ministral) is not conclusive, so Qwen is the favorite and Ministral stays as the challenger until both are trained on the full dataset.

## Decision

**Qwen3.5 2B is the main student and Ministral 3B the challenger.** The two were tied and both were clearly ahead of Qwen 0.8B. On a tie, the smaller model wins: Qwen 2B is about a third smaller, so faster and cheaper to serve on a CPU-only VPS. Ministral stays in the plan because a tie before training doesn't show which model learns better, so both were trained.

[Step 4](../04_fine_tuning/) confirmed the choice: after full fine-tuning they were still tied on dev (0.75 vs 0.72), with Qwen ahead in French.

## Lessons and limitations

- **Score the gist, not the objects.** An embedding-similarity score was tried first and rejected: it rewards naming the right objects even when the story is wrong. "A bee enters the house, fire, rainbow" scored 0.70 against "bees nested in our chimney", while the gist judge gives it 0.
- **Prompting fixes format, not judgement.** Good examples were enough to get almost every output into the right format, but rules that need judgement (time ⏪/⏩, speech 💬) were not followed. That is what fine-tuning is for.
- **Small sets give wide intervals.** With 50 sentences per language, a 0.16 gap is still a tie.
- **Limitations:** the decoder (Haiku) was not compared with other decoders; the scorer had no human check at this stage (it passed one later, in [step 5](../05_evaluation/)); and each model ran once, so run-to-run variation is not in the intervals.

## Reproduce

```bash
python 01_student_selection/bakeoff.py           # needs Ollama + the four models
python 01_student_selection/score.py             # reuses cache/, so no API calls for these results
python 01_student_selection/score.py --sanity    # scorer sanity check
python 01_student_selection/human_check.py       # optional: guess emoji sentences yourself
```
