# Step 10: v2 fine-tuning

Train new LoRA adapters on the v2 dataset and measure them on the v2 dev set against the v1 model.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** ✅ done. Main, control and 4B main trained and scored; the demo uses 2B main (step 11).

# Plan

## Decisions already taken

| Decision | Choice | Who / where |
|---|---|---|
| Starting point | A new LoRA on the original Qwen3.5 2B, not on top of the v1 adapter | Author, v2 plan |
| Runs | **main** on EN+FR+ES (6,000 rows) and **control** on EN+FR only (the same 4,000 EN/FR rows) | Author, v2 plan |
| Training settings | Same as v1 (step 4): 3 epochs, batch 4 × 2 accumulation steps, learning rate 0.0002, LoRA rank 16, keep the epoch with the lowest validation loss | v1 recipe, so only the data changes |
| Instruction | "Convert the sentence into an emoji sentence that keeps its gist. Answer with emojis only." v1's line without "short", because v2 discourages very short outputs | Author, this step |
| Score | v2 dev (300 sentences, 100 per language), Haiku decoder (v1 prompt, unchanged), v2.1 judge, invalid outputs = 0 | Steps 7 and 8 |
| Baseline | v1 model on v2 dev: 0.47 [0.44, 0.51] (`07_v2_rules_and_scoring/results/v1_dev_scores.csv`). The decoder is unchanged, so it stays comparable | Step 7 |
| Test set | Untouched until the end of v2 | Project rule |

## Method

**Validation split:** 10% of each language and source (real or AI), drawn with a fixed seed from the full file: 600 rows for main.
Control drops the Spanish rows from both parts, so both runs train on the same 3,600 EN/FR examples and validate on the same 400 EN/FR examples.
The main run trains on 5,400 examples in total.

**Length:** the longest v2 example is 118 tokens (instruction, sentence and label), and the longest label is 47 tokens. So v1's 256-token limit and 60 generated tokens are enough.

**Dev run:** both runs write emojis for all 300 dev sentences. For control, the Spanish part shows what the model does in a language it never trained on.
The v2 runs generate in 16-bit in Colab, while the v1 baseline ran as the served 8-bit GGUF. In step 6 the two formats scored the same on dev (0.75).

**Comparisons**, paired on the same sentences with 95% bootstrap intervals:
- main − v1 and control − v1: did v2 training help?
- main − control on EN+FR: does adding Spanish help or hurt English and French?
- main − control on ES: what does training on Spanish add?

## Files

| Path | Purpose |
|---|---|
| `train_qwen_v2.ipynb` | Colab notebook: set `RUN = "main"` or `"control"` and `MODEL_SIZE = "2b"` or `"4b"`, train, run on dev, download a zip |
| `score_dev.py` | Counts missing decodes and judgements (no API calls), or with `--run` scores the runs and prints the comparisons |
| `runs/run_qwen<size>_v2_<run>/` | Dev outputs, settings and loss history of each run (adapters are not committed) |
| `results/dev_scores_<run>.csv` | Per-sentence outputs, decodes and criteria scores |

**Scorer check:** run on the v1 outputs as a stand-in run, `score_dev.py` reproduced the v1 baseline exactly (0.47, difference +0.00) with no API calls.

## Cost

About 600 Haiku decodes and 600 Sonnet judgements if no outputs are already cached: roughly $2 (rough estimate from prompt sizes). `score_dev.py` without `--run` prints the exact number of calls before any spend.

## Reproduce

1. Open `train_qwen_v2.ipynb` in Colab with a **T4 GPU**, set `RUN = "main"`, upload `09_v2_dataset_and_labels/train_main.jsonl` and `08_v2_spanish_and_eval_sets/sets/dev.csv`, run the cells in order, and run step 7 **once**.
2. Unzip the downloaded archive into `runs/run_qwen2b_v2_main/`.
3. Repeat in a new session with `RUN = "control"`.
4. Then:

```bash
python 10_v2_fine_tuning/score_dev.py          # what is missing, no API calls
python 10_v2_fine_tuning/score_dev.py --run    # decode + judge, then report
```

## Results

**Main run (EN+FR+ES):** 5,400 training examples, 25 minutes on a T4. Validation loss by epoch: 0.848, **0.838**, 0.876, so epoch 2 was kept (as in v1). 295 of 300 dev outputs are valid. The 5 invalid ones all exceed 10 emojis, 3 of them because each digit of a long number counts as an emoji. Outputs average 6.0 emojis, against 6.7 for v1.

Score on v2 dev (0–1, invalid = 0, v2.1 judge), with 95% intervals:

| | v1 model | v2 main | main − v1 (paired) |
|---|---|---|---|
| All (300) | 0.47 [0.44, 0.51] | 0.47 [0.44, 0.51] | −0.00 [−0.03, +0.03] |
| English | 0.47 | 0.46 | −0.02 [−0.07, +0.04] |
| French | 0.46 | 0.47 | +0.01 [−0.05, +0.06] |
| Spanish | 0.49 | 0.50 | +0.01 [−0.06, +0.07] |
| Real | 0.43 | 0.46 | +0.03 [−0.03, +0.08] |
| AI | 0.52 | 0.49 | −0.03 [−0.07, +0.01] |

By type (v1 → main): conditional 0.60 → 0.52, mishap 0.63 → 0.56, speech 0.56 → 0.50, long 0.43 → 0.47, moment 0.45 → 0.49, numbers 0.52 → 0.55, question 0.42 → 0.43; the others within 0.01. These groups have 15 to 45 sentences each, so these differences are not proven.
Mean criteria points (of 2) on valid outputs: topic 1.24 → 1.24, event 0.80 → 0.81, outcome 0.76 → 0.72.

For reference, the teacher (Flash with the v2 prompt) scored 0.57 on this dev set in step 9.

Scoring cost: 270 Haiku decodes and 270 Sonnet judgements (estimate $0.94).

**Control run (EN+FR):** 3,600 training examples, 18 minutes. Validation loss by epoch: 0.825, **0.814**, 0.854, so epoch 2 was kept. Its validation set has no Spanish, so these losses can't be compared with main's. 294 of 300 dev outputs are valid: 4 exceed 10 emojis, and 2 broke off mid-way (a cut emoji, a run of digits).

| | v1 model | v2 main | v2 control | main − control (paired) |
|---|---|---|---|---|
| All (300) | 0.47 | 0.47 | 0.46 [0.42, 0.49] | +0.02 [−0.01, +0.05] |
| English | 0.47 | 0.46 | 0.43 | +0.03 [−0.02, +0.08] |
| French | 0.46 | 0.47 | 0.45 | +0.02 [−0.02, +0.07] |
| EN+FR (200) | 0.47 | 0.46 | 0.44 | +0.03 [−0.01, +0.06] |
| Spanish | 0.49 | 0.50 | 0.50 | −0.00 [−0.07, +0.06] |

control − v1: −0.02 [−0.05, +0.02] overall, −0.03 [−0.07, +0.01] on EN+FR.
By type (v1 / main / control): conditional 0.60 / 0.52 / 0.62, mishap 0.63 / 0.56 / 0.52, long 0.43 / 0.47 / 0.42, numbers 0.52 / 0.55 / 0.50, question 0.42 / 0.43 / 0.44. With 15 to 45 sentences per type, none of these differences is proven.
Scoring cost: 237 Haiku decodes and 237 Sonnet judgements (estimate $0.82).

**What the two runs show:**
- **Adding Spanish did not hurt English and French.** Main is level with or slightly above control on EN+FR (+0.03 [−0.01, +0.06]).
- **Training on Spanish did not raise the Spanish score.** Control, which never saw Spanish, scores the same in Spanish as main (0.50 vs 0.50). The base model already reads Spanish, and the emojis don't depend on the language.
- **Neither v2 run beats v1 on this score.**

**4B main run (EN+FR+ES):** Qwen3.5 4B with the same data, split and settings, to see what a model twice the size adds. 61 minutes on a T4 (2B: 25). Validation loss by epoch: 0.770, **0.766**, 0.828, so epoch 2 was kept. That's well below 2B main's 0.838, and both runs validate on the same 600 rows, so the losses compare directly. 295 of 300 dev outputs are valid: all 5 invalid ones have 11 or 12 emojis (✅✅✅, long numbers). Outputs are about as long as 2B main's, but only 21 of 300 are identical to them.

| | v1 model | v2 main (2B) | v2 main (4B) | 4B − v1 (paired) | 4B − 2B main (paired) |
|---|---|---|---|---|---|
| All (300) | 0.47 | 0.47 | **0.52** [0.49, 0.56] | **+0.05 [+0.01, +0.08]** | **+0.05 [+0.01, +0.08]** |
| English | 0.47 | 0.46 | 0.50 | +0.03 [−0.03, +0.09] | +0.05 [−0.01, +0.11] |
| French | 0.46 | 0.47 | 0.51 | +0.04 [−0.02, +0.11] | +0.04 [−0.02, +0.09] |
| Spanish | 0.49 | 0.50 | 0.56 | +0.07 [+0.01, +0.13] | +0.06 [+0.00, +0.12] |
| Real | 0.43 | 0.46 | 0.49 | +0.06 [+0.01, +0.11] | +0.03 [−0.02, +0.08] |
| AI | 0.52 | 0.49 | 0.56 | +0.04 [−0.01, +0.09] | +0.07 [+0.02, +0.12] |

By type (v1 / 2B main / 4B main): future 0.48 / 0.47 / 0.63, conditional 0.60 / 0.52 / 0.69, moment 0.45 / 0.49 / 0.56, speech 0.56 / 0.50 / 0.57, past 0.45 / 0.45 / 0.51, question 0.42 / 0.43 / 0.47, mishap 0.63 / 0.56 / 0.52; the others within 0.05 of v1. With 15 to 45 sentences per type, none of these is proven on its own.
Mean criteria points (of 2) on valid outputs: 4B gains on all three, topic 1.24 → 1.34, event 0.81 → 0.91, outcome 0.72 → 0.83 (2B main → 4B).
Scoring cost: 252 Haiku decodes and 253 Sonnet judgements (estimate $0.88).

**What the 4B run shows:** it's the first v2 model to beat v1 with an interval above zero (+0.05 overall), and it beats 2B main by the same margin. It covers about half of the gap between v1 (0.47) and the teacher (0.57). The gain is spread over the three criteria and all three languages, so it reads as a better model overall rather than a fix for one weakness. The cost is size: about 2× the memory and slower answers, which matters for the CPU-only demo server.

## Blind review by Sonnet (first-time user view)

After trying the v2 demo, the author felt v2 was clearly better than v1, while the dev score shows a tie. To check that impression without the author's bias, Sonnet 5.5 judged the two models blind (`blind_review.py`):

1. **Sentences:** Sonnet wrote 100 test sentences "to test how good a text-to-emoji converter is", in English, French and Spanish (35 / 33 / 32). It wrote 104, and the first 100 were kept. These sentences are for evaluation only and never go into training.
2. **Outputs:** v1 (the served 8-bit GGUF) and v2 main (its adapter on the 8-bit base) converted them locally. Both models gave 100 valid outputs, 7 of them identical. v2 averages 4.0 emojis per output, against 4.7 for v1.
3. **Evaluation:** Sonnet saw each sentence with the two outputs as converter A and B, with no information about the project. It was asked to judge as a user who sees only the sentence and its emojis, name its criteria, compare A and B, give conclusions, and pick A, B or tie for each sentence. It ran twice, with A and B swapped.

**Per-sentence preferences:**

| | v2 preferred | v1 preferred | Tie |
|---|---|---|---|
| Run 1 (A = v1) | 38 | 27 | 35 |
| Run 2 (A = v2) | 42 | 24 | 34 |
| Same pick in both runs (81 of 100) | 33 | 21 | 27 |

Net preference for v2 (v2 wins minus v1 wins, averaged over both runs): **+0.14 [−0.01, +0.28]**. That's a lean towards v2, not a proven win. By language, v2 leads in French in both runs. English and Spanish are close, except that Spanish favours v2 in run 2 (15 to 7).

**Sonnet's criteria** (similar in both runs): meaning (who, what, negation, tense), accuracy of each emoji, idioms and abstract ideas, clarity and economy (no padding or repetition), and details such as times, places and people.

**Sonnet's conclusions** (both runs give v2 a slight edge and call the two close):
- **Where v2 is better:**
  - It reliably shows who acts (🙋, 👥, 🫵) and keeps the verb.
  - It gets times and numbers right.
  - Its outputs are cleaner, without v1's repetition (🍲🍲🍲🍲, 🌧️🌧️🌧️🌧️).
  - It makes fewer outright errors (v1 uses 🏰 for the Eiffel Tower, and clocks showing the wrong hour).
- **Where v1 is better:**
  - It keeps more concrete details (Paris 🗼, desert 🏜️, Thursday, 💍 at a wedding).
  - It handles idioms better ("llover a cántaros", "break a leg").
  - It's richer in emotion.
- **v2's errors:** 🥛 for "cántaros", 🐗 for a lion's roar, 🍪 for a sandwich, a mechanic for a firefighter, 🌅 for sunset.

**Limits of this check:**
- **Short, simple sentences:** they average 7 words (the longest has 11), and only 4 are questions. Long sentences and questions, which v2 targets, are barely tested.
- **One judge:** Sonnet is a single model with its own taste. A preference is not the same as the round-trip score.
- **v2 setup:** v2 ran as an adapter on the 8-bit base, which differs slightly from the merged model.
- **Cost:** 3 Sonnet calls, about 13k input and 14k output tokens, about $0.16.

Files: `results/blind_review/` (`sentences.csv`, `outputs.csv`, `eval_1.json`, `eval_2.json` with Sonnet's full answers).

## Decision

<!-- Your decision and why. -->
