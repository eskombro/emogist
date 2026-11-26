# Step 7: v2 rules and scoring

Relax the format rules so that short answers are allowed, and replace the single 0–2 gist score with three criteria, so that a related guess scores above an unrelated one. This is the first step of v2. Steps 0 to 6 keep the v1 rules and scores unchanged.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** rules and rubric written; the rubric is not validated yet. Human ratings with the new rubric come after the new dev and test sets, which add Spanish.

# Main features and functionalities

- **v2 format rules:** 1 to 10 emojis instead of 3 to 10. Fewer than 3 is a style warning, not an error.
- **v2 judge:** scores topic, event and outcome separately and combines them into a score from 0 to 1.
- **Versioned, not replaced:** v1 rules, prompt and cache stay the defaults, so every earlier result can still be reproduced.

# Main parts of the project

| Path | Purpose |
|---|---|
| `compare_rules.py` | Checks every saved output with the v1 and v2 rules; no model calls |
| [`../emoji_eval/validate.py`](../emoji_eval/validate.py) | `check_validity_v2` and `check_conventions_v2`, next to the unchanged v1 functions |
| [`../emoji_eval/roundtrip.py`](../emoji_eval/roundtrip.py) | `JUDGE_PROMPT_V2`, `judge_v2` and `criteria_score`, with their own cache `cache/judge_v2_cache.csv` |

## Format rules

| | v1 (steps 0–6) | v2 (step 7 on) |
|---|---|---|
| Emoji count, ➡️ not counted | 3 to 10 | **1 to 10** |
| Emojis only, single line | error | error |
| Fewer than 3 emojis | error | **warning** |
| Other style conventions | warnings | same warnings |

The 3-emoji minimum came from the step 0 conventions and was never measured. It had three side effects:
- **Training:** labels with fewer emojis were filtered out, so the model never learned short answers.
- **Scoring:** a short answer scored 0, like a wrong one.
- **Demo:** the demo refused to show a short answer at all.

v2 discourages short answers instead of forbidding them. The teacher prompt will ask for as few emojis as the idea needs, usually 3 to 8. The training data then shows short answers only where the teacher found them fitting.

Planned wording for rule 1 of the teacher prompt, to be applied when the v2 dataset is built:

> Use as few emojis as the idea needs, usually 3 to 8 and never more than 10. One or two are fine only when the sentence is that simple.

The share of short labels will be checked in a pilot before the full build.

### Effect on saved outputs

`compare_rules.py` checked the 5,434 distinct outputs saved in the repository: student and teacher outputs, labels, and samples.

| | v1 | v2 |
|---|---|---|
| Invalid | 174 | 97 |
| Too few emojis | 77 | 0 |
| Too many emojis (11 or more) | 75 | 75 |
| Non-emoji characters | 22 | 22 |

The 77 newly valid outputs are mixed. Some are clear (`🍽️🧼`, `🚗 ➡️ 🏥`), and others are vague (`✅`, `🤔`). This is why short answers stay a warning and are not encouraged.

All 6,056 saved outputs, including the cache, give the same v1 result as before this step.

## Scoring rubric

The v1 judge gives one gist score: 2 kept, 1 partly, 0 wrong. In practice, a guess that gets the topic right but the facts wrong often scores 0, the same as an unrelated guess. The 0–2 scale also forced the human rater into choices they weren't comfortable with.

The v2 judge scores three criteria. Each one gets **2 = yes, 1 = partly, 0 = no**:

| Criterion | Question |
|---|---|
| **Topic** | Is it about the same situation or subject? |
| **Event** | Does the same main thing happen, with the right people doing it? Swapped roles or a negated event make it wrong. |
| **Outcome** | Does it end the same way: result, consequence, or the emotion the story is about? **n/a** when the original has no outcome beyond its event. |

The score is the share of the points available, from 0 to 1. An n/a outcome is left out, not counted as a 0.

| Topic | Event | Outcome | Score |
|---|---|---|---|
| 2 | 2 | 2 | 1.00 |
| 2 | 0 | 0 | 0.33 (related, wrong facts) |
| 2 | 2 | n/a | 1.00 |
| 1 | 1 | 1 | 0.50 |
| 0 | 0 | 0 | 0.00 |

What doesn't change:
- **Decoder:** Haiku, one guess per output.
- **Judge model:** Sonnet, which only sees the two sentences.
- **v1 gist score:** stays available, so v2 results can also be compared with steps 0 to 6.

The game's local judge (untrained Qwen) will use the same criteria, so game scores and evaluation scores mean the same thing.

## Validation plan

The rubric is used for results only after a human check, set up the same way as in step 5:
- the project author rates about 60 pairs per criterion, in English, French and Spanish, with the scores hidden;
- the pairs come from the new dev sets, so this waits for the next step;
- **pass mark, set before rating:** a rank correlation of at least 0.7 between the human total and the judge's total. On the v1 ratings, the Sonnet judge reached 0.71.

## Human check

**Result: the v2 judge missed the pass mark (0.67 vs 0.7).** The v1 judge reached 0.72 on the same pairs. The intervals overlap almost entirely, so neither judge is clearly better.

**Setup:**
- The v1 model (step 6, 8-bit) wrote emoji sentences for the 300 v2 dev sentences. 295 are valid under the v2 rules: EN 100, FR 97, ES 98.
- Haiku decoded all 295.
- The author rated 60 blind pairs, 10 per language and source (seed 7), seeing only the original and the guess.
- Both Sonnet judges then scored the same pairs.

**Files** (all in `results/`): `rubric_ratings.csv` (ratings), `rubric_scores.csv` (per-pair judge scores), `rubric_pairs.csv` (key). Run `rubric_check.py --score --round 1` to redo the comparison; it reads the judge caches.

| | Author | v2 judge | v1 judge |
|---|---|---|---|
| Mean score (0–1) | 0.57 | 0.41 | 0.34 |
| Rank correlation with author | — | 0.67 [0.48, 0.81] | 0.72 [0.56, 0.83] |

v2 judge against the author on each criterion (weighted kappa):

| Criterion | Kappa | Pairs |
|---|---|---|
| Topic | 0.49 [0.30, 0.66] | 60 |
| Event | 0.49 [0.31, 0.65] | 60 |
| Outcome | 0.48 [0.24, 0.70] | 43 |

**Where they disagree:**
- **The judge is stricter on every criterion.** Where the two differ, the author gave the higher score 21 times out of 25 on topic, 26 out of 30 on event, and 16 out of 18 on outcome.
- **Hypotheticals.** The judge counts a conditional or a hypothetical that the guess turns into a real event as a different event. The author mostly accepts it.
  - Pair 28: "If my brother's car stays broken, he'll miss his daughter's birthday" → "The boy was in a car crash, so the girl missed her birthday party". Author 2/2/1, judge 1/0/0.
- **Who did what.** The judge treats a change of doer as a wrong event.
  - Pair 58: "we" fixed the leak ourselves → "the plumber" fixed it. Author 2/2/2, judge 1/0/0.
- **n/a outcome.** The judge chose n/a for 17 pairs; the author never used it. Leaving the author's outcome out on those 17 pairs, as a post-hoc check, keeps the correlation at 0.67, so n/a is not the cause.
- **By language** (20 pairs each, so only a rough guide): the rank correlation is 0.52 in English, 0.75 in French and 0.76 in Spanish.

### Round 2: revised rubric (v2.1)

**Rules the author set after round 1.** Emojis can't show either of these, so the judge no longer penalises them:
- **Mood.** A condition, wish, plan, request or question guessed as something that really happened (or the reverse) counts as the same event.
- **Who exactly the people are**, beyond number, gender, rough age and a visible job. "My brother" and "a boy" are the same people. A different person doing the action is not an error.

**What still counts as wrong:** reversed roles (who did what to whom is backwards) and negation.

**Code:** the revised prompt is `JUDGE_PROMPT_V21` in `emoji_eval/roundtrip.py`, with its own cache (`cache/judge_v21_cache.csv`). Round 1 results stay reproducible.

**Plan, set before rating:**
- 42 fresh pairs (7 per language and source, seed 8), with none from round 1. File: `results/rubric2_sheet.csv`.
- Same rating instructions as round 1, plus the two rules above.
- The prompt is not changed after this point.
- **Pass mark:** the same, a rank correlation of at least 0.7 between the author's total and the v2.1 judge's total.
- **If it fails:** the v1 judge stays the main score for v2.

**Result: the v2.1 judge passed (0.77 vs 0.7).** At the author's request, the v1 judge also ran on the same 42 pairs. Files: `results/rubric2_ratings.csv` and `results/rubric2_scores.csv`.

| | Author | v2.1 judge | v1 judge |
|---|---|---|---|
| Mean score (0–1) | 0.60 | 0.44 | 0.42 |
| Rank correlation with author | — | 0.77 [0.60, 0.88] | 0.64 [0.40, 0.80] |

- **Versus v1:** v2.1 is ahead by 0.13 [−0.03, +0.31] on the same pairs. That's likely but not certain.
- **Per criterion** (weighted kappa): topic 0.54 [0.31, 0.71], event 0.61 [0.43, 0.75], outcome 0.45 [0.21, 0.65] on 28 pairs. The author used n/a 5 times, the judge 11.
- **By language** (14 pairs each): rank correlation 0.67 in English, 0.72 in French, 0.84 in Spanish.
- **The judge is still stricter than the author.** Where they differ on a criterion, the author's score is higher 43 times out of 52. Comparisons between models depend on ranking, not on the scale, so this doesn't affect them. Scores should be read as "judge points", not as the author's.

## Baseline: the v1 model on the v2 dev set

This is the score v2 training has to beat. Setup:
- **Model:** the served v1 model (8-bit, step 6), at temperature 0.
- **Data:** all 300 dev sentences.
- **Judging:** Haiku decodes each output and the v2.1 judge scores it. An output that breaks the v2 format rules scores 0 (5 of 300).
- **Script:** `score_v1_dev.py`. It writes `results/v1_dev_scores.csv`.

Scores are from 0 to 1, with 95% intervals:

| | Real | AI | All |
|---|---|---|---|
| English | 0.42 [0.33, 0.51] | 0.53 [0.45, 0.61] | 0.47 [0.41, 0.53] |
| French | 0.45 [0.36, 0.55] | 0.47 [0.39, 0.56] | 0.46 [0.40, 0.53] |
| Spanish | 0.41 [0.33, 0.49] | 0.56 [0.48, 0.65] | 0.49 [0.42, 0.55] |
| **All** | 0.43 [0.38, 0.48] | 0.52 [0.47, 0.57] | **0.47 [0.44, 0.51]** |

- **Spanish is already level with English and French**, although the v1 model never saw Spanish in training.
- **Real sentences are harder than AI sentences** (0.43 vs 0.52), as in v1.
- **Event is the weakest criterion.** Mean points out of 2: topic 1.24, event 0.80, outcome 0.76 (outcome on the 204 outputs where it applies). The model usually gets the situation across but loses what happened.
- **By type** (18 to 75 sentences each, wide intervals): questions 0.42, long sentences 0.43 and real "other" sentences 0.44 score lowest. Mishap (0.63) and conditional (0.60) sentences score highest.

## Decision

<!-- Your decision: what you take from this step and what comes next. -->
