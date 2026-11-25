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

## Decision

<!-- Your decision: what you take from this step and what comes next. -->
