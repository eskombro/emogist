# Step 5: Evaluation

Check how far the scores can be trusted, how close fine-tuned Qwen gets to its teacher, and where it still loses the gist.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** done. The judge passed a one-person check, fine-tuned Qwen closed 64% of the gap to its teacher on the test set, and long sentences are its clearest weakness on dev.

# Main features and functionalities

- **Human check of the judge:** blind 0–2 ratings compared with the judge's scores.
- **Teacher ceiling:** the teacher scored on the same test set, with the final labelling prompt.
- **Error breakdown:** aggregate results by language, length and sentence type, with no new model calls.

# Main parts of the project

| Path | Purpose |
|---|---|
| `judge_check.py` | Creates the blind rating sheet and compares the ratings with the judge |
| `score_teacher.py` | Runs and scores the teacher on the test set |
| `error_breakdown.py` | Breaks down the dev scores of Qwen and the teacher |
| `results/judge_check_sheet.csv` | The 52 pairs with the author's ratings, published so the check can be reproduced |
| `results/teacher_test_outputs.csv`, `teacher_test_scores.csv` | Teacher outputs, decodes and scores on the test set |
| `results/error_breakdown.csv` | Scores by group on dev |

## Can the judge be trusted?

The project author rated 52 pairs, each made of an original test sentence and the decoder's guess, on the judge's 0–2 scale. The pairs were drawn at random, 13 per model and language, from the untrained and fine-tuned Qwen outputs. The model and the judge's score were hidden. The pass mark, a weighted kappa of at least 0.6, was set before the comparison.

| Measure | Result |
|---|---|
| Weighted kappa | **0.66** [0.52, 0.78] |
| Exact agreement | 58% (30 of 52) |
| Opposite calls (0 vs 2) | 0 |
| Average score, human | 1.19 |
| Average score, judge | 0.92 |
| Human − judge | +0.27 [+0.12, +0.42] |

The judge passes, by a thin margin. It is stricter than the human rater, who scored higher on 18 pairs and lower on 4. The offset is +0.27 for both the untrained and the fine-tuned model, so comparisons between models look more reliable than absolute scores.

This has clear limits:

- There is **one rater**, who had already seen the test sentences.
- It checks the **judge only**. The decoder would need people who have not seen the test set.

## How close is Qwen to its teacher?

Gist score on the 100 test sentences (0–2, invalid outputs count as 0). The teacher gets the full labelling prompt and one attempt, without the retry and filters used to build the dataset.

| Model | Valid | French | English | **Overall** | Lost / partly / fully kept |
|---|---|---|---|---|---|
| Qwen3.5 2B, untrained, full prompt | 99 | 0.96 | 0.66 | **0.81** | 40 / 39 / 21 |
| **Qwen3.5 2B, fine-tuned** | **100** | **1.08** | **1.18** | **1.13** | **21 / 45 / 34** |
| Teacher (DeepSeek V4.1 Flash) | 98 | 1.36 | 1.26 | **1.31** | 16 / 37 / 47 |

Teacher minus fine-tuned Qwen, with 95% bootstrap intervals:

- Overall: **+0.18 [+0.03, +0.33]**
- French: +0.28 [+0.10, +0.46]
- English: +0.08 [−0.14, +0.32], not proven

Fine-tuning closed 64% of the gap between the untrained model and the teacher (0.32 of 0.50). Qwen receives only a one-line instruction, while the teacher receives the full rules and 20 examples.

## Where does Qwen lose the gist?

This breakdown uses the 100 real dev sentences, so that any idea for retraining stays away from the test set. It is exploratory: the groups come from simple text rules, they overlap, and many are small.

| Sentence length | Sentences | Qwen | Gist lost | Teacher | Teacher − Qwen |
|---|---|---|---|---|---|
| Up to 5 words | 21 | 1.00 | 29% | 1.38 | +0.38 [+0.00, +0.76] |
| 6–8 words | 53 | 0.72 | 43% | 0.74 | +0.02 [−0.15, +0.19] |
| **9+ words** | **26** | **0.62** | **50%** | **1.04** | **+0.42 [+0.15, +0.65]** |

- **Long sentences are the clearest weakness.** Qwen loses the gist on half of them.
- **No language conclusion.** The teacher's lead is larger in English on dev (+0.26 vs +0.14) but larger in French on test.
- **Number format is fixed.** The final Qwen and Ministral models wrote no plain digits on dev, unlike the pilot.
- **Number meaning is not measured.** Dev has only 2 sentences with a number.
- **Other gaps in dev:** no questions, 3 sentences with cause or result, 6 with reported speech and 7 about the future.

## Limitations

- The test set was used to shortlist models in steps 1 and 2, so absolute test scores may be slightly optimistic.
- Test and dev scores are not comparable, because the test sentences are easier.
- The decoder (Claude Haiku 5.5) has not been validated against people.
- Ministral 3 3B has no test score.

## Decision

<!-- Your decision and why. Facts to weigh:
     - Judge check: kappa 0.66 [0.52, 0.78], judge stricter by 0.27, one rater.
     - Test: untrained 0.81, fine-tuned 1.13, teacher 1.31. Teacher - Qwen +0.18 [+0.03, +0.33].
     - Dev: long sentences are the clearest weakness (+0.42 behind the teacher, 50% lost).
     - Retraining for long sentences would first need a larger, more varied dev set. -->

## Next

Step 6: serve the fine-tuned Qwen with GGUF and FastAPI, with a demo page.

## Reproduce

```bash
python 05_evaluation/judge_check.py --score   # compare the saved ratings with the judge
python 05_evaluation/score_teacher.py         # reuses the saved outputs and cached scores
python 05_evaluation/error_breakdown.py       # no model calls
```

Running `judge_check.py` without `--score` creates a new blank sheet, and refuses to overwrite an existing one.
