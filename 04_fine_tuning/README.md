# Step 4: Fine-tuning

Train a small LoRA adapter so a student model learns to write emoji sentences from the teacher's labels.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** both students trained on the full dataset and Qwen declared the winner. On the final test, fine-tuning raised Qwen's gist score from 0.81 to 1.13. The Ministral test run is deferred.

# Main features and functionalities

- **LoRA fine-tuning** with Unsloth on a free Google Colab T4 GPU.
- **Separate notebooks** for Qwen3.5 2B and Ministral 3 3B.
- **Safety checks** against training twice or generating before training has finished.
- **Round-trip scoring** on 100 real dev sentences the students never train on.

# Main parts of the project

| Path | Purpose |
|---|---|
| `pilot_train_qwen.ipynb` | Trains and exports a Qwen3.5 2B adapter |
| `pilot_train_ministral.ipynb` | Same recipe for Ministral 3 3B |
| `train_qwen.ipynb`, `train_ministral.ipynb` | Full-dataset training, with Drive checkpoints and best-epoch selection |
| `test_qwen.ipynb`, `test_ministral.ipynb` | Final test run, gated by a check that the adapter reproduces its dev outputs |
| `score_pilot.py` | Scores all runs against the untrained baseline and the teacher |
| `score_test.py` | Scores the final test outputs against the untrained baseline; includes Ministral if its outputs exist |
| `runs/` | Dev outputs, settings and loss history of each run (adapters are not committed) |
| `results/pilot_scores.csv`, `full_scores.csv`, `test_scores.csv` | Per-sentence outputs, decodes and scores |

## Method

Each run starts again from the original model, with 3 epochs, 360 training and 40 validation examples, batch size 4 with 2 accumulation steps, learning rate 0.0002 and LoRA rank 16. The trained model receives only a one-line instruction, while the untrained baseline receives the full rules and 20 examples.

## Results

Gist score on 100 real dev sentences (0–2, invalid outputs count as 0):

| Model | Valid | Gist FR | Gist EN | **Gist overall** | Lost / partly / fully kept | vs. untrained Qwen |
|---|---|---|---|---|---|---|
| Qwen 2B, prompted, untrained | 85 | 0.34 | 0.50 | **0.42** | 65 / 28 / 7 | |
| Qwen 2B + A (real) | 94 | 0.64 | 0.54 | **0.59** | 56 / 29 / 15 | +0.17 [+0.03, +0.32] |
| Qwen 2B + B (AI) | 97 | 0.72 | 0.40 | **0.56** | 60 / 24 / 16 | +0.14 [+0.00, +0.29] |
| **Qwen 2B + C (mixed)** | **97** | 0.70 | 0.66 | **0.68** | **48** / 36 / 16 | **+0.26 [+0.12, +0.41]** |
| Ministral 3B + C (mixed) | 93 | 0.62 | 0.50 | **0.56** | 53 / 38 / 9 | |
| *Teacher (DeepSeek V4.1 Flash)* | *99* | *1.02* | *0.88* | *0.95* | *30 / 45 / 25* | *+0.53 [+0.37, +0.69]* |

Paired differences:

- C − A: +0.09 [−0.02, +0.20]
- C − B: +0.12 [−0.01, +0.24]
- Ministral + C − Qwen + C: −0.12 [−0.25, +0.02]

The mixed set is the best choice but its lead is not proven. Qwen is probably ahead of Ministral, also not proven. Untrained Ministral was not measured on the dev set, so its training gain is unknown.

Validation loss by epoch:

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Qwen + A | 0.750 | **0.722** | 0.740 |
| Qwen + B | 0.971 | **0.938** | 0.954 |
| Qwen + C | 0.927 | **0.887** | 0.893 |
| Ministral + C | 1.059 | 0.948 | **0.938** |

Qwen begins to overfit after epoch 2 on these small sets, while Ministral was still improving. Each run validates on its own held-out examples, so loss values are not comparable across runs.

## Full-dataset results

Both students trained on 2,700 examples and validated on 300, keeping the epoch with the lowest validation loss (epoch 2 for both). Scores are gist on the 100 real dev sentences.

| Model | Valid | French | English | Overall | Adapter |
|---|---|---|---|---|---|
| Qwen3.5 2B | 98 | 0.88 | 0.62 | 0.75 | 44 MB |
| Ministral 3 3B | 96 | 0.66 | 0.78 | 0.72 | 99 MB |
| Teacher (DeepSeek V4.1 Flash) | 99 | 1.02 | 0.88 | 0.95 | |

Ministral − Qwen is −0.03 [−0.17, +0.11], so the two are tied overall. Qwen leads in French by 0.22 [+0.04, +0.40]. Ministral's English lead of 0.16 [−0.06, +0.38] is not proven.

**Qwen was declared the winner before the test set was run.** Ministral is measured on the test set for comparison only, and its result does not change this choice.

## Final test results

Qwen3.5 2B ran once on the 100 test sentences (50 French, 50 English). The untrained baseline is the same model through Ollama with the full labelling prompt (rules + 20 examples). The fine-tuned model receives only the one-line instruction.

| Qwen3.5 2B | Valid | French | English | **Overall** | Lost / partly / fully kept |
|---|---|---|---|---|---|
| Untrained, full prompt | 99 | 0.96 | 0.66 | **0.81** | 40 / 39 / 21 |
| **Fine-tuned** | **100** | **1.08** | **1.18** | **1.13** | **21 / 45 / 34** |

Paired gains from fine-tuning, with 95% bootstrap intervals:

- Overall: **+0.32 [+0.17, +0.47]**
- English: +0.52 [+0.32, +0.72]
- French: +0.12 [−0.08, +0.32], not proven

How to read these numbers:

- **Test and dev scores are not comparable.** The untrained model scores 0.81 on test and 0.42 on dev, so the test sentences are easier.
- **The test set was used to shortlist models in steps 1 and 2**, so it is not untouched and absolute scores may be slightly optimistic.
- **The teacher scores 1.31 on this test** with the final labelling prompt, as measured in [step 5](../05_evaluation/).
- **The scorer is model-based and provisional:** Claude Haiku 5.5 decodes and Claude Sonnet 5.5 judges. Step 5 checks the judge against one human rater, and the decoder is not validated yet.
- **Ministral 3 3B has not been run on the test set**, because the Colab GPU quota ran out. It is optional, as Qwen was chosen before the test.

## Lessons and limitations

- Fine-tuning replaced a long prompt with a one-line instruction and still improved the score.
- **Adding number examples fixed the number format:** the pilot model wrote `30` or `13-14` as plain digits, while both final models wrote none on dev. Dev has too few number sentences to show whether the meaning is kept.
- These are single runs on 100 dev sentences, and that dev set has informed several decisions.
- Loss measures closeness to the teacher's exact emojis, not whether the gist is preserved.
- Rerunning a training cell continues from the trained model. One run was discarded for this reason, and the notebooks now block it.

## Decision

<!-- Your decision and why. Facts to weigh:
     - Full data, dev: Qwen 0.75, Ministral 0.72, tied (-0.03 [-0.17, +0.11]).
     - Qwen leads in French on dev; its adapter is less than half the size (44 MB vs 99 MB).
     - Qwen was chosen before the test. Test: 0.81 untrained -> 1.13 fine-tuned, +0.32 [+0.17, +0.47].
     - Ministral has no test score yet. -->

## Next

[Step 5](../05_evaluation/) evaluates the fine-tuned Qwen in more depth. Optionally, run `test_ministral.ipynb` once when a GPU is available, put its outputs in `runs/run_ministral3b_full/`, and rerun `score_test.py`.

## Reproduce

Open a notebook in Colab, select a **T4 GPU**, run the cells in order, run step 7 **once**, and put the downloaded zip in `runs/`. Unzip it into `runs/run_<name>/`, then run:

```bash
python 04_fine_tuning/score_pilot.py         # pilot runs
python 04_fine_tuning/score_pilot.py --full  # full-dataset runs
python 04_fine_tuning/score_test.py          # final test
```
