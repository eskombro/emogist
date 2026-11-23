# Step 3: Dataset

Build about 3,000 English and French training examples, each pairing a sentence with an emoji sentence from the teacher.

<!-- Your summary: what this step is about, in your own words. -->

**Status:** full dataset built: 3,000 examples, described in the [dataset card](full/README.md).

# Main features and functionalities

- **Real sentences** from Tatoeba, filtered towards everyday situations about people.
- **AI-generated sentences** driven by a grid of topics, patterns, feelings, lengths and speakers.
- **Human-reviewed labelling prompt** with 20 examples written or checked by the author.
- **Resumable labelling** with automatic format checks and one retry.
- **Pilot training sets** to measure which source of sentences teaches the student best.

# Main parts of the project

| Path | Purpose |
|---|---|
| `fetch_real_sentences.py` | Downloads and filters Tatoeba, writes the real pool and dev set |
| `blocklist.py` | Per-language word lists for sexual, violent and hateful content |
| `generate_sentences.py` | Generates AI sentences with DeepSeek V4.1 Flash |
| `pick_example_candidates.py`, `label_examples.csv` | Worksheet for the human-written few-shot examples |
| `label_prompt.py`, `compare_prompts.py` | Prompt versions and their comparison on the dev set |
| `label_pilot.py` | Labels the pilot sentences and builds sets A, B and C |
| `build_dataset.py` | Labels the full dataset and writes `full/train.jsonl` |
| `sample_labels.py` | Draws a random sample for human review |
| `sentences/`, `labels/`, `pilot/`, `full/`, `review/`, `results/` | Generated data, review sheet and comparison results |

## Real sentences

Source: [Tatoeba](https://tatoeba.org) exports, **CC BY 2.0 FR**. Author and sentence ID are kept for attribution.

| | All | Original | By self-declared natives | Clean format | Everyday, after blocklist | Kept |
|---|---|---|---|---|---|---|
| **English** | 2,041,577 | 1,421,595 | 1,092,593 | 914,517 | 388,552 | 3,050 |
| **French** | 731,363 | 146,059 | 104,963 | 73,969 | 40,497 | 3,050 |

Rules keep 5–25 word statements that mention a person, and drop questions, years, sentences with several proper nouns, and impersonal French such as "il faut". Selection also limits authors, stock names such as "Tom", repeated openings and near-duplicates. The 3,050 kept per language become 50 **dev** and 3,000 **pool** sentences, with no overlap with the test set.

The filters are simple. Some abstract sentences and grammar errors remain, and the blocklist removes a few harmless sentences such as "My neck is killing me."

## AI-generated sentences

DeepSeek V4.1 Flash (MIT) writes one sentence per request at temperature 0.9. Each request combines one of 48 topics, 6 patterns (speech, past, future, mishap, moment, numbers), 7 feelings, 3 lengths and 3 speakers. It also shows three real Tatoeba sentences with the same pattern. French is written directly in French.

Claude does not write these sentences or their examples, because they become training data.

## Labelling prompt

The teacher is DeepSeek V4.1 Flash, without thinking, at temperature 0.3. Prompt versions were compared on the 100 real dev sentences:

| Version | Change | Valid | Gist FR | Gist EN | Gist overall |
|---|---|---|---|---|---|
| v1 | Prompt from steps 1 and 2 | 95 | 0.94 | 0.80 | 0.87 |
| v2 | Keycap numbers banned | 93 | 1.00 | 0.74 | 0.87 |
| **v4** | Keycaps allowed for exact numbers, up to 4 repeats, 20 human-reviewed examples | **99** | 1.02 | 0.88 | **0.95** |

v4 − v1 is +0.08 [−0.06, +0.23], so the gain is likely but not proven. Sentences that lost the gist fell from 41 to 30. v3 was an intermediate rule that was never run.

**Keycap decision:** keycaps were first banned, but that lost ages, times and prices without improving the gist. They are now allowed when the exact number matters. Malformed forms such as `10️⃣` remain invalid.

After this comparison, the author made small edits to reduce overused ➡️ arrows (13 → 11 of 20 examples). The pilot labels use that final worksheet, recorded as `v4-human-examples@4addf4cb`.

## Pilot data

880 sentences were labelled (220 per language and source), and **858 passed** the format checks. 16 needed the retry.

| Training set | Contents | Avg. words | ➡️ | 💬 | ⏪/⏩ |
|---|---|---|---|---|---|
| **A: real** | 200 real per language | 8.3 | 36% | 19% | 7% |
| **B: AI** | 200 AI per language | 14.8 | 68% | 23% | 26% |
| **C: mixed** | 100 real + 100 AI per language | 11.3 | 52% | 23% | 17% |

The sets differ in length and style as well as source. Their training results are in [step 4](../04_fine_tuning/).

## Full dataset

Each language has 750 real and 750 AI sentences, of which 125 use the numbers pattern. The teacher labelled 3,310 sentences and 3,150 passed the checks. The first valid labels in pool order filled each quota, and the 880 pilot labels were reused.

English numbers sentences failed most often (19% in the first batch), because each keycap digit counts toward the 10-emoji limit. 78% of the numbers examples contain keycaps, against 11% of the whole dataset.

The author read part of a random 120-example sample and its numbers rows, and judged the labels ready. This was an informal check with no recorded score.

## Limitations

- Automatic checks verify **format, not meaning**, so a valid label can still be poor.
- One Tatoeba contributor wrote 384 of the 1,500 real sentences.
- The dev set has now informed several decisions, so the final result must come from the test set, which was kept out of training and prompt tuning.

## Decision

<!-- Your decision and why. Facts to weigh:
     - Mixed real + AI data scored best (0.68), although its lead over A and B is not proven.
     - AI sentences alone hurt English (0.40).
     - Planned: 750 real + 750 AI sentences per language, plus a numbers pattern. -->

## Next

Train both students on the full dataset in [step 4](../04_fine_tuning/).

## Reproduce

```bash
python 03_dataset/fetch_real_sentences.py
python 03_dataset/generate_sentences.py --per-lang 665 --numbers-per-lang 170
python 03_dataset/compare_prompts.py --versions v1-current v4-human-examples
python 03_dataset/label_pilot.py
python 03_dataset/build_dataset.py
python 03_dataset/sample_labels.py
```

These need Ollama signed in for DeepSeek V4.1 Flash. `compare_prompts.py` also needs `ANTHROPIC_API_KEY`. Completed work is reused, so reruns only process what's missing.
