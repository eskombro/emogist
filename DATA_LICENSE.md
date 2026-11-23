# Data licences

The code is under the [MIT License](LICENSE). The data has several origins, so each part keeps the most open terms it can.

| Data | Licence |
|---|---|
| **Real rows** (`source` = `real`): Tatoeba sentences and their emoji labels | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) |
| **AI rows** (`source` = `ai`): DeepSeek-generated sentences and their emoji labels | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), to the extent any rights exist |
| **Few-shot worksheet** (`03_dataset/label_examples.csv`) | Rows marked Tatoeba: CC BY 2.0 FR. All other rows: CC0 1.0 |
| **Evaluation material**: `data/test_sentences.csv`, `cache/`, and the `results/` and `runs/` folders | Not openly licensed, see below |

These rules apply to `03_dataset/full/`, `pilot/`, `labels/`, `review/` and `sentences/`. In files without a `source` column, IDs starting with `ai-` are AI rows and all others are real rows.

## Using real rows

Credit the sentence's author and Tatoeba, link to the licence, and state any changes. For example:

> Sentence by `<author>`, from [Tatoeba](https://tatoeba.org), CC BY 2.0 FR. Emoji label added by the emogist project (AI-generated).

Authors are in the `author` field of `train.jsonl` and in `03_dataset/sentences/real_pool.csv` and `real_dev.csv`. The sentence text is unchanged from Tatoeba. The emoji labels are treated as adaptations, so they stay under the same licence.

## Evaluation material

These files contain text written by Claude (Anthropic): test sentences drafted with Claude, decodes and judgements. They are published so the results can be checked and reproduced. **They are not licensed for training AI models**, because Anthropic's terms restrict that use. Tatoeba sentences that appear in them remain under CC BY 2.0 FR.

## No warranty

The data is provided as is. AI-generated content may be wrong, and no endorsement by Tatoeba, its contributors, DeepSeek or Anthropic is implied.
