# emogist

Training a small open model to turn everyday English and French sentences into emoji sentences that keep the gist.

> "I fell asleep on the beach and got a terrible sunburn." → `🏖️😴 ➡️ 🦞🔥 😭`

<!-- Your intro: why this project, what you want to learn. -->

# Main features and functionalities

- **Full learning loop:** dataset → fine-tuning → evaluation → serving, documented step by step.
- **Emoji grammar:** a small set of rules for time, speech, negation, numbers and cause, in [CONVENTIONS.md](CONVENTIONS.md).
- **Teacher/student training:** an open teacher model labels sentences, and a small open student learns from them with a LoRA adapter.
- **Round-trip evaluation:** one model decodes the emojis without seeing the original, and another judges whether the gist survived.
- **Bilingual data:** real human sentences plus AI-generated everyday stories, written natively in each language.

# Main parts of the project

## How it works

```
                 teacher model (labels)            small student model (fine-tuned)
sentences ──▶  sentence → emoji sentence  ──▶  dataset  ──▶  sentence → emoji sentence  ──▶  API + demo page
```

Every step is evaluated the same way: a **round-trip score**. One model decodes the emoji sentence back into text without seeing the original, and another judges whether the decoded text keeps the gist (see [step 1](01_student_selection/)).

## Steps

| Step | Status | Key result |
|---|---|---|
| [0. Exploration](00_exploration/) | ✅ done | Small models need one sentence per request; validation is mandatory |
| [1. Student selection](01_student_selection/) | ✅ done | Qwen3.5 2B and Ministral 3B tied; Qwen3.5 0.8B behind |
| [2. Teacher selection](02_teacher_selection/) | ✅ done | Five large teachers tied at ~1.3 gist vs ~0.9 for students; thinking adds nothing; DeepSeek V4.1 Flash best on tie-breakers, gpt-oss 120B a credible backup |
| [3. Dataset](03_dataset/) (3,000 EN+FR sentences) | ✅ done | 50/50 mix of real and AI sentences, including exact numbers. [Dataset card](03_dataset/full/README.md) |
| [4. Fine-tuning](04_fine_tuning/) (LoRA, Unsloth) | ✅ done | Qwen 2B chosen on dev, tied with Ministral 3B. On the test set, fine-tuning raised gist from 0.81 to 1.13 (+0.32 [+0.17, +0.47]). Ministral's test run is deferred |
| [5. Evaluation](05_evaluation/) | ✅ done | Judge passed a one-person check (kappa 0.66). Fine-tuned Qwen 1.13 vs teacher 1.31 on test, closing 64% of the gap. Long sentences are its clearest weakness |
| [6. Serving](06_serving/) (GGUF + FastAPI, [live demo](https://emogist.esklab.com)) | ✅ done | The 8-bit model (2 GB) matches the full model on dev (0.75). It loads on demand on a shared VPS |
| [7. v2 rules and scoring](07_v2_rules_and_scoring/) | 🔄 in progress | Short answers (1–2 emojis) allowed as a warning. The three-criteria judge (topic, event, outcome) missed its first human check (0.67 vs 0.7). After two rules were added (mood and exactly who the people are are not penalised), v2.1 passed on 42 fresh pairs: 0.77 [0.60, 0.88], against 0.64 for the v1 judge. Baseline of the v1 model on v2 dev: 0.47 [0.44, 0.51], with Spanish (never trained) level with English and French. v1 results unchanged |
| [8. Spanish and new evaluation sets](08_v2_spanish_and_eval_sets/) | ✅ done | New dev and test sets: 300 sentences each, 100 per language (EN, FR, ES), half real and half AI, all reviewed by the author (79 of 679 removed) |
| [9. v2 dataset and labels](09_v2_dataset_and_labels/) | ✅ done | 6,000 training rows: 2,000 per language (EN, FR, ES), half real and half AI. One label per sentence from DeepSeek V4.1 Flash with the v2 prompt and the author's examples (best-of-n gave no gain). New signs ❓ 💭 🤔 ❔ 📍 🔁; 🔀 and 🔄 tried and dropped. Author's sample review: 7% of labels judged wrong |
| [10. v2 fine-tuning](10_v2_fine_tuning/) | ✅ done | New LoRA on the original Qwen 2B with v1 settings: main (EN+FR+ES) 0.47, the same as v1 on v2 dev; the EN+FR control shows Spanish neither hurt nor helped. A blind Sonnet review leans towards v2 (+0.14, not proven). 4B scores 0.52 (+0.05, proven) but was not kept: twice the size for a small gain |

## Repository layout

```
emoji_eval/          shared code: prompt, validator, model runner, round-trip scorer
data/                fixed test set (evaluation only) + data card
cache/               cached decodes and judgements, so scores are reproducible for free
00_exploration/      first manual tests
01_student_selection/  compare small models before training
02_teacher_selection/  compare large models that label the data
03_dataset/          sentence sources, labelling prompt, pilot data
04_fine_tuning/      Colab training notebooks, pilot runs and scores
05_evaluation/       human check of the judge, teacher ceiling, error breakdown
06_serving/          model export, GGUF comparison and serving results
07_v2_rules_and_scoring/  v2 format rules and three-criteria judge
08_v2_spanish_and_eval_sets/  Spanish tooling, new dev and test sets
09_v2_dataset_and_labels/  v2 training sentences and teacher labels
10_v2_fine_tuning/   v2 training notebook and dev scoring
demo/                the live demo: API, web page, Docker and deployment files
CONVENTIONS.md       the emoji grammar used everywhere
```

# Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # installs emoji_eval in editable mode + dependencies
cp .env.example .env                 # add the API keys you need
```

Local models run with [Ollama](https://ollama.com). The scorer uses the Anthropic API (evaluation only).

# AI disclosure

- **Test sentences** were drafted with Claude and reviewed by the author; they are used for evaluation only.
- **Claude** (Haiku 5.5, Sonnet 5.5) is used as an **evaluator** (decoder and judge), never to write, select or filter training sentences or labels. Claude also helped write this project's code and documentation.
- **Training labels** are generated by DeepSeek V4.1 Flash (MIT), and part of the training sentences are generated by the same model. Both will be disclosed in the dataset card.
- **Real sentences** come from [Tatoeba](https://tatoeba.org) (CC BY 2.0 FR), with author attribution kept in the data files.
- **Few-shot labelling examples** were reviewed and written by the author.
- **Demo page text and example sentences** were written with Claude. They are not used for training.

# Licence

- **Code:** [MIT](LICENSE).
- **Tatoeba sentences and their labels:** CC BY 2.0 FR, with author credit.
- **AI-generated sentences and their labels:** CC0 1.0.
- **Evaluation material:** published for reproducibility, not licensed for training AI models.

Details and attribution are in [DATA_LICENSE.md](DATA_LICENSE.md).
