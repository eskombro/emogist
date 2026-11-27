# Emogist

Training a small open-weight model to turn everyday English, French and Spanish sentences into emoji sentences that keep the gist. Live demo: [emogist.esklab.com](https://emogist.esklab.com).

> "I fell asleep on the beach and got a terrible sunburn." → `🏖️😴 ➡️ 🦞🔥 😭`

![The live demo translating "My alarm didn't ring and I missed my flight." into emojis](docs/emogist-demo.gif)

# The goal of this project

This is an educational project: the author's goal was to learn how to fine-tune a small LLM to make it better at a specific task. A small AI model learns from a bigger model's capabilities, following a teacher/student approach (SFT: Supervised Fine-Tuning).

The idea is to help a very small model, one that can be served on a cheap and simple VPS or on a local machine, gain capabilities while adding little overhead, all at the lowest possible cost and using open-source and free tools as much as possible.

The author tested 4 open-weight models to select the best student (Qwen3.5 2B, Qwen3.5 0.8B, Ministral 3B and SmolLM3 3B), and 5 LLMs to select a teacher (DeepSeek V4.1 Flash, DeepSeek V4 Pro, Mistral Large 4, gpt-oss 120B and gpt-oss 20B), then built a dataset and fine-tuned the selected student to close as much of the gap to its teacher as possible.

- **Cost of training** (all experiments included, dataset and fine-tuning): ≈ $15 ([details](#what-it-cost))
- **Time invested:** 4 days

# Results

**How scores work:** one model decodes the emojis back into a sentence without seeing the original, and another judges whether the gist survived. v1 uses a gist score from 0 to 2; v2 uses a stricter score from 0 to 1 over three criteria (topic, event, outcome), so v1 and v2 numbers can't be compared with each other. Brackets are 95% confidence intervals.

**v1: English and French, on the test set (0–2)**

| Model | Score |
|---|---|
| Qwen3.5 2B, untrained, with the full rules and 20 examples in the prompt | 0.81 |
| **Qwen3.5 2B, fine-tuned, with a one-line instruction** | **1.13** |
| Teacher (DeepSeek V4.1 Flash) | 1.31 |

Fine-tuning raised the score by **+0.32 [+0.17, +0.47]** and closed **64% of the gap** to the teacher, while replacing a long prompt with one line.

**v2: English, French and Spanish, on the new dev set (0–1)**

| Model | Score |
|---|---|
| v1 model | 0.47 |
| **v2, Qwen3.5 2B (served in the demo)** | **0.47** |
| v2, Qwen3.5 4B | 0.52 |
| Teacher (DeepSeek V4.1 Flash) | 0.57 |

Cleaner labels, new signs and Spanish didn't move the score at 2B, but a bigger model did: 4B gained **+0.05 [+0.01, +0.08]** on every criterion and every language. The 2B was kept for the demo, because twice the memory on a CPU-only server wasn't worth a small gain. A blind review by Claude Sonnet leans towards v2 over v1 (cleaner outputs, clearer subjects), but not significantly. The v2 test set has not been run yet.

**Also measured:**
- **The scorer was checked against a human rater:** weighted kappa 0.66 for the v1 judge, rank correlation 0.77 for the v2 judge, both above pass marks set before rating.
- **Serving:** the 8-bit model (2 GB) scores the same as full precision and answers in about 3–4 s on a 3-core CPU VPS, loading only when a visitor starts typing.
- **Cost:** about $15 for everything; the final v2 model alone would cost about $2–3.

# Lessons learned

- **Score the gist, not the objects.** An embedding-similarity score rewarded the right objects in the wrong story; a round trip with a judge doesn't.
- **Validate the judge before trusting it.** Pass marks were set before each human check. The first v2 judge failed, and its disagreements turned into two clear rules and a judge that passed.
- **Past a certain size, teachers converge.** Five large models tied, and letting the teacher think added nothing. The prompt and examples mattered more than the choice of teacher.
- **Examples teach more than rules.** Two few-shot examples using 🫵 for the narrator spread to 11% of all labels; fixing the examples did more than any rule text.
- **A model judging its own work flatters itself.** Best-of-n with the teacher as selector looked like a large gain to the teacher and gave none to an independent judge.
- **Better data has limits at a fixed size.** At 2B, v2's cleaner data tied v1; at 4B, the gain was real.
- **Measuring costs more than training.** Training was free on Colab; most of the $15 went into scoring.

# Features

- **Full learning loop:** dataset → fine-tuning → evaluation → serving, documented step by step.
- **Emoji grammar:** a small set of rules for time, speech, negation, numbers and cause, in [CONVENTIONS.md](CONVENTIONS.md).
- **Teacher/student training:** an open teacher model labels sentences, and a small open student learns from them with a LoRA adapter.
- **Round-trip evaluation:** one model decodes the emojis without seeing the original, and another judges whether the gist survived.
- **Three languages:** real human sentences plus AI-generated everyday stories, written natively in English, French and Spanish (Spanish from v2, step 8 on).

# Tech stack

| Area | Tools |
|---|---|
| Models | Qwen3.5 2B (student), DeepSeek V4.1 Flash (teacher) on Ollama Cloud, Claude Haiku and Sonnet (evaluation only) through the Anthropic API |
| Training | Supervised fine-tuning with LoRA: Unsloth, TRL (`SFTTrainer`), PEFT (adapter merge), Google Colab (T4 GPU) |
| Evaluation | LLM-as-judge round trip, human checks (weighted kappa, rank correlation), paired bootstrap confidence intervals |
| Serving | llama.cpp, GGUF (8-bit), LoRA adapters switched per request, FastAPI with streaming |
| Deployment | Docker, GitHub Actions, Caddy, Cloudflare, a shared VPS |
| Web | Plain HTML, CSS and JavaScript, in English, French and Spanish |
| Data | Python, Tatoeba exports, JSONL and CSV datasets |

# How it works

## Steps

| Step | Status | Key result |
|---|---|---|
| [0. Exploration](00_exploration/) | ✅ done | Small models need one sentence per request, and every model wrote some invalid outputs, so validation is mandatory |
| [1. Student selection](01_student_selection/) | ✅ done | Qwen3.5 2B and Ministral 3B tied before training; Qwen3.5 0.8B clearly behind |
| [2. Teacher selection](02_teacher_selection/) | ✅ done | Five large teachers tied (~1.3 vs ~0.9 for the students); DeepSeek V4.1 Flash chosen on tie-breakers; thinking added nothing |
| [3. Dataset](03_dataset/) | ✅ done | 3,000 English and French examples, half real and half AI sentences ([dataset card](03_dataset/full/README.md)) |
| [4. Fine-tuning](04_fine_tuning/) (LoRA, Unsloth) | ✅ done | Qwen 2B chosen over Ministral 3B; on the test set, fine-tuning raised the score from 0.81 to 1.13 |
| [5. Evaluation](05_evaluation/) | ✅ done | Judge validated against a human (kappa 0.66); Qwen closes 64% of the gap to the teacher; long sentences are its weakest point |
| [6. Serving](06_serving/) (GGUF + FastAPI) | ✅ done | The 8-bit model (2 GB) is as good as the full one and loads on demand on a shared VPS |
| [7. v2 rules and scoring](07_v2_rules_and_scoring/) | ✅ done | Short answers allowed; a three-criteria judge passed its human check on the second round (0.77); v1 model baseline 0.47 |
| [8. Spanish and new evaluation sets](08_v2_spanish_and_eval_sets/) | ✅ done | New 300-sentence dev and test sets in English, French and Spanish, every sentence reviewed by the author |
| [9. v2 dataset and labels](09_v2_dataset_and_labels/) | ✅ done | 6,000 labels in three languages, new signs ❓ 💭 🤔 ❔ 📍 🔁; 7% of a reviewed sample judged wrong ([dataset card](09_v2_dataset_and_labels/DATASET.md)) |
| [10. v2 fine-tuning](10_v2_fine_tuning/) | ✅ done | v2 at 2B ties v1 (0.47); 4B scores 0.52 but wasn't kept; dev set only |
| [11. v2 demo](11_demo_v2/) | ✅ done | The [live demo](https://emogist.esklab.com) runs v2, with a before/after compare view, a Spanish page and share links |

## What it cost

Training a small model doesn't need expensive hardware: every training run used a free Google Colab T4 GPU (the v2 model trained in 25 minutes). The money went on API calls.

| Item | Cost |
|---|---|
| Teacher models on Ollama Cloud: candidate tests, generated sentences and labels, v1 and v2 | ≈ $2 |
| Scoring with Claude Haiku (decodes) and Claude Sonnet (judgements), every experiment | ≈ $12 |
| GPU: about 4–5 hours of Colab T4, including the 4B run | $0 |
| **Total, both versions and all experiments** | **≈ $15** |

The final v2 model alone would cost about $2–3: about $1.30 to label 6,000 sentences, one free training run and about $1 to score it. Most of the money went into measuring, not training. The figures are estimates from cached calls and usage pages, and serving (the VPS) is not included.

# Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # installs emoji_eval in editable mode + dependencies
cp .env.example .env                 # add the API keys you need
```

What each part needs:

| To run | You need |
|---|---|
| Scoring (round trip) | `ANTHROPIC_API_KEY` in `.env`. Saved results reuse `cache/`, so rerunning a step's scoring makes no API calls |
| Student candidates (step 1) | [Ollama](https://ollama.com) with the local models |
| Teachers, sentence generation and labels (steps 2, 3, 8, 9) | Ollama signed in to Ollama Cloud (`ollama signin`) |
| Fine-tuning (steps 4 and 10) | Google Colab with a free T4 GPU: open the step's notebook |
| Model export and serving (steps 6 and 11) | [llama.cpp](https://github.com/ggml-org/llama.cpp) and the packages in `06_serving/requirements-export.txt` |
| The demo, locally | See [demo/README.md](demo/README.md#run-locally) |

The fine-tuned models are not published, so the demo and the export scripts need the adapters to be trained first. Each step's README ends with a "Reproduce" section with its exact commands.

# AI disclosure

- **v1 test sentences** (`data/test_sentences.csv`) were drafted with Claude and reviewed by the author; they are used for evaluation only.
- **v2 dev and test sentences** (step 8) are real Tatoeba sentences and sentences generated by DeepSeek V4.1 Flash and gpt-oss 120B, all reviewed by the author. They are used for evaluation only.
- **Blind-review sentences** (step 10): 100 short sentences written by Claude Sonnet 5.5, for evaluation only.
- **Claude** (Haiku 5.5, Sonnet 5.5) is used as an **evaluator** (decoder and judge), never to write, select or filter training sentences or labels. Claude also helped write this project's code and documentation.
- **Training labels** are generated by DeepSeek V4.1 Flash (MIT), and the AI half of the training sentences by the same model. Both are marked as AI-generated in the dataset cards ([v1](03_dataset/full/README.md), [v2](09_v2_dataset_and_labels/DATASET.md)) and in each row's `source`, `generator` and `teacher` fields.
- **Real sentences** come from [Tatoeba](https://tatoeba.org) (CC BY 2.0 FR), with author attribution kept in the data files.
- **Few-shot labelling examples** were reviewed and written by the author.
- **The author's role:** every decision (models, rules, what to keep or drop), the emoji conventions and few-shot examples, the review of every evaluation sentence and of label samples, and the human checks of the judge. Claude helped carry out the plan in code and text.
- **Demo page text and random-button sentences** were written with Claude. They are not used for training.

# Licence

- **Code:** [MIT](LICENSE).
- **Tatoeba sentences and their labels:** CC BY 2.0 FR, with author credit.
- **AI-generated sentences and their labels:** CC0 1.0.
- **Evaluation material:** published for reproducibility, not licensed for training AI models.

Details and attribution are in [DATA_LICENSE.md](DATA_LICENSE.md).
