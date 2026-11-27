# Step 0: Exploration

## What was tried

First manual tests on 20 sentences (10 FR, 10 EN), judged by eye, before any scripted evaluation.

| Model | Usable /20 | Valid | Notes |
|---|---|---|---|
| Big model on Le Chat (Mistral web) | ~12 | 17/20 | Follows conventions; invalid outputs: Chinese characters, "..." |
| Ministral 3B (one sentence per request) | ~8 | 19/20 | Best structure, overuses ⏪, longer, 0.7 s |
| Qwen3.5 2B (one sentence per request) | ~10–11 | 19/20 | Best at picking the right details, 0.4 s |
| Qwen3.5 0.8B (one sentence per request) | ~6–7 | 19/20 | Very variable between runs, 0.2 s |

## Lessons and limitations

- Small models need **one sentence per request**, with few-shot examples sent as previous chat turns. Asking for 20 at once turned the 0.8B into noise.
- **Every model produced at least one invalid output**, so validation is mandatory, in the evaluation and later in the API.
- At temperature 0.7, runs vary a lot: use about 0.2–0.3 and a bigger test set for comparisons.
- Manual guessing showed that **actions come through reliably**; scene, roles and humour are lost first.
- **Structure** (format, conventions) should be easy to teach by fine-tuning; **picking the right concepts** depends on what the base model already knows.
- **Limitations:** 20 sentences judged by eye by one person, so the "usable" counts are rough impressions, not scores. Step 1 replaces them with a scripted round-trip score.

## Decisions made here

- Languages: English and French first, each evaluated separately; Spanish later.
- Teacher (labelling model): not Claude, OpenAI, Google or xAI, whose terms restrict training other models on their outputs. Candidates: Mistral and DeepSeek. Claude may be used as an evaluator.
- Student candidates: Qwen3.5 2B (main), Qwen3.5 0.8B (small enough for the browser), Ministral 3B (comparison).

**What happened next:**
- Spanish was added in v2 ([step 8](../08_v2_spanish_and_eval_sets/)).
- OpenAI's open-weight gpt-oss models (Apache 2.0) were added as teacher candidates in [step 2](../02_teacher_selection/): the restriction is in OpenAI's API terms, not in that licence. DeepSeek V4.1 Flash became the teacher.
- Qwen3.5 0.8B scored clearly below 2B in [step 1](../01_student_selection/), and the in-browser version was not pursued. Qwen3.5 2B became the student, served from a VPS.
