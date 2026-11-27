# Step 8: Spanish and new evaluation sets

Add Spanish, and build new dev and test sets for English, French and Spanish. They have the same mix of sentences as the training data, so v2 models are measured on sentences like the ones they learn from.

**Status:** done. `sets/dev.csv` and `sets/test.csv` hold 300 reviewed sentences each: 100 per language, half real and half AI-generated.

# Main parts of the project

| Path | Purpose |
|---|---|
| `rules_v2.py` | v2 sentence rules: Spanish word lists and blocklist, questions allowed, plain-code tags |
| `fetch_real_v2.py` | Downloads a Tatoeba snapshot and draws the real dev and test sentences |
| `generate_ai_v2.py` | Generates the AI sentences with two open models; `--dry-run es` prints prompts without calling a model |
| `build_sets.py` | Combines both halves into `sets/dev.csv` and `sets/test.csv` and applies the review |
| `sets/review_sheet.csv` | Every sentence in the sets, with the author's verdict (`ok` or `remove`) |

Step 3's scripts are imported, never edited, so step 3 still rebuilds exactly as before.

# Why new sets

- **The v1 test set** was drafted with Claude. It was used to shortlist models in steps 1 and 2, and the author has read it.
- **The v1 dev set** holds only real Tatoeba sentences, while the training data was half AI-generated. This probably explains why test scores ran higher than dev scores.
- **Thin categories:** the v1 dev set was too small for some sentence types (0 questions, 2 numbers in step 5).
- **v2 adds Spanish,** which needs sets of its own.

The v1 sets stay frozen, so steps 0 to 6 can still be reproduced.

# Plan

| Set | English | French | Spanish | Total |
|---|---|---|---|---|
| Dev | 100 | 100 | 100 | 300 |
| Test | 100 | 100 | 100 | 300 |

Each language in each set has 50 real and 50 AI-generated sentences.

1. **Spanish sentence tooling.**
   - Tatoeba Spanish (`spa`), original sentences by self-declared native speakers.
   - The same everyday filters as English and French, with Spanish word lists, impersonal forms and blocklist.
   - The generator gets Spanish instructions, written natively, not translated from English.
2. **New patterns in the generator:** questions and conditionals, plus a fixed share of long sentences (9 words or more). These were weak or unmeasured in v1, and the v2 training data will use the same patterns.
3. **Drawing by code:** a fixed random seed, stratified by language, source, pattern and length, with a cap per Tatoeba author. Nobody picks sentences by hand.
4. **No leakage:**
   - no sentence or near-duplicate from the v1 training data, v1 dev or v1 test, so v1 and v2 models can be compared fairly;
   - only original Tatoeba sentences, never translations, so the same sentence can't appear in two languages;
   - the generator never uses dev or test sentences as examples.
5. **Tags from plain rules:** length group, question, number, negation and speech, used to break results down.
6. **Human review:** the author reads every sentence and removes broken or unnatural ones. Each removed sentence is replaced by the next one in the random order.

**Cost:** no Anthropic API calls. Sentence generation uses Ollama Cloud.

**Results are reported separately for real and AI sentences,** as well as combined. The 50/50 mix was kept on purpose: the step 3 pilot found the mix better for training than either source alone, and real sentences are closer to what demo users type.

## Real half

A fresh Tatoeba snapshot (2026-11-25), with step 3's filters, plus the v2 changes:
- **Spanish** is added.
- **Questions** are allowed.
- **Sentences opening with ¿ or ¡** pass the format check.

| | English | French | Spanish |
|---|---|---|---|
| Original sentences by native speakers | 1,092,593 | 104,963 | 65,821 |
| Everyday filter passed | 425,246 | 43,936 | 12,977 |
| Left out because v1 used them | 800 | 800 | — |
| Kept (dev + test + training pool) | 3,000 | 3,000 | 3,000 |

**Quotas** per language and set (50 real sentences):
- 10 questions;
- 15 long statements (9 words or more);
- 25 other statements.

No author has more than 5 sentences in a set. The remaining 2,900 per language form the real pool for v2 training, minus any sentence used as a replacement during the review.

**Known limitation:** Spanish usually drops the subject pronoun ("Fui al mercado"), so the person filter, which looks for words like "me", "mi", "él" or family words, misses many first-person sentences. Spanish keeps a quarter of its clean sentences, against roughly half for the other two languages, and is biased towards sentences that name a person explicitly.

## AI half

- **Generators:** two open models on Ollama alternate by plan cell, for more variety: DeepSeek V4.1 Flash (MIT) and gpt-oss 120B (Apache 2.0).
- **Patterns:** step 3's six, plus questions and conditionals. Quotas per language and set: 7 questions, 7 conditionals, and 6 of each other pattern.
- **Spanish instructions** are written natively, with neutral wording: no words used only in Spain or only in Latin America (móvil/celular, coche/carro).
- **All three languages** are asked to avoid slang, crude words and regional expressions.
- **Examples shown to the generators** are real Tatoeba sentences, never from any dev or test set. Some contain regionalisms, so the review checks for slips.

## Review

The author reviewed every sentence, in rounds. Each removed sentence was replaced by the next spare of the same language, source and type, then the replacements were reviewed in turn.

| Round | Reviewed | Removed |
|---|---|---|
| 1 | 600 | 56 |
| 2 | 56 | 16 |
| 3 | 16 | 3 |
| 4 | 3 | 2 |
| 5 | 2 | 1 |
| 6 | 1 | 1 |
| 7 | 1 | 0 |

In total, 679 sentences were read and 79 removed (12%). Every sentence in the final sets has an `ok` verdict.

**What was removed most often:**
- real Tatoeba questions written as quoted two-person dialogues;
- crude or sexual real sentences the blocklist missed;
- abstract, proverb-like or literary real sentences;
- incoherent AI sentences, mostly from gpt-oss.

### Final sets

| Tag (plain-code rules, both sets) | English | French | Spanish |
|---|---|---|---|
| Question | 34 | 34 | 34 |
| Conditional | 17 | 15 | 18 |
| Number | 20 | 24 | 17 |
| Long (9+ words) | 130 | 129 | 131 |
| Short (up to 5 words) | 15 | 22 | 13 |

The `speech` tag undercounts in French (6), because its word list misses verbs such as "a écrit" or "a annoncé". The AI sentences carry their generator pattern in `sets/dev.csv` and `sets/test.csv` (24 speech sentences per language), so break down by pattern rather than by tag where possible.

**gpt-oss 120B dropped for the training data.** In round 1, the author removed 19 of its 149 AI sentences (13%), against 3 of 151 for DeepSeek Flash (2%). Its removed sentences were mostly incoherent or ungrammatical. The reviewed gpt-oss sentences stay in the dev and test sets. The v2 training data, which is too large to review line by line, was generated by DeepSeek Flash only ([step 9](../09_v2_dataset_and_labels/)).

**Licences:**
- Tatoeba rows: CC BY 2.0 FR, with attribution.
- AI-generated rows (DeepSeek Flash and gpt-oss 120B): CC0.
- Evaluation sets: please don't train on them, so they stay useful as benchmarks.

See [`DATA_LICENSE.md`](../DATA_LICENSE.md).

## Decision

**The new 300-sentence dev and test sets replace the v1 sets for all v2 work, and the v2 training data uses DeepSeek Flash only.**
- **Sets:** 100 sentences per language, half real and half AI, all reviewed. They match the training mix, cover questions, conditionals and long sentences, and share nothing with v1, so v1 and v2 models can be compared fairly.
- **Generator:** gpt-oss 120B wrote six times more rejected sentences than Flash (13% vs 2%). Its reviewed sentences stay in dev and test, but the training data is too large to review line by line, so it comes from Flash alone.

Next: build and label the v2 training data in [step 9](../09_v2_dataset_and_labels/).

## Lessons and limitations

- **Evaluation sets should look like the training data.** v1's real-only dev set and Claude-drafted test set probably explain why test scores ran higher than dev scores.
- **Human review catches what filters miss.** 12% of drawn sentences were removed: dialogues, crude sentences the blocklist missed, abstract proverbs and incoherent AI text.
- **Generators differ even when their scores tie.** gpt-oss 120B tied with Flash as a teacher in step 2, but wrote far more broken sentences as a generator.
- **Limitations:** the Spanish real sentences are biased towards ones that name a person, because the person filter misses dropped subject pronouns. The `speech` tag undercounts in French. One person reviewed every sentence, and their judgement of "natural" shapes the sets.
