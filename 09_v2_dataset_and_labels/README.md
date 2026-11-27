# Step 9: v2 dataset and labels

Build the v2 training data: 2,000 examples per language in English, French and Spanish. Each sentence gets one label from DeepSeek V4.1 Flash, with a new prompt, new signs and the author's examples. Picking the best of several labels was tested and dropped: it gave no gain.

**Status:** ✅ done: 6,000 training rows (v7 labels), reviewed on a sample. Described in the [dataset card](DATASET.md).

# Plan

## Decisions already taken

| Decision | Choice | Who / where |
|---|---|---|
| Sentences | Fresh for all three languages, with the step 8 rules (questions and conditionals included) | Author, this step |
| Size | 2,000 per language (1,000 real + 1,000 AI), 6,000 in total | Author, this step |
| Teacher and generator | DeepSeek V4.1 Flash only; gpt-oss dropped (13% of its sentences removed vs 2% for Flash) | Author, step 8 |
| Label selection | Planned: the best of several Flash candidates, chosen by Flash through a round trip. **Dropped after the pilot** (no gain): one label per sentence | Author, this step |
| Human review | Random samples: 150 examples (50 per language), then 60 more after the relabel | Author, this step |
| Training runs (step 10) | Main model on EN+FR+ES, and an EN+FR control on the same EN/FR rows | Author, v2 plan |

**Terms:** DeepSeek allows training other models on its outputs, including distillation, as long as published AI-generated content is labelled as such. Claude is not used to write, select or filter any training sentence or label. Claude is used only to measure results on the dev set.

## Sentences

**Real (1,000 per language):** `draw_real_train.py` → `sentences/real_train.csv`
- Drawn in the fixed shuffled order of the step 8 pool (`08_v2_spanish_and_eval_sets/sets/real_pool.csv`, 2,900 per language). The pool already excludes v1 train, dev and test.
- Left out: pool sentences used in the v2 dev or test sets (replacements), near-duplicates of any dev or test sentence (v1 and v2, real and AI), and the worksheet sentences.
- Quotas: 20% questions, 30% long, the rest other. At most 10% of a language's sentences from one Tatoeba author, except for questions: they come from few authors, so all available ones are taken (author's decision). French and Spanish have fewer questions than the quota (189 and 127), and other sentences fill the gap.
- 3% margin (1,030 drawn) for sentences that end up with no label.

**AI (1,000 per language):** `generate_ai_train.py` → `sentences/ai_train.csv`
- Flash only, with the step 8 generator (Spanish prompt as corrected by the author).
- 8 patterns: 140 questions, 140 conditionals and 120 each of speech, past, future, mishap, moment and numbers, plus a 3% margin.
- Plan cells already used in step 8 are skipped. Every step 8 sentence, dev and test, and the worksheet count as known sentences, so near-duplicates are rejected. Example sentences shown to the generator never come from dev or test.
- Requests run in parallel, but sentences are accepted in plan order, so the result doesn't depend on timing.

**Labels:** `label_train.py`. One label per sentence with the v2 prompt: two tries, then up to two repairs. The first 1,000 labelled real sentences per language (in draw order) and the first labelled AI sentences up to each pattern's quota (in generation order) are kept.

**Test batch (3% of the AI quotas, 20 sentences per language and source):** all 120 sentences got a label (109 on the first try, 5 after a repair), 5.4 emojis on average, ➡️ in 41%, no style warnings, none sharing a 3+ emoji group with an example.

**Cost:** about 10,500 Flash requests for the full run. The labelling prompt is about 1,800 tokens, 91% served from Ollama's cache. From the author's usage page, about $0.065 per 1,000 requests, so about $0.70 (author's OK).

**Full run result:**
- 3,102 AI sentences generated (3,083 requests; a handful rejected by the step 8 rules), 3,090 real drawn.
- **First labelling pass (prompt v5):** 121 of 6,192 sentences got no label, and 88 of them were questions, mostly real ones. Two causes: the question wasn't the last sentence ("Can I have a hug?" "I have a boyfriend."), so "❓ at the very end" couldn't fit; and wh-questions where Flash kept ❔ even after two repairs.
- **Author's decision:** ❓ goes right after the question part, as its own group (at the end when the question comes last). The rule text, the check and the repair hints were updated (prompt v6). The v5 labels were reused from the cache and re-checked; only the 90 sentences still without a label were labelled again with v6 (257 requests). 39 sentences have no label in the end.
- The 3% margin of real sentences is cut from the last non-questions, so no labelled question is lost.
- **Kept:** 6,000 rows, 1,000 per language and source; 923 questions (real: EN 201, FR 182, ES 120; AI: 140 per language). 5,953 labels from v5, 47 from v6.
- 5.8 emojis on average, ➡️ in 44%, no style warnings, 8 labels sharing a 3+ emoji group with an example.
- About 10,500 Flash requests in total, 13.9 M input tokens.

**Author's review, round 1** (`results/review_round1.csv`, 150 rows of the v6 labels in `results/train_rows_v6.csv`): 19 removed (13%): EN 7, FR 9, ES 3; real 12, AI 7. Recurring note: 🫵 used for "I" or "on". Plain counts on all 6,000 rows then showed:
- 🫵 with no "you" word in the sentence: 674 rows (11%), mostly real misuses ("He admitted that I was right." → `👨💬 🫵✅`). Two of the author's speech examples used 🫵 for the narrator inside speech, which Flash generalised.
- "?" without ❓ or the reverse: 94; 🔀/🔄 still used: 53; copies of an example label for an unrelated sentence: 7.

**Prompt v7 and relabel of all 6,000 rows (author's decision):**
- The author rewrote the two speech examples with 🙋 for the narrator; the prompt says "🫵 is only 'you', the person being spoken to; 'I / me' is 🙋 and 'we / us' 👥, also inside speech".
- New retry rules in code (`labels.strict_problem`): no copy of an example label, ❓ exactly when the sentence has a "?", no 🔀/🔄.
- Format fix in code (author's decision): a ❓ glued to a neighbouring emoji gets its own group (`👥❓` → `👥 ❓`). Only a space changes; no emoji is added, removed or chosen. Without it, 95 of the 137 unlabelled sentences were questions.
- About 7,550 requests (13.9 M input tokens).

| Plain count | v6 | v7 |
|---|---|---|
| 🫵 without a "you" word | 674 | 293 (partly imperatives and Spanish without pronoun) |
| "?" / ❓ mismatch | 94 | 0 |
| 🔀 or 🔄 | 53 | 0 |
| Copy of an example | 7 | 0 |
| No label | 39 | 74 |

**Final data (v7):** 6,000 rows (1,000 per language and source), 929 questions (real: EN 198, FR 186, ES 125; AI: 140 per language), 5.9 emojis on average, ➡️ in 43%, no style warnings. `train_main.jsonl` 6,000, `train_control.jsonl` 4,000. Review round 2: `review_sheet.csv` (copy in `results/review_round2.csv`), 60 rows (20 per language), none from round 1.

**Author's review, round 2 (v7 labels):** 4 of 60 removed (7%, roughly 3% to 16% given the small sample), against 13% in round 1. EN 1, FR 1, ES 2; real 3, AI 1. No repeating problem: one each of 🫵 used for "me", an unneeded ❔, an unneeded 🔁, and a sentence too complex to draw. **Author's decision:** the review is a measurement only; all 6,000 rows stay in the training files.

## Labelling prompt v2

`label_prompt_v2.py`. Starts from the v1 prompt (`v4-human-examples`). Changes:
- **Format rule:** as few emojis as the idea needs, usually 3 to 8, at most 10; 1 or 2 only for simple sentences.
- **A short section on the signs with a fixed meaning** (❓, 💭, 🤔, ❔, 📍, 🔁), worded from CONVENTIONS.md.
- **Few-shot examples written by the author:** the worksheet `label_examples_v2.csv`, 43 examples (EN 16, FR 14, ES 13). Every example is checked against the v2 rules when the prompt is built.

**Retries (`labels.py`):** two tries at temperature 0.3, then up to two *repairs*: Flash sees its broken label and a one-line description of the rule it breaks, and writes a new one. The code never suggests emojis. Retries at a higher temperature were tried first and didn't help: Flash repeated the same mistake.

**Style checks (`style_check_v2.py`, Ollama only):** Flash labels the 300 dev sentences with each prompt; plain counts only. Three rounds, with the author's conventions and examples adjusted between them:

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| No valid label | 12 | 14 | **2** |
| Questions ending with ` ❓` | 82% | 84% | **92%** |
| Conditions starting with 🔀 | 89% | 89% | 93% |
| Labels identical to an example | 0 | 0 | 0 |

- Run 1: ❔ was used for "when / why" and 🔄 kept its usual meaning "change". Wh-questions were set to ❓ only, and 🔄 to every / all.
- Run 2: Flash mixed "every" (🔄 before) and "again" (🔄 after). "Again" became 🔁, and the repairs replaced the high-temperature retries.

**Quality check (`quality_check_v2.py`, author's OK, about $2.50):** labels of both prompts on the 300 dev sentences, decoded by Haiku and scored by the Claude v2.1 judge.

| | v1 prompt | v2 prompt (run 3) |
|---|---|---|
| All 300 | 0.59 | 0.53 |
| v2 − v1 | | −0.07 [−0.10, −0.03] |

- **The loss came from two signs:** labels with 🔀 scored 0.31 against 0.70 for the v1 labels of the same sentences, and labels with 🔄 scored 0.34 against 0.55. The decoder, which only knows the rules a demo visitor would know, read them as "shuffle" and "switch" ("If no one shows up…" → "The seating plan changed…").
- **Labels with none of the new signs scored about the same:** 0.58 against 0.60.
- ❓, 💭, 🤔 and ❔ cost little or nothing; 🔁 and 📍 a little (20 and 10 labels).
- **Author's decision:** drop 🔀 and 🔄. Visitors can't be expected to learn signs before trying the demo. The v2.1 rubric doesn't count mood (if / real / question), so a sign for "if" can't earn points, only lose them.

**After dropping them (style check 4, then re-scored with the author's OK, about $1.10):**

| | v1 prompt | v2 prompt (run 4) | v2 − v1 |
|---|---|---|---|
| All 300 | 0.59 | 0.57 | −0.03 [−0.06, +0.00] |
| EN / FR / ES | 0.59 / 0.56 / 0.63 | 0.54 / 0.55 / 0.61 | −0.05 / −0.01 / −0.02 |
| Real / AI | 0.56 / 0.63 | 0.56 / 0.57 | −0.00 / −0.05 |
| Conditions (27) | 0.67 | 0.67 | −0.00 |
| Questions (51) | 0.58 | 0.56 | −0.02 |

- Style: 94% of questions end with ❓, all 27 conditions use ➡️, 4 of 300 sentences get no label (as many as with the v1 prompt).
- **None of the kept signs costs meaning:** labels with ❓ or 🔁 lose 0.02, those with 📍, 💭, ❔ or 🤔 score the same or higher.
- **What remains of the gap comes from shorter labels:** where the v2 label is shorter than the v1 one (116 sentences), it scores 0.56 against 0.61; at equal length, 0.60 against 0.61. The new rule 1 ("as few emojis as the idea needs") sometimes drops a detail the decoder needed.

## Best-of-n labels (tested, not used)

The original plan, dropped after the pilot below. For each sentence:
1. Flash writes one label at temperature 0.3, as in v1.
2. Flash decodes it without seeing the original, then scores the guess with the v2.1 rubric.
3. **If the score is below the maximum,** Flash writes 3 more candidates at a higher temperature, and the best-scoring valid candidate wins. Ties go to the earliest candidate.

This adaptive version only spends extra requests where the first label loses something.

## Checks before the full run

1. **Flash as selector.** Flash scores the 102 pairs the author rated in step 7, with the v2.1 rubric. This shows how far its scores follow the author's. Ollama only, no Claude cost.
   **Result: Flash is usable as the selector.** Script: `check_flash_judge.py`, output `results/flash_judge_check.csv`.
   - **Round 1** (60 pairs, rated before the two v2.1 rules existed): Flash 0.65 [0.45, 0.79], against 0.67 for the Claude v2 judge on the same pairs.
   - **Round 2** (42 pairs, the like-for-like test): Flash 0.67 [0.40, 0.85], against 0.77 for the Claude v2.1 judge. The difference, −0.10 [−0.33, +0.08], is not proven, but Flash is probably a little weaker.
   - **Average score:** Flash is about as strict as Claude (0.39 to 0.42, against the author's 0.57 to 0.60).
   - **Format:** Ollama Cloud ignores the JSON schema for Flash, so `flash_roundtrip.py` asks for JSON in words and checks the answer.
2. **Pilot on dev.**
   - 60 dev sentences (10 per language and source).
   - A: single label. B: best-of-n.
   - Both are decoded by Haiku and scored by the Claude v2.1 judge.
   - Cost about $0.75; the author's OK is needed first.
   - **If B doesn't clearly beat A, the full run uses single labels.**

   **Result: best-of-n gives no gain, so the full run uses single labels.** Script: `pilot_best_of_n.py`, output `results/pilot_best_of_n.csv`.

   | | A: single label | B: best-of-n |
   |---|---|---|
   | Claude v2.1 score | 0.57 | 0.58 |

   - **B − A:** +0.01 [−0.05, +0.06].
   - **Extra candidates:** drawn for 45 sentences, and the label changed for 31.
   - **Flash rated its own choices much higher than Claude did.** On the 31 changed labels, Flash's score rose from 0.37 to 0.73, but Claude's only from 0.57 to 0.58: 12 better, 9 worse. Flash picks the labels it decodes well itself, and that doesn't carry over to another decoder. With one decode and one score per candidate, part of the choice is also luck.
   - **Failures:** 2 of 60 sentences got no valid first label after the retry, so the full run needs more retries.
   - **The teacher's single labels already beat the v1 student:** 0.57 here, against 0.47 for the v1 model on the whole dev set (a different sample).

## Review and output

- **Sample review:** the author reads a random sample (sentence and label) and marks each ok or remove: 150 rows (50 per language) in round 1, 60 (20 per language) in round 2. The removal rate is reported. If a problem repeats, the fix goes into the rules, not into hand-picking.
- **Output:**
  - `train_main.jsonl`: EN+FR+ES, 6,000 examples;
  - `train_control.jsonl`: the same 4,000 EN and FR rows.

  Each row keeps its source, pattern, tags, Tatoeba author, generator, teacher, prompt version and the attempt that produced the label, so AI content is labelled as such. See the [dataset card](DATASET.md).

## Requests and cost

Actual figures (approximate). Ollama Cloud cost about $0.065 per 1,000 Flash requests.

| Part | Ollama Cloud requests | Anthropic cost |
|---|---|---|
| Flash selector check | ~200 | — |
| Pilot on dev (best-of-n) | ~1,000 | ~$0.75 |
| Style checks 1–4 | a few thousand | — |
| Quality check, then re-score after dropping 🔀/🔄 | — | ~$2.50 + ~$1.10 |
| Full run: AI sentences and v5 labels | ~10,500 | — |
| v6 relabel of failures | 257 | — |
| v7 relabel of all rows | ~7,550 | — |

## Worksheet and conventions

- **Few-shot worksheet:** `label_examples_v2.csv`, made by `pick_example_candidates_v2.py`.
  - It started from the 20 v1 examples, plus 36 empty candidates drawn by rule from the real pool: 18 Spanish (everyday, vague, speech, numbers) and 3 questions and 3 conditionals per language.
  - The author wrote the emojis and added hand-written rows: 43 examples in the end.
  - Worksheet sentences are kept out of the training data.
- **Conventions for the new types (author's decision):**
  - ❓ as its own group, right after the question part, marks a question, including "why / what" questions (first "at the very end", changed with prompt v6).
  - 💭 means think and 🤔 means maybe.
  - ❔ attached marks something unknown or asked about ("asked if…", "don't know where…"), only in sentences that are not questions. 📍 attached marks a place or "where".
  - 🔁 attached to an action means again (`🍔🔁`).
  - Conditions have no sign: the "if" part, then ➡️, as in v1.
  - **Tried and dropped:** 🔀 for "if" and 🔄 for "every / all" (see the quality check above).

  Recorded in CONVENTIONS.md. `check_conventions_v2` checks the positions of ❓, ❔, 📍 and 🔁, and a label that breaks one is retried. Whether a sentence needed a sign (a "?" with no ❓) is only counted in the style report on dev, because keyword matching makes mistakes.
- **Not enough real questions for French and Spanish.** The pool has 294 English, 191 French and 135 Spanish questions, before taking out the dev/test replacements, and the quota needs 200 per language. **Author's decision:** take every available question and fill the rest of the real half with other sentences. The AI half adds 140 questions per language.

## Training start (step 10)

Both v2 runs, main and control, train a **new LoRA adapter on the base Qwen3.5 2B**, not on top of the v1 adapter:
- **Clean comparison:** v1 and v2 start from the same model, so the gain comes from the data, and the main and control runs differ only in the Spanish rows.
- **No conflicting rules:** v1 labels follow the old format (at least 3 emojis, no question or condition rules). Continuing from v1 would mix the two.

## Decision

**Train v2 on the 6,000 v7 labels, one Flash label per sentence, with all rows kept.**
- **One label per sentence:** best-of-n gave no gain (+0.01) for extra requests, so it was dropped.
- **Prompt v7:** the new signs ❓ 💭 🤔 ❔ 📍 🔁 stay because they cost no meaning; 🔀 and 🔄 were dropped because decoders misread them. The v7 relabel fixed the 🫵 misuse and the question-mark mismatches.
- **All rows kept:** the author's sample review judged 7% of v7 labels wrong, down from 13%, with no repeating problem. The review measures quality; it doesn't filter rows.
- **Two training files:** the main file (EN+FR+ES) and an EN+FR control on the same rows, so step 10 can show what Spanish adds.

Next: train both adapters from the base Qwen 2B in [step 10](../10_v2_fine_tuning/).

## Lessons and limitations

- **A model judging its own work flatters itself.** Flash rated its best-of-n picks much higher (0.37 → 0.73) than Claude did (0.57 → 0.58): it chose labels it decodes well, not labels others decode well.
- **New signs need to be learnable.** 🔀 and 🔄 looked logical, but a decoder that only knew what a demo visitor knows read them as "shuffle" and "switch". A demo visitor would do the same.
- **Examples teach more than rules.** Two speech examples using 🫵 for the narrator spread to 11% of all labels. Fixing the examples did more than any rule text.
- **Fix problems in the rules, not by hand.** Each repeated problem in the review became a rule, a check or a better example, then everything was relabelled.
- **Limitations:** the review samples are small (60 rows in round 2, so 7% could be anywhere from about 3% to 16%), and about 7% of the training labels are probably wrong. Shorter v2 labels sometimes drop a detail (0.56 vs 0.61 where shorter). French and Spanish have fewer real questions than planned.
