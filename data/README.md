# Data

## `test_sentences.csv`: fixed test set

- 100 everyday sentences: 50 French (`fr-01`…`fr-50`) and 50 English (`en-01`…`en-50`). Columns: `id`, `lang`, `sentence`.
- Used **only for evaluation**, never for training. Few-shot examples and training data deliberately avoid its topics.
- Covers different lengths, negation, past/future, numbers, speech, humour and roles.
- The first 10 sentences per language are the original manual test set; the rest were added to get more reliable comparisons.

**AI disclosure:** most sentences were drafted with Claude (Anthropic) and reviewed by the author. They are used as evaluation inputs only, not as training data.

## Training dataset

To come in step 3: about 3,000 sentences written natively in English and French, labelled by the chosen teacher model. Labels will be AI-generated, which will be stated in the dataset card.
