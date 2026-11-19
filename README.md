# MiniGPT for Ethan and India

This project trains a small GPT-style model from scratch on 32 English question-and-answer pairs. The personal examples use the name Ethan and the fact that Ethan studies data science. The rest cover India.

## What the model learns

The tokenizer assigns an integer to each character. For an input like `Indi`, the shifted target is `ndia`, so the model learns to predict the next character at every position. Causal attention lets each position use earlier characters. Cross-entropy measures how much probability the model assigns to the actual next characters, and AdamW uses the gradients to update the weights.

A blank line separates records in `input.txt`. The final four records are held out for validation with the starter dataset. Validation loss measures next-character prediction on those records. Test factual accuracy by asking both a question from the training set and a paraphrase. A fluent answer can still contain the wrong fact.

## Run

Open a terminal in the extracted folder. Use an environment with Python and PyTorch 2.x. To install PyTorch in your active Python environment, run:

```bash
python -m pip install torch
```

The archive includes a trained checkpoint. Use the sample questions directly or rerun the training command to reproduce the smaller CPU experiment:

```bash
python train_gpt.py ask "What does Ethan study?"
python train_gpt.py ask "What is the capital of India?"
python train_gpt.py ask "Which city is India's capital?"
python train_gpt.py train --small --batch-size 8 --steps 1500
```

The script picks CUDA, Apple Metal (MPS), or CPU in that order. Training saves `mini_gpt.pt` beside the script. The checkpoint includes the weights, architecture settings, and ordered list of characters. The `ask` command reloads those and encodes the question with the same character mapping. Add `--checkpoint /path/to/model.pt` before the subcommand to use a different checkpoint.

Run `python train_gpt.py train` to use the original width, attention head count, layer count, context length, batch size, and training step count. The larger profile takes longer to train on CPU.

## Add knowledge

Append Q&A records to `input.txt`, keeping one blank line between records. The script reserves the last 10% of records, rounded up, for validation. Keep examples you want the model to train on before that boundary. Prompts can only use characters found in the corpus. A different script layout and attention projection mean the checkpoint from this project is intended for this version of the script.

This is a small training experiment. To answer reliably over a larger set of facts, a retrieval system can look up relevant records and send those records to an existing language model alongside the question.

## Observed CPU run

The included checkpoint used a 358,909-parameter model, a batch size of 8, and 1,500 optimizer steps. Training loss fell from 4.1327 to 0.1045. Validation loss was 2.1050 at step 300 and rose to 3.1104 by step 1,500. This divergence is an example of overfitting. The final checkpoint is included so you can reproduce that behavior.

| Question | Output from the included checkpoint on CPU |
| --- | --- |
| What does Ethan study? | Ethan studies data science. |
| What is the capital of India? | New Delhi is the capital of India. |
| Which city is India's capital? | is the capital of Assam. |

Sampling can vary across devices.
