# Topic 8: Fine-Tuning an LLM

This directory is my portfolio submission for [Topic 8: Fine-Tuning an LLM](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/finetuning.html).

## Table of Contents

- [Resources](#resources)
- [Setup](#setup)
- [Step 1: Load The Data](#step-1-load-the-data)
- [Step 3: Evaluate The Base Model](#step-3-evaluate-the-base-model)
- [Step 5: Train](#step-5-train)
- [Step 6: Evaluate After Fine-Tuning](#step-6-evaluate-after-fine-tuning)
- [Step 7: Novel Schema Questions](#step-7-novel-schema-questions)
- [Implementation Files](#implementation-files)
- [Output Files](#output-files)
- [Discussion Questions](#discussion-questions)
- [Notes](#notes)

## Resources

- [Topic 8: Fine-Tuning an LLM](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/finetuning.html)
- [Fine-Tuning Exercise: Text-to-SQL with Tinker](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/sql_finetuning_lesson_plan.html)
- [Introduction to Fine-Tuning LLMs](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/finetuning_intro_lesson_plan.html)
- [Installing Tinker](https://tinker-docs.thinkingmachines.ai/install)

## Setup

Install the Python dependencies:

```bash
pip install tinker transformers python-dotenv
```

Set your Tinker key in `.env` or your shell:

```bash
TINKER_API_KEY=your_key_here
```

## Step 1: Load The Data

Quick dataset check:

```bash
python step1.py
```

## Step 3: Evaluate The Base Model

Small smoke test:

```bash
python finetune_sql.py --step 3 --max-eval-examples 20
```

Full base-model evaluation on the held-out 200 examples:

```bash
python finetune_sql.py --step 3
```

Result:

```text
Base model accuracy: 44.50% (89/200)
```

## Step 5: Train

Run one full training pass only:

```bash
python finetune_sql.py --step 5
```

Example training output:

```text
Epoch 1/1, update 100, loss: 0.0440
Epoch 1/1, update 200, loss: 0.0378
Epoch 1/1, update 300, loss: 0.0406
Epoch 1/1, update 307, loss: 0.0155
```

## Step 6: Evaluate After Fine-Tuning

Evaluate the most recently saved fine-tuned checkpoint:

```bash
python finetune_sql.py --step 6
```

Evaluate a specific fine-tuned checkpoint path:

```bash
python finetune_sql.py --step 6 --checkpoint-path tinker://your/checkpoint/path
```

Result:

```text
Fine-tuned model accuracy: 90.00% (180/200)
```

## Step 7: Novel Schema Questions

Compare the base model and fine-tuned model on the five novel-schema questions:

```bash
python step7_novel_schema.py
```

Run step 7 with a specific fine-tuned checkpoint path:

```bash
python step7_novel_schema.py --checkpoint-path tinker://your/checkpoint/path
```

Summary result:

```text
Base model matches: 2/5
Fine-tuned model matches: 3/5
```

## Implementation Files

- [`step1.py`](./step1.py) loads the dataset, prints a sample example, and performs the train/test split.
- [`finetune_sql.py`](./finetune_sql.py) handles step 3 base evaluation, step 5 training, and step 6 fine-tuned evaluation with checkpoint support.
- [`step7_novel_schema.py`](./step7_novel_schema.py) runs the five novel-schema test cases and compares base versus fine-tuned model outputs.

## Output Files

- [`output/step1.txt`](./output/step1.txt) contains the step 1 dataset inspection and split output.
- [`output/step3.txt`](./output/step3.txt) contains the base-model evaluation run on the 200 held-out examples.
- [`output/step5.txt`](./output/step5.txt) contains the training log and loss updates from fine-tuning.
- [`output/step6.txt`](./output/step6.txt) contains the fine-tuned model evaluation run on the same held-out test set.
- [`output/step7.txt`](./output/step7.txt) contains the novel-schema comparison report for base and fine-tuned models.

## Discussion Questions

### Before vs. after

The most obvious improvement was on the held-out in-distribution evaluation set. The base model reached `44.50% (89/200)`, while the fine-tuned model reached `90.00% (180/200)`, which is an absolute improvement of `+45.50 percentage points`. Based on that jump, I think the model learned both SQL syntax and schema grounding. The fine-tuned model was much better at producing the right SQL shape for the training-style prompts, not just copying fragments.

On the additional novel-schema questions from step 7, the improvement was smaller: the base model matched `2/5`, while the fine-tuned model matched `3/5`. That supports the lesson plan's point that the model improved most on in-distribution text-to-SQL tasks. It generalizes somewhat to new schemas, but the gains are weaker once the prompt moves away from the schemas and patterns seen in the training data.

### RAG comparison

A RAG system would probably work reasonably well for the easier single-table questions, especially when the task looks like a familiar SQL template such as a simple `WHERE` filter or a straightforward aggregate. In those cases, retrieving similar `(question, SQL)` examples could give the model a useful pattern to imitate.

RAG would struggle more on the harder questions that require composition across schema structure, aggregation, ordering, or joins. The model still has to decide which columns to use, how to assemble clauses in the correct order, and how to reason over a schema it has not seen before. That is where fine-tuning seems more useful than retrieval alone, because the challenge is not just finding a similar example but composing a correct new query.

### Error analysis

The step 7 results show that syntax improved more reliably than true out-of-distribution relational reasoning. On the products question, the base model returned row selection instead of `COUNT(*)`, while the fine-tuned model moved closer to the right structure but still added an incorrect `GROUP BY category`. That suggests the fine-tuned model learned more of the expected SQL pattern, but not always the exact logic.

The hardest join question makes the limitation clearer. The base model hallucinated a `department` column directly on `enrollments`, which shows weak schema grounding. The fine-tuned model produced SQL that looked more sophisticated and syntactically plausible, but the join and aggregation logic was still wrong. In other words, fine-tuning improved fluency and structure, but out-of-distribution multi-table reasoning is still a real failure mode.

## Notes

The dataset file should be present at `Topic8FineTuning/sql_create_context_v4.json`.
Step 5 saves the latest sampler checkpoint path to `latest_sampler_checkpoint.json` so step 6 can run later without retraining.
Step 7 saves a text report to `Topic8FineTuning/output/step7.txt` by default.
