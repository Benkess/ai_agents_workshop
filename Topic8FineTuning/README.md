# Topic 8: Fine-Tuning an LLM

This directory is my portfolio submission for [Topic 8: Fine-Tuning an LLM](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic8FineTuning/finetuning.html).

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
Base model accuracy: 45.50% (91/200)
```

## Step 5: Train

Run one full training pass only:

```bash
python finetune_sql.py --step 5
```

```text
--- Step 5: Training Fine-Tuned Model ---
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

## Step 7: Novel Schema Questions

Compare the base model and fine-tuned model on the five novel-schema questions:

```bash
python step7_novel_schema.py
```

Run step 7 with a specific fine-tuned checkpoint path:

```bash
python step7_novel_schema.py --checkpoint-path tinker://your/checkpoint/path
```

## Notes

The dataset file should be present at `Topic8FineTuning/sql_create_context_v4.json`.
Step 5 saves the latest sampler checkpoint path to `latest_sampler_checkpoint.json` so step 6 can run later without retraining.
Step 7 saves a text report to `Topic8FineTuning/output/step7.txt` by default.
