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
python finetune_sql.py --max-eval-examples 20
```

Full base-model evaluation on the held-out 200 examples:

```bash
python finetune_sql.py
```

## Step 5: Train

Run one full training pass after the base evaluation:

```bash
python finetune_sql.py --train
```

## Step 6: Evaluate After Fine-Tuning

Train and then evaluate the fine-tuned model on the same held-out test set:

```bash
python finetune_sql.py --train --evaluate-after
```

## Notes

The dataset file should be present at `Topic8FineTuning/sql_create_context_v4.json`.
