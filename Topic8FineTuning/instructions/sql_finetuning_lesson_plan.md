# Fine-Tuning Exercise: Text-to-SQL with Tinker
*A Hands-On Exercise for Graduate AI Agents Course*

---

| Field | Details |
|---|---|
| **Duration** | 45–60 minutes (intro ~10 min, exercise ~35–50 min) |
| **Prerequisites** | Completed the Pig Latin fine-tuning exercise; Tinker account with API key |
| **Base Model** | `meta-llama/Llama-3.2-1B` |
| **Dataset** | `b-mc2/sql-create-context` (pre-downloaded JSON file provided) |
| **Platform** | Tinker API (all computation is remote; runs on any laptop) |

---

## 1. Motivation: Why Fine-Tuning Beats RAG Here

In the Pig Latin exercise, we taught a model a simple string transformation rule. Now you will teach it a *compositional skill*: translating natural language questions into SQL queries, given a database schema.

This is a case where **Retrieval-Augmented Generation (RAG) falls short**. You could retrieve similar question-SQL pairs from a database and stuff them into the context window. But the model still needs to:

- **Parse the structure** of the question (what is being asked, what filters apply, what aggregations are needed)
- **Map natural language concepts to column names** it has never seen before
- **Compose valid SQL syntax** — SELECT, WHERE, JOIN, GROUP BY, ORDER BY — in the right order with the right logic
- **Generalize** to novel table schemas and question phrasings that don't match anything in the retrieval corpus

These are *skills*, not *facts*. You cannot look up the answer to "How many students scored above 90 in the math class?" in a reference document — the answer depends on the specific schema, and the model must *generate* the correct SQL by combining syntax knowledge with schema understanding. Fine-tuning bakes this compositional ability into the model's weights.

**Key question to hold in mind:** *After fine-tuning on thousands of (question, schema, SQL) triples, what has the model actually learned? Is it memorizing patterns, or has it internalized SQL grammar?*

---

## 2. The Dataset: `b-mc2/sql-create-context`

This dataset combines the WikiSQL and Spider benchmarks into 78,577 examples. Each example has three fields:

| Field | Description | Example |
|---|---|---|
| `question` | A natural language question | "How many heads of departments are older than 56?" |
| `context` | A SQL `CREATE TABLE` statement | `CREATE TABLE head (age INTEGER, ...)` |
| `answer` | The SQL query that answers the question | `SELECT COUNT(*) FROM head WHERE age > 56` |

The dataset was specifically designed for text-to-SQL fine-tuning. The `CREATE TABLE` context provides table names, column names, and data types — enough for the model to ground its SQL generation without needing actual row data.

The instructor has pre-downloaded the dataset as a JSON file: `sql_create_context_v4.json`. It is a JSON array of objects, each with `question`, `context`, and `answer` keys.  You can find it on the course web page.  This is the code that downloaded it:

```
pip install datasets
python -c "
from datasets import load_dataset
import json
ds = load_dataset('b-mc2/sql-create-context', split='train')
with open('sql_create_context_v4.json', 'w') as f:
    json.dump([dict(row) for row in ds], f)
print(f'Saved {len(ds)} examples')
"
```

---

## 3. Exercise Overview

You will:

1. **Sample before** — ask the base (un-fine-tuned) Llama-3.2-1B to generate SQL from a natural language question. Observe its accuracy.
2. **Prepare training data** — load the dataset, format each example as a prompt/completion pair, and tokenize.
3. **Train** — run a supervised fine-tuning loop on Tinker for one full epoch (~307 batches with batch size 256) using all data except the 200 held-out test examples (~78,377 examples).
4. **Sample after** — ask the fine-tuned model the same questions. Observe the improvement.
5. **Discuss** — reflect on what the model learned and where it still fails.

---

## 4. Step-by-Step Instructions

### Step 0: Setup

Make sure your environment is ready:

```bash
pip install tinker transformers python-dotenv
export TINKER_API_KEY=your_key_here
```

Place the dataset file `sql_create_context_v4.json` in your working directory. (The instructor will tell you where to find it.)

### Step 1: Load and Explore the Data

```python
import json, random

with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nSample example:")
ex = data[0]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")
```

Skim a few examples. Notice the range of SQL complexity — some are simple SELECTs, others involve JOINs, GROUP BY, and subqueries.  Next, split the data into training and test sets. You will use 200 data points for testing and the rest for training.

```python
NUM_TEST_EXAMPLES = 200  # Held-out for evaluation; all remaining data used for training
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples: {len(test_data)}")
```

### Step 2: Define the Prompt Format

We need a consistent template that presents the schema and question as a prompt, with the SQL query as the completion. The model will learn to predict everything after `SQL:`.

```
Table schema:
CREATE TABLE head (age INTEGER, name VARCHAR, ...)
Question: How many heads of departments are older than 56?
SQL: SELECT COUNT(*) FROM head WHERE age > 56
```

This format makes the task unambiguous: the model sees the schema, sees the question, and must produce the SQL.

### Step 3: Evaluate the Base Model

Create the Tinker client using `meta-llama/Llama-3.2-1B`. Evaluate on the testing data.

```python
import tinker

service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)
base_accuracy = evaluate_test_set(
    base_sampling_client, tokenizer, test_data, "base"
)
print(f"Base model accuracy: {base_accuracy:.2%} ({int(base_accuracy * 200)}/200)")
```

**How evaluation works:** For each test example, (1) feed the schema and question to the model to get generated SQL, (2) compare it to the expected SQL. We use **execution-based comparison**: build an in-memory SQLite DB from the schema, run both queries, and check if they return the same result set. Here is the key code:

```python
def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    """Generate SQL from the model given schema and question."""
    prompt = f"""Table schema:
{context}
Question: {question}
SQL: """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"])
    result = sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""

def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    """Evaluate one example: generate SQL, then check if it matches expected."""
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])

def evaluate_test_set(sampling_client, tokenizer, test_data: list) -> float:
    """Compute accuracy = fraction of test examples where generated SQL matches expected."""
    correct = sum(1 for ex in test_data if eval_one(sampling_client, tokenizer, ex))
    return correct / len(test_data)
```

The full set of code for execution-based comparison is in the file `sql_matches.py`, which you should copy or import into your program.  It parses the schema, extracts string/numeric literals from both queries to build realistic seed data, and runs on multiple DB instances to reduce false positives.

### Step 4: Prepare Training Data

Format each one using the template, tokenize, and set loss weights so the model only learns to predict the SQL portion (not the prompt).  This function converts one example into a Tinker Datum:

```python
def format_prompt(example: dict) -> tuple[str, str]:
    """Format example as prompt and completion for text-to-SQL."""
    prompt = f"""Table schema:
{example['context']}
Question: {example['question']}
SQL: """
    completion = example["answer"]
    return prompt, completion

def process_example(example: dict, tokenizer) -> types.Datum:
    """Convert a (question, context, answer) example into a Tinker Datum."""
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Add space before completion, end with \n\n so the model learns to stop
    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Next-token prediction: input is tokens[:-1], target is tokens[1:]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(weights, dtype=np.float32),
        },
    )
```

Now prepare all the training data and shuffle it.

```python
processed_train = [
    process_example(ex, tokenizer) for ex in train_data
]
random.shuffle(processed_train)
```

### Step 5: Train

Run the training loop. For each batch:
1. Take a batch of 256 examples
2. Call `forward_backward` with `cross_entropy` loss
3. Call `optim_step` with Adam
4. Print the loss every 100 steps

You should see the loss decrease over one full epoch (~307 batches). The whole run typically takes 10–20 minutes.

```python
from tinker import types

NUM_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 5e-4  # Tinker-recommended for Llama-3.2-1B with LoRA

step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_idx in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_idx : batch_idx + BATCH_SIZE]
        if len(batch) == 0:
            break

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Compute loss (weighted cross-entropy over completion tokens only)
        to_arr = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())
        logprobs = np.concatenate([to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([to_arr(d.loss_fn_inputs["weights"]) for d in batch])
        loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        if step % 100 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, update {step}, loss: {loss:.4f}")
```

### Step 6: Evaluate the Fine-Tuned Model (After)

Save weights with `save_weights_for_sampler`, create a sampling client from the checkpoint path, and run `evaluate_test_set` on the same 200 test questions. The accuracy is the fraction of questions where the generated SQL returns the same result set as the expected SQL (execution-based comparison). Compare this to the base model accuracy from Step 3.

### Step 7: Test on Additional Novel Schema Questions

Use these questions that involve schemas **not in the training set** (e.g., `employees`, `products`, `students` — the training data uses Spider-style tables like `head`, `department`). These are out-of-distribution; expect lower accuracy here than on the 200 in-distribution test questions. Manually inspect the results.

**Easy (single table, simple WHERE):**
1. Schema: `CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)`
   Question: "What are the names of employees in the engineering department?"

2. Schema: `CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)`
   Question: "How many products cost more than 50 dollars?"

**Medium (aggregation, ORDER BY):**
3. Schema: `CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)`
   Question: "What is the highest score in the science class?"

4. Schema: `CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)`
   Question: "List the top 3 customers by total order amount."

**Hard (JOIN, GROUP BY):**
5. Schema: `CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)`
   Question: "How many students are enrolled in each department?"

---

## 6. Discussion Questions

After completing the exercise:

- **Before vs. after**: What specific improvements did you observe? Did the model learn SQL syntax, schema grounding, or both? What was the change in accuracy on the 200 held-out test questions? How well did it do on the additional manual test questions (Step 7)?
  - Hint: on the 200 in-distribution test questions, accuracy typically improves from ~37% (base) to ~87% (fine-tuned). The Step 7 questions use novel schemas and may show lower accuracy.
- **RAG comparison**: Imagine you had a RAG system with 1,000 (question, SQL) pairs in a vector database. For which of the test questions above would RAG work well? For which would it struggle? Why?
- **Error analysis**: When the fine-tuned model gets a query wrong, *how* does it fail? Wrong column names? Wrong SQL syntax? Wrong logic? Each failure mode tells you something different about what the model learned.

---

## 7. Bonus Challenge

(Optional) Can you improve accuracy on the novel-schema questions (Step 7)? The model excels on in-distribution data (~87%) but struggles on schemas it never saw. Try adding examples with similar schemas to the training set, or experiment with learning rate and batch size.

---

## 8. Environment Reference

### Configuration Notes

- **Batch size** (`BATCH_SIZE`): Number of examples per training step. Default: 256. Increase to speed up training; decrease if you hit memory limits. If you change it, scale learning rate by √(new/old) (e.g., 128 → LR × 0.7).
- **Evaluation workers** (`NUM_EVAL_WORKERS`): Number of parallel workers for evaluation. Each worker sends one question to the model; with N workers, up to N questions run concurrently. More workers reduce evaluation time when you have idle GPU capacity.
- **Evaluation method**: Accuracy is computed via execution-based comparison (SQLite). For each test example, the script builds a DB from the schema, runs both generated and expected SQL, and checks if result sets match. Run the full pipeline with `bash run_finetuning.sh`.

### Dependencies

```bash
pip install tinker transformers python-dotenv
```

### Dataset

File: `sql_create_context_v4.json` (21.8 MB, 78,577 examples)

Source: [https://huggingface.co/datasets/b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)

License: CC-BY-4.0

Download (instructor prep):
```bash
pip install datasets
python -c "
from datasets import load_dataset
import json
ds = load_dataset('b-mc2/sql-create-context', split='train')
with open('sql_create_context_v4.json', 'w') as f:
    json.dump([dict(row) for row in ds], f)
print(f'Saved {len(ds)} examples')
"
```

### Tinker API Key

```bash
export TINKER_API_KEY=your_key_here
```

### Optional: Checkpoint and Download

After training, you can save:

1. **Training state** (`save_state`) — optimizer state and full checkpoint for resuming.
2. **Sampler weights** (`save_weights_for_sampler`) — LoRA adapter weights for inference.

To download the fine-tuned model for local use:

```bash
tinker checkpoint download <checkpoint_path>
```

The checkpoint path is returned when you call save. The downloaded archive contains LoRA weights. Merge with the base model using PEFT/transformers:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base, "./path/to/downloaded/adapter")
```

See [Tinker download docs](https://tinker-docs.thinkingmachines.ai/download-weights) for details.

### Prompt Template

```
Table schema:
{context}

Question: {question}

SQL: {answer}
```

---

*End of Exercise — Text-to-SQL Fine-Tuning with Tinker*
