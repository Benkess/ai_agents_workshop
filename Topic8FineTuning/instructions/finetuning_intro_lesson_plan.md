# Introduction to Fine-Tuning LLMs
*A Lesson Module for Graduate AI Agents Course*

---

| Field | Details |
|---|---|
| **Duration** | 60–75 minutes (lecture ~45 min, guided walkthrough ~20 min) |
| **Prerequisites** | Familiarity with how LLMs work (tokens, next-token prediction, transformer architecture at a high level) |
| **Format** | Lecture + read-through walkthrough. Students do not write code in this lesson — they will code in the next exercise. |
| **Platform** | Tinker API (thinkingmachines.ai) |

---

## 1. What Is Fine-Tuning and Why Does It Matter?

Pre-trained LLMs are generalists. They have seen enormous amounts of text during pre-training and can do many things passably, but they are not specialists at anything in particular. Fine-tuning takes a pre-trained model and continues training it on a much smaller, task-specific dataset so the model internalizes a particular skill or behavior.

The key intuition: pre-training teaches the model *language* — grammar, facts, reasoning patterns, code syntax. Fine-tuning teaches the model a *job* — how to respond in a particular format, follow instructions, translate between representations, or behave according to specific rules.

**When does fine-tuning make sense over alternatives?**

Consider three approaches to making an LLM do something new:

- **Prompting / few-shot**: You write instructions and examples in the prompt. Works well for tasks the model already *almost* knows how to do. Costs tokens at every inference call. Limited by the context window.
- **RAG (Retrieval-Augmented Generation)**: You retrieve relevant documents and paste them into the context. Works when the task is about *facts* — the model needs information it doesn't have. Does not help when the model needs to learn a *skill* or *format*.
- **Fine-tuning**: You modify the model's weights so the behavior is baked in. Works when you need the model to reliably perform a transformation, follow a format, or exhibit a skill it currently cannot do. The cost is paid once at training time, not at every inference.

**The rule of thumb:** If the task requires *knowledge*, use RAG. If the task requires a *skill*, fine-tune. Many real applications use both.

---

## 2. Fine-Tuning Approaches: Full, LoRA, and QLoRA

### 2.1 Full-Parameter Fine-Tuning

The simplest conceptual approach: take all the model's parameters and continue training them with gradient descent on your new dataset, using the same cross-entropy (next-token prediction) loss used during pre-training.

For a 7B parameter model, this means updating all 7 billion parameters on every training step. This requires:

- Storing the model weights in GPU memory (in fp16, 7B parameters ≈ 14 GB)
- Storing the optimizer state (Adam keeps two additional copies of every parameter: momentum and variance, so another ~28 GB)
- Storing the gradients (~14 GB)
- Plus activations during the forward pass

Total: a 7B model easily requires 60+ GB of GPU memory for full fine-tuning. A 70B model requires a cluster of GPUs.

Full fine-tuning gives you the maximum flexibility — every parameter can change — but it is expensive and can be overkill for many tasks. It also risks *catastrophic forgetting*: if your fine-tuning dataset is small or narrow, the model may lose general capabilities it had before.

### 2.2 LoRA (Low-Rank Adaptation)

**The core idea:** Instead of updating all the model's weight matrices, freeze the original weights and attach small trainable "adapter" matrices alongside them.

For each weight matrix W (size n × n) in the transformer, LoRA adds two small matrices: B (size n × r) and A (size r × n), where r is the *rank* — typically 16 or 32. The effective weight becomes:

```
W' = W + B × A
```

Only B and A are trained. The original W is frozen.

**Why this works:** The key insight from the LoRA paper (Hu et al., 2021) is that the *change* needed during fine-tuning is typically low-rank — you don't need to modify all dimensions of every weight matrix. A rank-32 adapter on a model with hidden dimension 4096 adds only 2 × 4096 × 32 = 262,144 parameters per layer, compared to 4096 × 4096 = 16.8 million in the original matrix. That is about 1.5% of the parameters.

**Practical benefits:**

- Memory: You only store gradients and optimizer state for the small adapter matrices. A 7B model that needs 60+ GB for full fine-tuning might need only 16–20 GB with LoRA.
- Speed: Fewer parameters to update means faster training steps.
- Portability: The adapter weights are small files (tens of MB). You can swap adapters without reloading the base model.
- Reduced catastrophic forgetting: The base weights are frozen, so the model retains its general capabilities.

**What rank to use?** Tinker defaults to rank 32. Higher ranks give the model more capacity to learn complex changes but cost more memory and compute. For reinforcement learning and small supervised learning datasets, even low ranks (8–16) work well. For large SL datasets, you may need rank 64 or 128. A useful rule of thumb from the Tinker documentation: LoRA gives good results as long as the number of LoRA parameters is at least as large as the number of completion tokens in your training data.

**Does LoRA match full fine-tuning?** The Thinking Machines team (who build Tinker) published research showing that LoRA matches full fine-tuning for RL and small-to-medium SL datasets. For very large SL datasets, full fine-tuning still has an edge — but for most practical tasks, LoRA is sufficient.

### 2.3 QLoRA (Quantized LoRA)

QLoRA (Dettmers et al., 2023) combines LoRA with *quantization* of the frozen base weights. Instead of keeping the base model in 16-bit precision, QLoRA stores it in 4-bit (NF4 format), dramatically reducing memory usage.

The math: a 7B model in 16-bit needs ~14 GB for weights alone. In 4-bit, it needs ~3.5 GB. Add the LoRA adapters (still in 16-bit) and the optimizer state for those adapters, and you can fine-tune a 7B model on a single consumer GPU with 8 GB of VRAM.

**The tradeoff:** Quantization introduces some approximation error. The base model is slightly less precise, and there can be subtle quality differences compared to full-precision LoRA. In practice, QLoRA achieves results very close to standard LoRA for most tasks, and the memory savings are enormous.

**When to use which:**

| Approach | Memory | Quality | When to Use |
|---|---|---|---|
| Full fine-tuning | Very high (60+ GB for 7B) | Best (all params trainable) | Large datasets, production models, research |
| LoRA | Moderate (16–20 GB for 7B) | Near-full-FT for most tasks | Most practical fine-tuning |
| QLoRA | Low (8–12 GB for 7B) | Slightly below LoRA | Limited GPU budget, prototyping, education |

### 2.4 Where Tinker Fits

In a traditional fine-tuning setup, you would run all of this on your own GPUs — managing CUDA, distributed training, memory management, and fault recovery. This is where most of the engineering pain lives.

Tinker takes a different approach: you write a simple Python loop on your laptop (CPU only) that specifies the data, the loss function, and the optimizer settings. Tinker's servers handle the actual GPU computation — LoRA training distributed across their cluster. You never touch a GPU.

The API has four core primitives:

- `forward_backward(data, loss_fn)` — computes the forward pass and gradients on Tinker's GPUs
- `optim_step(params)` — updates the LoRA adapter weights using the accumulated gradients
- `sample(prompt, params)` — generates text from the current model (base + adapter)
- `save_state()` / `load_state()` — checkpoint and resume

This is not a black-box "upload your data and we train for you" service. You control the training loop, the batch size, the learning rate, what data goes in each batch, and when to evaluate. Tinker just handles the distributed GPU infrastructure.

---

## 3. Supervised Fine-Tuning: The Mechanics

Regardless of whether you use full fine-tuning, LoRA, or QLoRA, the supervised fine-tuning (SFT) procedure follows the same pattern:

### 3.1 The Training Example Format

Each training example is a sequence of tokens divided into two parts:

- **Prompt tokens** (weight = 0): The input that provides context. The model sees these but is not penalized for "predicting" them incorrectly.
- **Completion tokens** (weight = 1): The output the model should learn to produce. The cross-entropy loss is computed only on these tokens.

For example, if you are training a model to translate English to Pig Latin:

```
English: banana split        ← prompt (weight = 0)
Pig Latin: anana-bay plit-say ← completion (weight = 1)
```

The model is trained to predict the completion tokens given the prompt tokens. The weight mask ensures the loss only counts on the part we care about.

### 3.2 Next-Token Prediction and the Shift

LLMs predict the *next* token given all previous tokens. So the training data must be "shifted by one": the input at position i is the token at position i, and the target at position i is the token at position i+1.

```
Input tokens:  [English] [:]  [ banana] [ split] [\n] [Pig] [ Latin] [:]  [ an]   [ana]  ...
Target tokens: [:]       [ banana] [ split]  [\n]    [Pig] [ Latin] [:] [ an]   [ana]  [-bay] ...
Weights:        0         0        0         0       0      0       0    1       1      1     ...
```

Notice how the weight switches from 0 to 1 at exactly the point where the completion begins. Everything before that is context; everything after is what the model should learn.

### 3.3 The Training Loop

A single training step:

1. Take a batch of training examples (e.g., 16 examples)
2. Run the forward pass: compute the model's predicted probability of each target token
3. Compute the loss: negative log-likelihood of the correct target tokens, weighted by the weight mask
4. Run the backward pass: compute gradients of the loss with respect to the trainable parameters
5. Update the parameters using an optimizer (typically Adam)

Repeat for many batches. Track the loss — it should decrease over time. Periodically evaluate on held-out test data to check for overfitting.

### 3.4 Key Hyperparameters

- **Learning rate**: The most important hyperparameter. Too high and the model diverges; too low and it learns nothing. LoRA requires a much higher learning rate than full fine-tuning — typically 20–100× higher, because the adapter matrices are small and need larger updates to have an effect.
- **Batch size**: Number of examples per gradient update. Larger batches give more stable gradients but cost more memory.
- **Number of steps / epochs**: How long to train. More is not always better — at some point the model starts memorizing the training data rather than generalizing.
- **LoRA rank**: Controls the capacity of the adapter. Higher rank = more parameters = more capacity, but also more memory and compute.

---

## 4. Walkthrough: Teaching Pig Latin with Tinker

Now we will walk through a complete fine-tuning example step by step. **You will not write code in this section** — just read through and understand each step. In the next exercise, you will write a similar script yourself for a more challenging task.

### 4.1 The Task

Teach Llama-3.2-1B to translate English phrases into Pig Latin. The rules are:

- If a word begins with a consonant, move the consonant to the end and add "ay" (e.g., "banana" → "anana-bay")
- If a word begins with a vowel, add "way" to the end (e.g., "exploration" → "exploration-way")

This is a good first fine-tuning task because: the rule is simple and deterministic, we can generate our own training data, and the base model is terrible at it (so the before/after contrast is dramatic).

### 4.2 Setup

Two dependencies:

```bash
pip install tinker transformers
export TINKER_API_KEY=your_key_here
```

The `tinker` package handles the API communication. The `transformers` package provides the tokenizer.

### 4.3 Create a Training Client

```python
import tinker
from tinker import types

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B",
    rank=32,
)
tokenizer = training_client.get_tokenizer()
```

This creates a LoRA adapter (rank 32) attached to the frozen Llama-3.2-1B weights on Tinker's servers. The `training_client` object is your handle to that remote model. The tokenizer runs locally on your machine.

### 4.4 Define the Training Data

Just 7 handwritten examples:

```python
examples = [
    {"input": "banana split",      "output": "anana-bay plit-say"},
    {"input": "quantum physics",   "output": "uantum-qay ysics-phay"},
    {"input": "donut shop",        "output": "onut-day op-shay"},
    {"input": "pickle jar",        "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck",       "output": "ubber-ray uck-day"},
    {"input": "coding wizard",     "output": "oding-cay izard-way"},
]
```

In a real fine-tuning job you would use thousands or millions of examples. Here we use 7 to keep the demo simple and fast.

### 4.5 Tokenize with Weight Masks

Each example must be converted into a `Datum` — Tinker's format for a training example. The critical step is setting the weight mask so the model only learns to predict the Pig Latin output, not the English prompt.

```python
def process_example(example, tokenizer):
    prompt = f"English: {example['input']}\nPig Latin:"

    # Tokenize prompt — the model sees this but is NOT trained on it
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # Tokenize completion — the model IS trained to produce this
    completion_tokens = tokenizer.encode(
        f" {example['output']}\n\n", add_special_tokens=False
    )
    completion_weights = [1] * len(completion_tokens)

    # Concatenate and shift for next-token prediction
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )

processed = [process_example(ex, tokenizer) for ex in examples]
```

**What is happening here:**

1. The prompt `"English: banana split\nPig Latin:"` is tokenized into tokens. Each gets weight 0.
2. The completion `" anana-bay plit-say\n\n"` is tokenized. Each gets weight 1. (Note the leading space — it separates the completion from the colon.)
3. The full token sequence is shifted by one position to create input/target pairs for next-token prediction.
4. The result is packaged as a `Datum` with `model_input` (what the model sees) and `loss_fn_inputs` (targets and weights for the loss function).

If we visualize the tokens around the transition from prompt to completion:

```
Input Token      Target Token     Weight
-------------------------------------------
' Latin'         ':'              0        ← still prompt
':'              ' an'            1        ← completion starts here
' an'            'ana'            1
'ana'            '-'              1
'-'              'bay'            1
'bay'            ' p'             1
...
```

The model learns: "after seeing `English: banana split\nPig Latin:`, the next tokens should be ` anana-bay plit-say`."

### 4.6 Train

The training loop is remarkably simple — 6 steps, all 7 examples in each batch:

```python
import numpy as np

for step in range(6):
    # Forward + backward: compute gradients on Tinker's GPUs
    fwdbwd_future = training_client.forward_backward(
        processed, "cross_entropy"
    )

    # Optimizer step: update the LoRA adapter weights
    optim_future = training_client.optim_step(
        types.AdamParams(learning_rate=1e-4)
    )

    # Wait for results and compute loss
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    logprobs = np.concatenate(
        [out['logprobs'].tolist()
         for out in fwdbwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [ex.loss_fn_inputs['weights'].tolist()
         for ex in processed]
    )
    loss = -np.dot(logprobs, weights) / weights.sum()
    print(f"Step {step}: loss = {loss:.4f}")
```

**What is happening on each step:**

1. `forward_backward` sends all 7 tokenized examples to Tinker. The server runs the forward pass through Llama-3.2-1B + the LoRA adapter, computes the cross-entropy loss (weighted by the weight mask), and backpropagates to get gradients for the adapter parameters. It returns a *future* — the computation happens asynchronously.
2. `optim_step` tells the server to update the adapter weights using Adam with learning rate 1e-4. Also returns a future.
3. We call `.result()` on both futures to wait for completion.
4. We compute the weighted average loss from the returned log-probabilities. This should decrease over the 6 steps.

Note: both API calls return futures immediately. We submit both before waiting, which allows them to pipeline on the server.

**Expected output (approximate):**

```
Step 0: loss = 8.2341
Step 1: loss = 5.1892
Step 2: loss = 3.0156
Step 3: loss = 1.8743
Step 4: loss = 1.1205
Step 5: loss = 0.7831
```

The loss drops sharply because we have only 7 examples and the model can memorize them quickly. In a real task with thousands of examples, the loss would decrease more gradually.

### 4.7 Sample from the Fine-Tuned Model

To generate text, we first save the current adapter weights and create a sampling client:

```python
sampler = training_client.save_weights_and_get_sampling_client(
    name="pig-latin-model"
)

prompt = types.ModelInput.from_ints(
    tokenizer.encode("English: coffee break\nPig Latin:")
)
params = types.SamplingParams(
    max_tokens=20, temperature=0.0, stop=["\n"]
)
result = sampler.sample(
    prompt=prompt, sampling_params=params, num_samples=8
).result()

print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"  {i}: {tokenizer.decode(seq.tokens)}")
```

**Expected output (approximate):**

```
Responses:
  0:  offee-cay eak-bray
  1:  offey-cay eak-bray
  2:  offecay eakbray
  ...
```

The model has learned the Pig Latin pattern — it moves consonants to the end and appends "ay" — even for the phrase "coffee break" which was not in the training data. The outputs are not perfectly consistent (note the slight variations across samples), which reflects the fact that we trained on only 7 examples for 6 steps. More data and more training would improve consistency.

### 4.8 What Just Happened

Let's recap what the model *actually learned*:

- The original Llama-3.2-1B weights are **completely frozen**. Not a single one of its 1.2 billion parameters changed.
- A LoRA adapter with rank 32 was attached to the model's weight matrices. Only these adapter parameters were trained — roughly 20 million parameters, or about 1.6% of the base model.
- After 6 training steps on 7 examples, the adapter learned a pattern: "when you see `English: [X]\nPig Latin:`, produce the Pig Latin translation of X."
- The adapter weights are a small file that could be saved, shared, and swapped without reloading the base model.

All of the actual computation — the forward passes, the gradient calculations, the weight updates — happened on Tinker's GPU cluster. The Python script on your laptop only sent data and received results over the network.

---

## 5. Looking Ahead: The SQL Exercise

In the next exercise, you will write a fine-tuning script yourself. Instead of Pig Latin (a toy task with 7 examples), you will teach the same Llama-3.2-1B model to translate natural language questions into SQL queries, using a dataset of 78,000 real examples from WikiSQL and Spider.

The SQL task is a much stronger test of fine-tuning because:

- The transformation is compositional — the model must combine SQL syntax knowledge with schema understanding
- The schemas are diverse — thousands of different table definitions with different column names and types
- RAG falls short — you can retrieve similar question-SQL pairs, but the model still needs to *compose* novel SQL for unseen schemas
- The before/after contrast will be dramatic — base Llama-3.2-1B will produce gibberish SQL, while the fine-tuned version should generate valid queries

The code structure will be nearly identical to the Pig Latin walkthrough: load data, tokenize with weight masks, run a training loop, sample before and after. The only differences are the dataset (5,000 examples from a file instead of 7 handwritten ones) and the prompt template (table schemas instead of English phrases).

---

## 6. Key Takeaways

- **Fine-tuning teaches skills; RAG provides facts.** Use fine-tuning when you need the model to reliably perform a transformation or follow a format.
- **LoRA is the practical default.** It matches full fine-tuning quality for most tasks while using a fraction of the memory and compute. QLoRA goes further by quantizing the base weights to 4-bit.
- **The weight mask is the critical mechanism.** By setting weights to 0 on prompt tokens and 1 on completion tokens, you teach the model *what to produce* without confusing it about *what it was given*.
- **Tinker separates the algorithm from the infrastructure.** You write a simple Python loop on your laptop; the GPU work happens remotely. This is a clean abstraction that lets you focus on data and training decisions.
- **Small models + fine-tuning can surprise you.** Llama-3.2-1B (1.2 billion parameters) is tiny by modern standards, but fine-tuned on the right data, it can learn specific skills that much larger models cannot do out of the box.

---

## 7. References and Further Reading

- **LoRA paper**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA paper**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models" (2023). [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **LoRA Without Regret** (Thinking Machines analysis of LoRA vs full fine-tuning): [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora)
- **Tinker documentation**: [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai/)
- **Tinker Cookbook** (code examples for SL, RL, RLHF, tool use): [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

---

*End of Module — Introduction to Fine-Tuning LLMs*
