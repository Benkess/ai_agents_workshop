import argparse
import json
import random
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv
from tinker import types

from sql_matches import sql_matches


BASE_MODEL = "meta-llama/Llama-3.2-1B"
NUM_TEST_EXAMPLES = 200
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_EPOCHS = 1
DEFAULT_LEARNING_RATE = 5e-4
CHECKPOINT_METADATA_PATH = Path(__file__).with_name("latest_sampler_checkpoint.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and fine-tune Llama 3.2 1B for text-to-SQL with Tinker."
    )
    parser.add_argument(
        "--step",
        choices=[3, 5, 6],
        default=3,
        type=int,
        help="Which lesson-plan step to run: 3 (base eval), 5 (train), or 6 (after eval).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Tinker checkpoint path to use for step 6.",
    )
    parser.add_argument(
        "--data-path",
        default=Path(__file__).with_name("sql_create_context_v4.json"),
        type=Path,
        help="Path to sql_create_context_v4.json",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help="Seed for reproducible train/test split.",
    )
    parser.add_argument(
        "--num-test-examples",
        default=NUM_TEST_EXAMPLES,
        type=int,
        help="Number of held-out evaluation examples.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=None,
        help="Optional cap for faster debugging. Uses all held-out examples by default.",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-epochs",
        default=DEFAULT_NUM_EPOCHS,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        default=DEFAULT_LEARNING_RATE,
        type=float,
        help="Adam learning rate for LoRA fine-tuning.",
    )
    return parser.parse_args()


def load_and_split_data(
    data_path: Path, seed: int, num_test_examples: int
) -> tuple[list[dict], list[dict]]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(data)

    test_data = data[:num_test_examples]
    train_data = data[num_test_examples:]
    return train_data, test_data


def format_prompt(example: dict) -> tuple[str, str]:
    prompt = (
        f"Table schema:\n{example['context']}\n"
        f"Question: {example['question']}\n"
        "SQL: "
    )
    completion = example["answer"]
    return prompt, completion


def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    prompt = f"Table schema:\n{context}\nQuestion: {question}\nSQL: "
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(
        max_tokens=150,
        temperature=0.0,
        stop=["\n\n", "Question:"],
    )
    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=params,
        num_samples=1,
    ).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""


def eval_one(sampling_client, tokenizer, example: dict) -> bool:
    sql = sample_from_model(
        sampling_client,
        tokenizer,
        example["context"],
        example["question"],
    )
    return sql_matches(sql, example["answer"], schema=example["context"])


def evaluate_test_set(sampling_client, tokenizer, test_data: list[dict]) -> float:
    correct = 0
    total = len(test_data)
    progress_interval = 1 if total <= 10 else 5 if total <= 50 else 10

    for idx, ex in enumerate(test_data, start=1):
        is_correct = eval_one(sampling_client, tokenizer, ex)
        if is_correct:
            correct += 1

        if idx == 1 or idx % progress_interval == 0 or idx == total:
            print(
                f"Evaluation progress: {idx}/{total} "
                f"(running accuracy: {correct / idx:.2%})"
            )

    return correct / len(test_data)


def process_example(example: dict, tokenizer) -> types.Datum:
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

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


def compute_batch_loss(fwdbwd_result, batch: list[types.Datum]) -> float:
    def to_arr(value):
        return value.to_numpy() if hasattr(value, "to_numpy") else np.array(value.tolist())

    logprobs = np.concatenate(
        [to_arr(output["logprobs"]) for output in fwdbwd_result.loss_fn_outputs]
    )
    weights = np.concatenate([to_arr(datum.loss_fn_inputs["weights"]) for datum in batch])
    return float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))


def run_training(
    training_client,
    tokenizer,
    train_data: list[dict],
    *,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    processed_train = [process_example(ex, tokenizer) for ex in train_data]
    rng = random.Random(seed)
    step = 0

    print(
        f"Preparing {len(processed_train)} training examples "
        f"for {num_epochs} epoch(s) with batch size {batch_size}..."
    )

    for epoch in range(num_epochs):
        rng.shuffle(processed_train)

        for batch_idx in range(0, len(processed_train), batch_size):
            batch = processed_train[batch_idx : batch_idx + batch_size]
            if not batch:
                continue

            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=learning_rate)
            )

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            step += 1
            if step % 100 == 0 or batch_idx + batch_size >= len(processed_train):
                loss = compute_batch_loss(fwdbwd_result, batch)
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"update {step}, loss: {loss:.4f}"
                )


def save_latest_checkpoint(path: str) -> None:
    metadata = {"checkpoint_path": path}
    CHECKPOINT_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved latest sampler checkpoint metadata to {CHECKPOINT_METADATA_PATH.name}")


def load_latest_checkpoint() -> str | None:
    if not CHECKPOINT_METADATA_PATH.exists():
        return None

    try:
        metadata = json.loads(CHECKPOINT_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    checkpoint_path = metadata.get("checkpoint_path")
    return checkpoint_path if checkpoint_path else None


def create_service_and_tokenizer():
    print("Creating Tinker service client...")
    service_client = tinker.ServiceClient()
    print(f"Creating LoRA training client for {BASE_MODEL}...")
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    print("Loading tokenizer...")
    tokenizer = training_client.get_tokenizer()
    return service_client, training_client, tokenizer


def run_step_3(eval_data: list[dict]) -> None:
    print("\n=== Running step 3: evaluate base model ===")
    _, training_client, tokenizer = create_service_and_tokenizer()
    print("Saving current base weights and creating sampling client...")
    base_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="base-model"
    )
    print("Starting base-model evaluation...")
    base_accuracy = evaluate_test_set(base_sampling_client, tokenizer, eval_data)
    print(
        f"Base model accuracy: {base_accuracy:.2%} "
        f"({int(round(base_accuracy * len(eval_data)))}/{len(eval_data)})"
    )


def run_step_5(args: argparse.Namespace, train_data: list[dict]) -> str:
    print("\n=== Running step 5: train fine-tuned model ===")
    _, training_client, tokenizer = create_service_and_tokenizer()
    run_training(
        training_client,
        tokenizer,
        train_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("Saving fine-tuned sampler weights...")
    save_result = training_client.save_weights_for_sampler("fine-tuned-model").result()
    checkpoint_path = save_result.path
    print(f"Saved fine-tuned sampler checkpoint: {checkpoint_path}")
    save_latest_checkpoint(checkpoint_path)
    return checkpoint_path


def resolve_checkpoint_path(explicit_path: str | None) -> str:
    if explicit_path:
        print(f"Using checkpoint path from --checkpoint-path: {explicit_path}")
        return explicit_path

    stored_path = load_latest_checkpoint()
    if stored_path:
        print(f"Using latest saved checkpoint path from metadata: {stored_path}")
        return stored_path

    raise ValueError(
        "Step 6 requires a saved sampler checkpoint. Run step 5 first or pass "
        "--checkpoint-path <tinker-path>."
    )


def run_step_6(args: argparse.Namespace, eval_data: list[dict]) -> None:
    print("\n=== Running step 6: evaluate fine-tuned model ===")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    service_client, training_client, tokenizer = create_service_and_tokenizer()
    print("Creating sampling client from saved checkpoint...")
    fine_tuned_sampling_client = training_client.create_sampling_client(checkpoint_path)
    print("Starting fine-tuned-model evaluation...")
    fine_tuned_accuracy = evaluate_test_set(
        fine_tuned_sampling_client,
        tokenizer,
        eval_data,
    )
    print(
        f"Fine-tuned model accuracy: {fine_tuned_accuracy:.2%} "
        f"({int(round(fine_tuned_accuracy * len(eval_data)))}/{len(eval_data)})"
    )


def main() -> None:
    args = parse_args()
    load_dotenv()

    train_data, test_data = load_and_split_data(
        args.data_path,
        seed=args.seed,
        num_test_examples=args.num_test_examples,
    )
    eval_data = (
        test_data[: args.max_eval_examples]
        if args.max_eval_examples is not None
        else test_data
    )

    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Evaluating on: {len(eval_data)} examples")

    if args.step == 3:
        run_step_3(eval_data)
        return

    if args.step == 5:
        run_step_5(args, train_data)
        return

    if args.step == 6:
        run_step_6(args, eval_data)
        return

    raise ValueError(f"Unsupported step: {args.step}")


if __name__ == "__main__":
    main()
