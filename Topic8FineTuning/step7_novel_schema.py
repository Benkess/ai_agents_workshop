import argparse
from pathlib import Path

from dotenv import load_dotenv

from finetune_sql import (
    create_service_and_tokenizer,
    load_latest_checkpoint,
    sample_from_model,
)
from sql_matches import sql_matches


DEFAULT_OUTPUT_PATH = Path(__file__).with_name("output") / "step7.txt"

NOVEL_SCHEMA_CASES = [
    {
        "label": "Easy 1",
        "difficulty": "easy",
        "schema": (
            "CREATE TABLE employees ("
            "id INTEGER, name VARCHAR, salary REAL, department VARCHAR)"
        ),
        "question": "What are the names of employees in the engineering department?",
        "reference_sql": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "label": "Easy 2",
        "difficulty": "easy",
        "schema": (
            "CREATE TABLE products ("
            "id INTEGER, name VARCHAR, price REAL, category VARCHAR)"
        ),
        "question": "How many products cost more than 50 dollars?",
        "reference_sql": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    {
        "label": "Medium 1",
        "difficulty": "medium",
        "schema": (
            "CREATE TABLE students ("
            "id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)"
        ),
        "question": "What is the highest score in the science class?",
        "reference_sql": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "label": "Medium 2",
        "difficulty": "medium",
        "schema": (
            "CREATE TABLE orders ("
            "id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)"
        ),
        "question": "List the top 3 customers by total order amount.",
        "reference_sql": (
            "SELECT customer FROM orders "
            "GROUP BY customer "
            "ORDER BY SUM(amount) DESC "
            "LIMIT 3"
        ),
    },
    {
        "label": "Hard 1",
        "difficulty": "hard",
        "schema": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
        "reference_sql": (
            "SELECT courses.department, COUNT(DISTINCT enrollments.student_id) "
            "FROM courses "
            "JOIN enrollments ON courses.id = enrollments.course_id "
            "GROUP BY courses.department"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run step 7 novel-schema SQL evaluation on base and fine-tuned models."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Tinker checkpoint path to use for the fine-tuned model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the step 7 text report.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    stored_path = load_latest_checkpoint()
    if stored_path:
        return stored_path

    raise ValueError(
        "Step 7 requires a saved sampler checkpoint. Run step 5 first or pass "
        "--checkpoint-path <tinker-path>."
    )


def evaluate_case(model_name: str, sampling_client, tokenizer, case: dict) -> dict:
    generated_sql = sample_from_model(
        sampling_client,
        tokenizer,
        case["schema"],
        case["question"],
    )
    is_match = sql_matches(
        generated_sql,
        case["reference_sql"],
        schema=case["schema"],
    )
    return {
        "model_name": model_name,
        "generated_sql": generated_sql,
        "is_match": is_match,
    }


def build_report(case_results: list[dict], checkpoint_path: str) -> str:
    lines = [
        "Step 7: Novel-Schema Evaluation",
        "",
        f"Fine-tuned checkpoint: {checkpoint_path}",
        f"Total cases: {len(case_results)}",
        "",
    ]

    base_matches = sum(1 for item in case_results if item["base"]["is_match"])
    fine_matches = sum(1 for item in case_results if item["fine_tuned"]["is_match"])
    lines.extend(
        [
            f"Base model matches: {base_matches}/{len(case_results)}",
            f"Fine-tuned model matches: {fine_matches}/{len(case_results)}",
            "",
        ]
    )

    for idx, item in enumerate(case_results, start=1):
        case = item["case"]
        base_status = "match" if item["base"]["is_match"] else "no match"
        fine_status = "match" if item["fine_tuned"]["is_match"] else "no match"
        lines.extend(
            [
                f"Case {idx}: {case['label']} ({case['difficulty']})",
                f"Question: {case['question']}",
                f"Schema: {case['schema']}",
                f"Reference SQL: {case['reference_sql']}",
                f"Base model [{base_status}]: {item['base']['generated_sql']}",
                f"Fine-tuned model [{fine_status}]: {item['fine_tuned']['generated_sql']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    load_dotenv()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    print("=== Running step 7: novel-schema evaluation ===")
    print(f"Using fine-tuned checkpoint: {checkpoint_path}")
    print("Creating base training client and tokenizer...")
    _, training_client, tokenizer = create_service_and_tokenizer()

    print("Creating base sampling client...")
    base_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="base-model-step7"
    )
    print("Creating fine-tuned sampling client...")
    fine_tuned_sampling_client = training_client.create_sampling_client(checkpoint_path)

    case_results = []
    for idx, case in enumerate(NOVEL_SCHEMA_CASES, start=1):
        print(f"Running case {idx}/{len(NOVEL_SCHEMA_CASES)}: {case['label']}")
        base_result = evaluate_case("base", base_sampling_client, tokenizer, case)
        fine_tuned_result = evaluate_case(
            "fine_tuned",
            fine_tuned_sampling_client,
            tokenizer,
            case,
        )
        case_results.append(
            {
                "case": case,
                "base": base_result,
                "fine_tuned": fine_tuned_result,
            }
        )

    report = build_report(case_results, checkpoint_path)
    print()
    print(report, end="")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    print(f"Saved step 7 report to {args.output_path}")


if __name__ == "__main__":
    main()
