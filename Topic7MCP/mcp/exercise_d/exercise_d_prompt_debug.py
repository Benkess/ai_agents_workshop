import argparse
import sys
from pathlib import Path

from exercise_d import (
    build_generation_messages,
    build_payload,
    messages_to_chat_dicts,
    save_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Exercise D prompt and save it as markdown for debugging."
    )
    parser.add_argument("paper_id", help="Seed paper identifier, e.g. ARXIV:2210.03629")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Topic7MCP/mcp/exercise_d/out/prompt_debug.md"),
        help="Markdown file to write the debug prompt into.",
    )
    parser.add_argument(
        "--author-limit",
        type=int,
        default=5,
        help="Maximum number of author profiles to include unless --all-authors is set.",
    )
    parser.add_argument(
        "--all-authors",
        action="store_true",
        help="Include all seed-paper authors in Author Profiles.",
    )
    return parser.parse_args()


def count_approx_tokens(text: str) -> int:
    # Rough debug-only estimate for prompt sizing.
    return max(1, len(text) // 4) if text else 0


def render_prompt_debug_markdown(
    *,
    paper_id: str,
    output_path: Path,
    author_limit: int | None,
    messages: list[dict[str, str]],
) -> str:
    system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
    combined = system_message + "\n\n" + user_message

    if author_limit is None:
        author_limit_label = "all"
    else:
        author_limit_label = str(author_limit)

    return (
        f"# Exercise D Prompt Debug\n\n"
        f"- Paper ID: `{paper_id}`\n"
        f"- Output file: `{output_path}`\n"
        f"- Author limit mode: `{author_limit_label}`\n"
        f"- System prompt characters: `{len(system_message)}`\n"
        f"- User prompt characters: `{len(user_message)}`\n"
        f"- Combined prompt characters: `{len(combined)}`\n"
        f"- Combined prompt bytes (UTF-8): `{len(combined.encode('utf-8'))}`\n"
        f"- Approximate combined tokens: `{count_approx_tokens(combined)}`\n\n"
        f"## System Message\n\n"
        f"```text\n{system_message}\n```\n\n"
        f"## User Message\n\n"
        f"```text\n{user_message}\n```\n"
    )


def main() -> int:
    args = parse_args()
    author_limit = None if args.all_authors else args.author_limit

    try:
        payload = build_payload(args.paper_id, author_limit)
    except Exception as exc:
        print(f"Failed to retrieve prompt data: {exc}", file=sys.stderr)
        return 1

    messages = messages_to_chat_dicts(build_generation_messages(payload))
    markdown = render_prompt_debug_markdown(
        paper_id=args.paper_id,
        output_path=args.output,
        author_limit=author_limit,
        messages=messages,
    )

    save_markdown(args.output, markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
