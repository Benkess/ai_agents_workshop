import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a saved Exercise C message history JSON file as Markdown."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a saved message history JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Markdown path. Defaults to the input path with a .md suffix.",
    )
    parser.add_argument(
        "--title",
        default="Exercise C Message History",
        help="Markdown document title.",
    )
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a list of messages in {path}, got: {type(payload).__name__}")
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Expected message {index} in {path} to be an object, got: {type(item).__name__}"
            )
    return payload


def to_json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


def parse_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def render_content(content: Any) -> list[str]:
    if content is None:
        return ["**Content**", "", "_None_"]
    if isinstance(content, str):
        if not content.strip():
            return ["**Content**", "", "_Empty string_"]
        return ["**Content**", "", "```text", content, "```"]
    return ["**Content**", "", to_json_block(content)]


def render_tool_calls(tool_calls: Any) -> list[str]:
    lines = ["**Tool Calls**", ""]

    if not isinstance(tool_calls, list):
        lines.append(to_json_block(tool_calls))
        return lines

    for index, tool_call in enumerate(tool_calls, start=1):
        lines.append(f"### Tool Call {index}")
        if not isinstance(tool_call, dict):
            lines.append("")
            lines.append(to_json_block(tool_call))
            lines.append("")
            continue

        lines.append("")
        lines.append(f"- `id`: `{tool_call.get('id', '')}`")
        lines.append(f"- `type`: `{tool_call.get('type', '')}`")

        function_payload = tool_call.get("function")
        if isinstance(function_payload, dict):
            lines.append(f"- `function.name`: `{function_payload.get('name', '')}`")
            arguments = parse_arguments(function_payload.get("arguments"))
            lines.append("- `function.arguments`:")
            lines.append("")
            if isinstance(arguments, str):
                lines.extend(["```text", arguments, "```"])
            else:
                lines.append(to_json_block(arguments))
        else:
            lines.append("- `function`:")
            lines.append("")
            lines.append(to_json_block(function_payload))

        lines.append("")
        lines.append("**Raw Tool Call**")
        lines.append("")
        lines.append(to_json_block(tool_call))
        lines.append("")

    return lines


def render_message(index: int, message: dict[str, Any]) -> list[str]:
    role = message.get("role", "unknown")
    lines = [f"## Message {index}: `{role}`", ""]
    lines.append(f"- `role`: `{role}`")

    if "tool_call_id" in message:
        lines.append(f"- `tool_call_id`: `{message['tool_call_id']}`")

    lines.append("")

    if "content" in message:
        lines.extend(render_content(message["content"]))
        lines.append("")

    if "tool_calls" in message:
        lines.extend(render_tool_calls(message["tool_calls"]))
        lines.append("")

    handled_keys = {"role", "content", "tool_calls", "tool_call_id"}
    extra = {key: value for key, value in message.items() if key not in handled_keys}
    if extra:
        lines.append("**Additional Fields**")
        lines.append("")
        lines.append(to_json_block(extra))
        lines.append("")

    lines.append("**Raw Message JSON**")
    lines.append("")
    lines.append(to_json_block(message))
    lines.append("")
    return lines


def render_markdown(title: str, source_path: Path, messages: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title}",
        "",
        f"Source: `{source_path}`",
        "",
        f"Total messages: {len(messages)}",
        "",
    ]

    for index, message in enumerate(messages, start=1):
        lines.extend(render_message(index, message))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_json
    output_path = args.output or input_path.with_suffix(".md")

    messages = load_history(input_path)
    markdown = render_markdown(args.title, input_path, messages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote Markdown history to {output_path}")


if __name__ == "__main__":
    main()
