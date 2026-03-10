"""
Exercise A answers:
- To find papers about "transformer attention mechanisms", use `search_papers`.
- To find who else published in the same area as a specific author, start with
  `search_authors_by_name` to identify the author, then use
  `get_author_papers` to inspect their publications and related authors.
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any

import requests


ASTA_MCP_URL = "https://asta-tools.allen.ai/mcp/v1"


def build_headers() -> dict[str, str]:
    api_key = os.getenv("ASTA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ASTA_API_KEY is not set. In PowerShell, run: "
            '$env:ASTA_API_KEY="your_class_key"'
        )

    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }


def build_tools_list_payload() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }


def parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        pass

    body = response.text.strip()
    data_lines = []

    for line in body.splitlines():
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if not data_lines:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}")

    data_text = "\n".join(data_lines)
    try:
        return json.loads(data_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}") from exc


def fetch_tools() -> list[dict[str, Any]]:
    response = requests.post(
        ASTA_MCP_URL,
        headers=build_headers(),
        json=build_tools_list_payload(),
        timeout=30,
    )

    payload = parse_mcp_response(response)

    if not response.ok:
        raise RuntimeError(
            f"Asta MCP request failed with HTTP {response.status_code}: "
            f"{payload.get('error', response.text.strip())}"
        )

    if "error" in payload:
        raise RuntimeError(f"Asta MCP returned a JSON-RPC error: {payload['error']}")

    try:
        return payload["result"]["tools"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Asta MCP response shape: {payload}"
        ) from exc


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def format_schema_type(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        readable_types = [str(item) for item in schema_type]
        return " | ".join(readable_types)
    if isinstance(schema_type, str):
        return schema_type
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "unknown"


def format_parameters(properties: dict[str, Any], names: list[str]) -> str:
    if not names:
        return "None"

    formatted = []
    for name in names:
        schema = properties.get(name, {})
        formatted.append(f"{name} ({format_schema_type(schema)})")
    return ", ".join(formatted)


def render_tool_summary(tool: dict[str, Any], output_format: str) -> str:
    name = tool.get("name", "<unknown>")
    description = collapse_whitespace(tool.get("description", "No description provided."))
    input_schema = tool.get("inputSchema") or {}
    properties = input_schema.get("properties") or {}
    required = input_schema.get("required") or []
    optional = [prop_name for prop_name in properties if prop_name not in required]

    required_text = format_parameters(properties, required)
    optional_text = format_parameters(properties, optional)

    if output_format == "md":
        return (
            f"## Tool: {name}\n"
            f"- Description: {description}\n"
            f"- Required: {required_text}\n"
            f"- Optional: {optional_text}\n"
        )

    return (
        f"Tool: {name}\n"
        f"  Description: {description}\n"
        f"  Required: {required_text}\n"
        f"  Optional: {optional_text}\n"
    )


def build_output(tools: list[dict[str, Any]], output_format: str) -> str:
    return "\n".join(render_tool_summary(tool, output_format) for tool in tools).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Asta MCP tools and optionally save the formatted output."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the output. Supported extensions: .txt, .md",
    )
    return parser.parse_args()


def resolve_output_format(output_path: Path | None) -> str:
    if output_path is None:
        return "txt"

    suffix = output_path.suffix.lower()
    if suffix not in {".txt", ".md"}:
        raise RuntimeError("Output path must end with .txt or .md")
    return suffix.lstrip(".")


def write_output(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    try:
        args = parse_args()
        tools = fetch_tools()
        output_format = resolve_output_format(args.output)
        content = build_output(tools, output_format)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        try:
            write_output(args.output, content)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Saved output to {args.output}")
        return 0

    print(content, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
