import argparse
import json
import os
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


def parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        pass

    data_lines = []
    for line in response.text.strip().splitlines():
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if not data_lines:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}")

    try:
        return json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Asta MCP returned invalid JSON: {response.text}") from exc


def call_tool_mcp(name: str, arguments: dict[str, Any], request_id: int) -> tuple[dict[str, Any], str]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }
    response = requests.post(
        ASTA_MCP_URL,
        headers=build_headers(),
        json=payload,
        timeout=30,
    )
    return parse_mcp_response(response), response.text


def parse_content_texts(mcp_payload: dict[str, Any]) -> Any:
    content = mcp_payload.get("result", {}).get("content", [])
    parsed: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            parsed.append(item)
            continue
        text_value = item.get("text")
        if not isinstance(text_value, str):
            parsed.append(item)
            continue
        try:
            parsed.append(json.loads(text_value))
        except json.JSONDecodeError:
            parsed.append(text_value)
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def save_response(
    outdir: Path,
    filename: str,
    tool_name: str,
    arguments: dict[str, Any],
    request_id: int,
) -> None:
    mcp_payload, raw_text = call_tool_mcp(tool_name, arguments, request_id)
    output = {
        "tool": tool_name,
        "arguments": arguments,
        "request_id": request_id,
        "mcp_payload": mcp_payload,
        "structured_result": mcp_payload.get("result", {}).get("structuredContent"),
        "parsed_content": parse_content_texts(mcp_payload),
        "raw_response_text": raw_text,
    }
    path = outdir / filename
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save Exercise B tool responses into JSON files."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("Topic7MCP/mcp/exercise_b/json_responses"),
        help="Directory to write JSON response files.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        args.outdir.mkdir(parents=True, exist_ok=True)

        save_response(
            args.outdir,
            "drill1_search_papers_by_relevance.json",
            "search_papers_by_relevance",
            {
                "keyword": "large language model agents",
                "fields": "title,abstract,year,authors",
                "limit": 5,
            },
            request_id=1,
        )
        save_response(
            args.outdir,
            "drill2_get_citations.json",
            "get_citations",
            {
                "paper_id": "ARXIV:1810.04805",
                "fields": "title,year,authors",
                "limit": 10,
                "publication_date_range": "2023-01-01:",
            },
            request_id=2,
        )
        save_response(
            args.outdir,
            "drill3_get_paper_references.json",
            "get_paper",
            {
                "paper_id": "ARXIV:2210.03629",
                "fields": "title,references",
            },
            request_id=3,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
