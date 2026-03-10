"""
Exercise B note:
- The live Asta MCP server tool names for this exercise are:
  - Drill 1: `search_papers_by_relevance`
  - Drill 2: `get_citations`
  - Drill 3: `get_paper` with `fields="title,references"`
"""

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


def call_tool(name: str, arguments: dict[str, Any], request_id: int) -> Any:
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
    mcp_payload = parse_mcp_response(response)

    if not response.ok:
        raise RuntimeError(
            f"Asta MCP request failed with HTTP {response.status_code}: "
            f"{mcp_payload.get('error', response.text.strip())}"
        )

    if "error" in mcp_payload:
        raise RuntimeError(f"Asta MCP returned a JSON-RPC error: {mcp_payload['error']}")

    try:
        result_payload = mcp_payload["result"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected tool response shape for {name}: {mcp_payload}"
        ) from exc

    if result_payload.get("isError"):
        content = result_payload.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            raise RuntimeError("\n".join(parts))
        raise RuntimeError(str(result_payload))

    structured = result_payload.get("structuredContent")
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    if structured is not None:
        return structured

    content = result_payload.get("content")
    if isinstance(content, list):
        parsed_items = []
        for item in content:
            if not isinstance(item, dict):
                parsed_items.append(item)
                continue
            text_content = item.get("text")
            if not isinstance(text_content, str):
                parsed_items.append(item)
                continue
            try:
                parsed_items.append(json.loads(text_content))
            except json.JSONDecodeError:
                parsed_items.append(text_content)
        if len(parsed_items) == 1:
            return parsed_items[0]
        return parsed_items

    return result_payload


def first_list_in_payload(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("result", "data", "papers", "citations", "references", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for value in payload.values():
            nested = first_list_in_payload(value)
            if nested is not None:
                return nested
    return None


def unwrap_record(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"title": str(item)}

    for key in ("paper", "citingPaper", "citedPaper", "reference", "referencedPaper"):
        nested = item.get(key)
        if isinstance(nested, dict):
            return nested
    return item


def normalize_result(payload: Any) -> list[dict[str, Any]]:
    records = first_list_in_payload(payload)
    if records is None:
        if isinstance(payload, dict):
            return [unwrap_record(payload)]
        raise RuntimeError(f"Unexpected parsed tool payload: {payload}")
    return [unwrap_record(record) for record in records]


def format_title_and_year(item: dict[str, Any], index: int | None = None) -> str:
    title = item.get("title") or "Untitled"
    year = item.get("year")
    year_text = str(year) if year is not None else "Unknown year"
    prefix = f"{index}. " if index is not None else "- "
    return f"{prefix}{title} ({year_text})"


def render_section(title: str, lines: list[str], output_format: str) -> str:
    if output_format == "md":
        return f"## {title}\n" + "\n".join(lines) + "\n"
    return f"=== {title} ===\n" + "\n".join(lines) + "\n"


def render_tool_call_lines(name: str, arguments: dict[str, Any], output_format: str) -> list[str]:
    payload_text = json.dumps(arguments, ensure_ascii=False)
    if output_format == "md":
        return [f"- Tool call: `{name}`", f"- Arguments: `{payload_text}`"]
    return [f"Tool call: {name}", f"Arguments: {payload_text}"]


def run_drill_1(output_format: str) -> str:
    arguments = {
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 5,
    }
    result = call_tool("search_papers_by_relevance", arguments, request_id=1)
    lines = render_tool_call_lines("search_papers_by_relevance", arguments, output_format)

    papers = normalize_result(result)
    for index, paper in enumerate(papers[:5], start=1):
        lines.append(format_title_and_year(paper, index=index))
    return render_section("Drill 1: Recent LLM agent papers", lines, output_format)


def run_drill_2(output_format: str) -> str:
    arguments = {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 10,
        "publication_date_range": "2023-01-01:",
    }
    result = call_tool("get_citations", arguments, request_id=2)

    citations = normalize_result(result)
    lines = render_tool_call_lines("get_citations", arguments, output_format)
    lines.append(f"Total citations returned: {len(citations)}")
    for citation in citations[:5]:
        lines.append(f"- {citation.get('title', 'Untitled')}")
    return render_section("Drill 2: Citations to the original BERT paper", lines, output_format)


def extract_references(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("references"), list):
        refs = [unwrap_record(reference) for reference in payload["references"]]
        return [ref for ref in refs if ref.get("paperId") or ref.get("year")]

    records = normalize_result(payload)
    first_record = records[0] if records else {}
    references = first_list_in_payload(first_record.get("references"))
    if references is not None:
        refs = [unwrap_record(reference) for reference in references]
        return [ref for ref in refs if ref.get("paperId") or ref.get("year")]
    return records


def sort_references_by_year(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        references,
        key=lambda item: (
            item.get("year") is None,
            item.get("year") if item.get("year") is not None else 9999,
            item.get("title") or "",
        ),
    )


def run_drill_3(output_format: str) -> str:
    arguments = {
        "paper_id": "ARXIV:2210.03629",
        "fields": "title,references",
    }
    result = call_tool("get_paper", arguments, request_id=3)
    lines = render_tool_call_lines("get_paper", arguments, output_format)

    references = extract_references(result)
    sorted_references = sort_references_by_year(references)

    if not sorted_references:
        lines.append("No references were returned.")
        return render_section("Drill 3: References used in the ReAct paper", lines, output_format)

    for reference in sorted_references:
        lines.append(format_title_and_year(reference))
    return render_section("Drill 3: References used in the ReAct paper", lines, output_format)


def build_output(output_format: str) -> str:
    sections = [
        run_drill_1(output_format),
        run_drill_2(output_format),
        run_drill_3(output_format),
    ]
    return "\n".join(section.rstrip() for section in sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Exercise B Asta MCP drills and optionally save the output."
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
        output_format = resolve_output_format(args.output)
        content = build_output(output_format)
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
