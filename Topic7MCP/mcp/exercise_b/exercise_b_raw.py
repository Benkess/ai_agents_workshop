import json
import os
import sys
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


def call_tool_raw(name: str, arguments: dict[str, Any], request_id: int) -> str:
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
    return response.text


def print_raw(title: str, name: str, arguments: dict[str, Any], request_id: int) -> None:
    print(f"=== {title} ===")
    print(f"Tool: {name}")
    print("Arguments:")
    print(json.dumps(arguments, indent=2))
    print("Raw response:")
    print(call_tool_raw(name, arguments, request_id))
    print()


def main() -> int:
    try:
        print_raw(
            "Drill 1 raw response",
            "search_papers",
            {
                "query": "large language model agents",
                "fields": "title,abstract,year,authors",
                "limit": 5,
            },
            request_id=1,
        )
        print_raw(
            "Drill 1 fallback raw response",
            "search_papers_by_relevance",
            {
                "keyword": "large language model agents",
                "fields": "title,abstract,year,authors",
                "limit": 5,
            },
            request_id=11,
        )
        print_raw(
            "Drill 2 raw response",
            "get_citations",
            {
                "paper_id": "ARXIV:1810.04805",
                "fields": "title,year,authors",
                "limit": 10,
                "publication_date_range": "2023-01-01:",
            },
            request_id=2,
        )
        print_raw(
            "Drill 3 raw response",
            "get_references",
            {
                "paper_id": "ARXIV:2210.03629",
                "fields": "title,year,authors",
                "limit": 100,
            },
            request_id=3,
        )
        print_raw(
            "Drill 3 fallback raw response",
            "get_paper",
            {
                "paper_id": "ARXIV:2210.03629",
                "fields": "title,references",
            },
            request_id=33,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
