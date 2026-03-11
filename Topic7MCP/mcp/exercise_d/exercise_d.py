import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTA_API_KEY = os.environ["ASTA_API_KEY"]

MODEL = "gpt-4o-mini"
MCP_URL = "https://asta-tools.allen.ai/mcp/v1"
RECENT_CITATIONS_START = "2023-03-11:"
MAX_ERROR_NOTE_CHARS = 240
MAX_PROMPT_TEXT_CHARS = 1600

client = OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = (
    "You are generating a structured markdown research report from retrieved paper data. "
    "This is not a chat. Write only markdown. "
    "Use the provided evidence to produce: "
    "1) a one-paragraph summary of the seed paper, "
    "2) a 'Foundational Works' section with exactly the provided key references, "
    "3) a 'Recent Developments' section with exactly the provided citing papers, and "
    "4) an 'Author Profiles' section with the provided notable works. "
    "If any retrieved data is missing, state that explicitly and do not invent details."
)


class ReportState(TypedDict):
    payload: dict[str, Any]
    messages: Annotated[list[AnyMessage], add_messages]
    report_markdown: str


def mcp_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "x-api-key": ASTA_API_KEY,
    }


def parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError:
        pass

    # Only split on transport newlines. `str.splitlines()` is unsafe here because
    # the MCP payload may legitimately contain Unicode line separators inside JSON strings.
    raw_text = response.text.replace("\r\n", "\n").replace("\r", "\n")
    data_lines = []
    for line in raw_text.strip().split("\n"):
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
        elif data_lines:
            data_lines.append(line)

    if not data_lines:
        raise RuntimeError("Asta MCP returned no parseable JSON data.")

    try:
        return json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Asta MCP returned invalid JSON data (chars={len(raw_text)}, pos={exc.pos})."
        ) from exc


def parse_json_sequence(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    items: list[Any] = []
    index = 0
    length = len(text)

    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        item, next_index = decoder.raw_decode(text, index)
        items.append(item)
        index = next_index

    return items


def call_asta_tool(name: str, arguments: dict[str, Any]) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }

    response = requests.post(MCP_URL, headers=mcp_headers(), json=payload, timeout=120)
    data = parse_mcp_response(response)

    if not response.ok:
        raise RuntimeError(
            f"Asta MCP {name} failed with HTTP {response.status_code}: "
            f"{data.get('error', response.text.strip())}"
        )

    if "error" in data:
        raise RuntimeError(f"Asta MCP {name} error: {data['error']}")

    content = data["result"].get("content", [])
    if not content:
        return None

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        else:
            text_parts.append(json.dumps(item))

    text = "\n".join(part for part in text_parts if part).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            items = parse_json_sequence(text)
        except json.JSONDecodeError:
            return text
        if len(items) == 1:
            return items[0]
        return items


def safe_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    required: bool = False,
    fallback: Any = None,
) -> Any:
    try:
        return call_asta_tool(name, arguments)
    except Exception as exc:
        if required:
            raise
        return {
            "_error": summarize_error(name, exc),
            "_fallback": fallback,
        }


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("result", "data", "papers", "authors", "references", "citations"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    return []


def citation_count_from_paper(paper: dict[str, Any]) -> int:
    count = paper.get("citationCount")
    if isinstance(count, int):
        return count

    citations = paper.get("citations")
    if isinstance(citations, list):
        return len(citations)

    return 0


def paper_title(paper: dict[str, Any]) -> str:
    return str(paper.get("title") or "Untitled paper")


def normalize_author(author: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": author.get("name") or "Unknown author",
        "authorId": author.get("authorId"),
        "affiliations": author.get("affiliations") or [],
        "citationCount": author.get("citationCount"),
        "paperCount": author.get("paperCount"),
    }


def normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    authors = [normalize_author(author) for author in as_list(paper.get("authors"))]
    fields = paper.get("fieldsOfStudy") or []
    if isinstance(fields, str):
        fields = [fields]

    normalized = {
        "paperId": paper.get("paperId"),
        "corpusId": paper.get("corpusId"),
        "title": paper.get("title") or "Untitled paper",
        "abstract": paper.get("abstract") or "",
        "year": paper.get("year"),
        "publicationDate": paper.get("publicationDate"),
        "venue": paper.get("venue"),
        "url": paper.get("url"),
        "tldr": paper.get("tldr"),
        "fieldsOfStudy": fields,
        "authors": authors,
        "citationCount": citation_count_from_paper(paper),
    }
    missing_notes = []
    if not normalized["abstract"]:
        missing_notes.append("Abstract unavailable in MCP response.")
    if not normalized["authors"]:
        missing_notes.append("Author metadata unavailable in MCP response.")
    if paper.get("missing_data_note"):
        missing_notes.append(str(paper["missing_data_note"]))
    if missing_notes:
        normalized["missing_data_note"] = " ".join(missing_notes)
    return normalized


def clip_text(text: Any, max_chars: int = MAX_PROMPT_TEXT_CHARS) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 14].rstrip() + " [truncated]"


def summarize_error(tool_name: str, exc: Exception) -> str:
    message = " ".join(str(exc).split())
    summarized = f"{tool_name} failed: {message}"
    if len(summarized) <= MAX_ERROR_NOTE_CHARS:
        return summarized
    return summarized[: MAX_ERROR_NOTE_CHARS - 14].rstrip() + " [truncated]"


def paper_needs_backfill(paper: dict[str, Any]) -> bool:
    return not paper.get("abstract") or not paper.get("authors")


def extract_backfillable_id(paper: dict[str, Any]) -> str | None:
    for key in ("paperId", "paper_id"):
        value = paper.get(key)
        if value:
            return str(value)
    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict):
        for key in ("ArXiv", "DOI", "CorpusId"):
            value = external_ids.get(key)
            if value:
                if key == "ArXiv":
                    return f"ARXIV:{value}"
                if key == "DOI":
                    return f"DOI:{value}"
                if key == "CorpusId":
                    return f"CorpusId:{value}"
    return None


def build_missing_entry(label: str, error: str) -> dict[str, Any]:
    return {
        "title": label,
        "abstract": "",
        "year": None,
        "publicationDate": None,
        "venue": None,
        "url": None,
        "fieldsOfStudy": [],
        "authors": [],
        "citationCount": 0,
        "missing_data_note": clip_text(error, MAX_ERROR_NOTE_CHARS),
    }


def fetch_seed_paper(paper_id: str) -> dict[str, Any]:
    result = safe_tool_call(
        "get_paper",
        {
            "paper_id": paper_id,
            "fields": "title,abstract,year,authors,fieldsOfStudy,citations,references,tldr,url,venue,publicationDate",
        },
        required=True,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected get_paper result type: {type(result).__name__}")
    return normalize_paper(result)


def fetch_seed_reference_ids(seed_paper_id: str) -> list[str]:
    result = safe_tool_call(
        "get_paper",
        {
            "paper_id": seed_paper_id,
            "fields": "references",
        },
        required=True,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected get_paper references result type: {type(result).__name__}")

    reference_ids = []
    for reference in as_list(result.get("references")):
        if isinstance(reference, dict):
            backfill_id = extract_backfillable_id(reference)
            if backfill_id:
                reference_ids.append(backfill_id)
    return reference_ids


def fetch_top_references(seed_paper_id: str) -> list[dict[str, Any]]:
    try:
        reference_ids = fetch_seed_reference_ids(seed_paper_id)
    except Exception as exc:
        return [build_missing_entry("References unavailable", f"Failed to fetch references: {exc}")]

    if not reference_ids:
        return [build_missing_entry("References unavailable", "No references were returned.")]

    batch_result = safe_tool_call(
        "get_paper_batch",
        {
            "ids": reference_ids,
            "fields": "title,abstract,year,authors,citations,fieldsOfStudy,url,venue,publicationDate",
        },
        fallback=[],
    )
    if isinstance(batch_result, dict) and "_error" in batch_result:
        return [build_missing_entry("References unavailable", batch_result["_error"])]

    references = [paper for paper in as_list(batch_result) if isinstance(paper, dict)]
    ranked = sorted(references, key=citation_count_from_paper, reverse=True)[:5]
    if not ranked:
        return [build_missing_entry("References unavailable", "Reference enrichment returned no papers.")]

    return [normalize_paper(paper) for paper in ranked]


def fetch_recent_citations(seed_paper_id: str) -> list[dict[str, Any]]:
    result = safe_tool_call(
        "get_citations",
        {
            "paper_id": seed_paper_id,
            "fields": "title,abstract,year,authors,fieldsOfStudy,url,venue,publicationDate",
            "limit": 20,
            "publication_date_range": RECENT_CITATIONS_START,
        },
        fallback=[],
    )
    if isinstance(result, dict) and "_error" in result:
        return [build_missing_entry("Recent developments unavailable", result["_error"])]

    citations = []
    for item in as_list(result) if not isinstance(result, list) else result:
        if isinstance(item, dict):
            citing_paper = item.get("citingPaper")
            if isinstance(citing_paper, dict):
                citations.append(citing_paper)
            else:
                citations.append(item)
    recent = citations[:5]
    if not recent:
        return [build_missing_entry("Recent developments unavailable", "No recent citing papers were returned.")]
    return [normalize_paper(paper) for paper in recent]


def choose_other_top_paper(
    papers: list[dict[str, Any]],
    *,
    seed_title: str,
) -> dict[str, Any] | None:
    seed_title_norm = seed_title.casefold().strip()
    candidates = []
    for paper in papers:
        title = paper_title(paper).casefold().strip()
        if title == seed_title_norm:
            continue
        candidates.append(paper)

    if not candidates:
        return None

    candidates.sort(key=citation_count_from_paper, reverse=True)
    return candidates[0]


def fetch_full_paper_details(paper_id: str | None, fallback_paper: dict[str, Any]) -> dict[str, Any]:
    if not paper_id:
        return normalize_paper(fallback_paper)

    result = safe_tool_call(
        "get_paper",
        {
            "paper_id": paper_id,
            "fields": "title,abstract,year,authors,citations,fieldsOfStudy,url,venue,publicationDate",
        },
        fallback=None,
    )
    if isinstance(result, dict) and "_error" in result:
        paper = normalize_paper(fallback_paper)
        paper["missing_data_note"] = result["_error"]
        return paper
    if isinstance(result, dict):
        return normalize_paper(result)
    return normalize_paper(fallback_paper)


def fetch_author_profiles(
    authors: list[dict[str, Any]],
    *,
    seed_title: str,
    author_limit: int | None,
) -> list[dict[str, Any]]:
    selected_authors = authors if author_limit is None else authors[:author_limit]
    if not selected_authors:
        return [
            {
                "author": {"name": "Unknown author", "authorId": None, "affiliations": []},
                "notable_work": build_missing_entry("No author data", "The seed paper did not include authors."),
            }
        ]

    profiles: list[dict[str, Any]] = []
    for author in selected_authors:
        author_id = author.get("authorId")
        if not author_id:
            profiles.append(
                {
                    "author": normalize_author(author),
                    "notable_work": build_missing_entry(
                        "No notable work available",
                        "Author ID missing from seed paper metadata.",
                    ),
                }
            )
            continue

        result = safe_tool_call(
            "get_author_papers",
            {
                "author_id": author_id,
                "paper_fields": "title,year,citations",
                "limit": 100,
            },
            fallback=[],
        )

        if isinstance(result, dict) and "_error" in result:
            profiles.append(
                {
                    "author": normalize_author(author),
                    "notable_work": build_missing_entry("No notable work available", result["_error"]),
                }
            )
            continue

        top_other = choose_other_top_paper(
            [paper for paper in as_list(result) if isinstance(paper, dict)],
            seed_title=seed_title,
        )
        if top_other is None:
            profiles.append(
                {
                    "author": normalize_author(author),
                    "notable_work": build_missing_entry(
                        "No notable work available",
                        "No other papers were returned for this author.",
                    ),
                }
            )
            continue

        profiles.append(
            {
                "author": normalize_author(author),
                "notable_work": fetch_full_paper_details(
                    extract_backfillable_id(top_other),
                    top_other,
                ),
            }
        )

    return profiles


def messages_to_chat_dicts(messages: list[AnyMessage]) -> list[dict[str, str]]:
    chat: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        chat.append({"role": role, "content": str(message.content)})
    return chat


def build_payload(paper_id: str, author_limit: int | None) -> dict[str, Any]:
    seed_paper = fetch_seed_paper(paper_id)
    references = fetch_top_references(paper_id)
    citations = fetch_recent_citations(paper_id)
    profiles = fetch_author_profiles(
        seed_paper["authors"],
        seed_title=seed_paper["title"],
        author_limit=author_limit,
    )

    return {
        "paper_id": paper_id,
        "generated_on": str(date.today()),
        "recent_citations_start": RECENT_CITATIONS_START,
        "author_profile_count": len(profiles),
        "seed_paper": seed_paper,
        "foundational_works": references,
        "recent_developments": citations,
        "author_profiles": profiles,
    }


def compact_paper_for_prompt(paper: dict[str, Any]) -> dict[str, Any]:
    author_names = [author.get("name") for author in paper.get("authors", []) if author.get("name")]
    compact_authors = author_names[:4]
    compact = {
        "title": paper.get("title"),
        "year": paper.get("year"),
        "publicationDate": paper.get("publicationDate"),
        "venue": paper.get("venue"),
        "url": paper.get("url"),
        "fieldsOfStudy": paper.get("fieldsOfStudy", []),
        "abstract": clip_text(paper.get("abstract")),
        "authors": compact_authors,
    }
    if len(author_names) > len(compact_authors):
        compact["additionalAuthorCount"] = len(author_names) - len(compact_authors)
    if paper.get("missing_data_note"):
        compact["missing_data_note"] = clip_text(paper["missing_data_note"], MAX_ERROR_NOTE_CHARS)
    return compact


def compact_payload_for_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": payload["paper_id"],
        "generated_on": payload["generated_on"],
        "recent_citations_start": payload["recent_citations_start"],
        "author_profile_count": payload["author_profile_count"],
        "seed_paper": compact_paper_for_prompt(payload["seed_paper"]),
        "foundational_works": [compact_paper_for_prompt(paper) for paper in payload["foundational_works"]],
        "recent_developments": [compact_paper_for_prompt(paper) for paper in payload["recent_developments"]],
        "author_profiles": [
            {
                "author": {
                    "name": profile["author"].get("name"),
                    "authorId": profile["author"].get("authorId"),
                },
                "notable_work": compact_paper_for_prompt(profile["notable_work"]),
            }
            for profile in payload["author_profiles"]
        ],
    }


def build_generation_prompt(payload: dict[str, Any]) -> str:
    prompt_payload = compact_payload_for_prompt(payload)
    payload_json = json.dumps(prompt_payload, indent=2, ensure_ascii=False)
    author_profile_count = payload.get("author_profile_count", 0)
    foundational_count = len(payload.get("foundational_works", []))
    developments_count = len(payload.get("recent_developments", []))
    return (
        "Generate the citation neighborhood report in markdown.\n"
        f"The 'Foundational Works' section must contain exactly {foundational_count} entries from the payload.\n"
        f"The 'Recent Developments' section must contain exactly {developments_count} entries from the payload.\n"
        f"The 'Author Profiles' section must contain exactly {author_profile_count} entries from the payload and no extra authors.\n\n"
        f"Retrieved research payload:\n```json\n{payload_json}\n```"
    )


def build_generation_messages(payload: dict[str, Any]) -> list[AnyMessage]:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_generation_prompt(payload)),
    ]


def generation_node(state: ReportState) -> ReportState:
    prompt = build_generation_prompt(state["payload"])
    messages: list[AnyMessage] = [
        *state["messages"],
        HumanMessage(content=prompt),
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_to_chat_dicts(messages),
    )
    report = response.choices[0].message.content or ""

    return {
        "payload": state["payload"],
        "messages": [HumanMessage(content=prompt), AIMessage(content=report)],
        "report_markdown": report,
    }


def build_graph():
    graph_builder = StateGraph(ReportState)
    graph_builder.add_node("generate_report", generation_node)
    graph_builder.add_edge(START, "generate_report")
    graph_builder.add_edge("generate_report", END)
    return graph_builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a citation neighborhood markdown report for a seed paper."
    )
    parser.add_argument("paper_id", help="Seed paper identifier, e.g. ARXIV:2210.03629")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output file path.",
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


def save_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def main() -> int:
    args = parse_args()
    author_limit = None if args.all_authors else args.author_limit
    graph = build_graph()

    try:
        payload = build_payload(args.paper_id, author_limit)
    except Exception as exc:
        print(f"Failed to retrieve seed paper data: {exc}", file=sys.stderr)
        return 1

    initial_state: ReportState = {
        "payload": payload,
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "report_markdown": "",
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        print(f"Failed to generate markdown report: {exc}", file=sys.stderr)
        return 1

    report = result["report_markdown"].strip()
    print(report)

    if args.output:
        save_markdown(args.output, report + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
