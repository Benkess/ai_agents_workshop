# MCP Exercise B

## Discussion after drills:

1) What differences did you notice in the structure of results across the three tools? 

    Answer: The tools return different shapes. `search_papers_by_relevance` returns a flat list of paper objects (title/year/authors). `get_citations` returns a list where each item wraps the actual paper under `citingPaper`, so you must unwrap that key. `get_paper` returns one paper object with a nested `references` list, and many reference entries only include `paperId` + `title` (often no `year`), which is why many outputs show "Unknown year". I also observed that the lesson-plan names `search_papers` and `get_references` are not available on this live endpoint.

2) How did you handle the JSON returned inside the content[0]["text"] field?

    Answer: I parse the SSE wrapper first (`event: message` / `data:`), decode the JSON-RPC object, then prefer `result.structuredContent.result` when present because it already contains full structured data. If `structuredContent` is missing, I parse each `result.content[i].text` string with `json.loads(...)` and normalize the records (including unwrapping `citingPaper` for citations and reading `references` from `get_paper`).
