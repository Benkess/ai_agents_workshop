# Prompt

Review [Topic7MCP/instructions/mcp_asta_lesson_plan.md](/Topic7MCP/instructions/mcp_asta_lesson_plan.md). Complete Exercise B. The instructions are copied bellow.

## instructions
### Exercise B: Direct Asta Tool Calls — Three Focused Drills (15 min)

Before building a full chatbot, call three specific Asta tools directly. Each drill uses a different tool and teaches a different access pattern.

**Drill 1 — `search_papers`: Find recent LLM agent papers**

Write a function that calls `search_papers` with the query `"large language model agents"`, requests the fields `title,abstract,year,authors`, and prints the top 5 results as a numbered list with title and year.

```python
payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_papers",
        "arguments": {
            "query": "large language model agents",
            "fields": "title,abstract,year,authors",
            "limit": 5
        }
    }
}
result = requests.post(url, headers=headers, json=payload).json()
content = result["result"]["content"][0]["text"]
# Parse content (JSON string) and print each paper
```

**Drill 2 — `get_citations`: Trace impact of a landmark paper**

Call `get_citations` for the original BERT paper (`ARXIV:1810.04805`), filtered to papers from 2023 onward. Print the count of results and the titles of the first 5 citing papers.

```python
"arguments": {
    "paper_id": "ARXIV:1810.04805",
    "fields": "title,year,authors",
    "limit": 10,
    "publication_date_range": "2023-01-01:"
}
```

**Drill 3 — `get_references`: Understand a paper's intellectual foundation**

Call `get_references` for the ReAct paper (`ARXIV:2210.03629`) and print the titles and years of its references, sorted by year ascending. This gives you a snapshot of the intellectual lineage of the paper.

> **Discussion after drills:** What differences did you notice in the structure of results across the three tools? How did you handle the JSON returned inside the `content[0]["text"]` field?

---

### Testing

I will test the code when you are done. Give me instrucitons.