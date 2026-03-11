# MCP Tool Integration with Ai2 Asta

This directory is my portfolio submission for the [MCP Tool Integration with Ai2 Asta](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic7MCP/mcp_asta_lesson_plan.html) section of [Topic 7: MCP and A2A](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic7MCP/mcp.html).

## Environment Setup

If `python` on your machine points to Python 2, replace `python` with `python3` in the commands below.

```bash
cd path/to/ai_agents_workshop
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install dependencies per exercise:

- Exercise A: `python -m pip install requests`
- Exercise B: `python -m pip install requests`
- Exercise C: `python -m pip install requests python-dotenv openai langgraph`
- Exercise D: `python -m pip install requests python-dotenv openai langgraph langchain-core`

Add your API keys in `.env` at the repo root:

- `ASTA_API_KEY`
- `OPENAI_API_KEY`

The prompts for later exercises also use the repo-root helper runner:

```bash
.venv/bin/python scripts/run_with_env.py <script> [args...]
```

Deactivate the environment when done:

```bash
deactivate
```

## Table of Contents

- [exercise_a/](exercise_a/)
- [exercise_b/](exercise_b/)
- [exercise_c/](exercise_c/)
- [exercise_d/](exercise_d/)

## Exercise A: Discover the Asta Tools

Exercise A manually calls the Asta MCP endpoint with `tools/list`, parses the returned tool schemas, and formats them as either plain text or Markdown. This makes the MCP discovery step explicit before any LLM is involved.

### Launch

Run from the repository root:

```bash
python Topic7MCP/mcp/exercise_a/exercise_a.py --output Topic7MCP/mcp/exercise_a/asta_tools.md
```

You can also print the tool summary directly to the terminal:

```bash
python Topic7MCP/mcp/exercise_a/exercise_a.py
```

### Key Files

- [exercise_a/exercise_a.py](exercise_a/exercise_a.py)
- [exercise_a/starter_code.py](exercise_a/starter_code.py)
- [exercise_a/asta_tools.md](exercise_a/asta_tools.md)
- [exercise_a/exercise_a.md](exercise_a/exercise_a.md)

### Outputs

- [exercise_a/asta_tools.md](exercise_a/asta_tools.md)
- [exercise_a/out_exercise_a.txt](exercise_a/out_exercise_a.txt)

### Question Answers

To find papers about transformer attention mechanisms, the conceptual lesson-plan answer is `search_papers`, but on the live endpoint the closest equivalent is `search_papers_by_relevance`.

To find who else publishes in the same area as a specific author, the practical workflow is to identify the author with `search_authors_by_name` and then inspect their publications and collaborators with `get_author_papers`.

## Exercise B: Direct Asta Tool Calls

Exercise B uses direct MCP `tools/call` requests for three focused drills: searching for recent agent papers, retrieving citations to BERT, and exploring the ReAct paper's references. The code normalizes the different result shapes returned by each tool.

### Launch

Run from the repository root:

```bash
python Topic7MCP/mcp/exercise_b/exercise_b.py --output Topic7MCP/mcp/exercise_b/out_exercise_b.md
```

You can also print the drill outputs directly:

```bash
python Topic7MCP/mcp/exercise_b/exercise_b.py
```

### Key Files

- [exercise_b/exercise_b.py](exercise_b/exercise_b.py)
- [exercise_b/exercise_b_raw.py](exercise_b/exercise_b_raw.py)
- [exercise_b/exercise_b_save_json.py](exercise_b/exercise_b_save_json.py)
- [exercise_b/json_responses/](exercise_b/json_responses/)

### Outputs

- [exercise_b/out_exercise_b.md](exercise_b/out_exercise_b.md)
- [exercise_b/out_raw.txt](exercise_b/out_raw.txt)
- [exercise_b/json_responses/drill1_search_papers_by_relevance.json](exercise_b/json_responses/drill1_search_papers_by_relevance.json)
- [exercise_b/json_responses/drill2_get_citations.json](exercise_b/json_responses/drill2_get_citations.json)
- [exercise_b/json_responses/drill3_get_paper_references.json](exercise_b/json_responses/drill3_get_paper_references.json)

### Known Issues

The lesson plan names did not match the live Asta MCP endpoint exactly. In practice, the working tools were `search_papers_by_relevance`, `get_citations`, and `get_paper`, not `search_papers` and `get_references`.

### Question Answers

The three drills returned different JSON shapes. `search_papers_by_relevance` produced a flat list of paper objects, `get_citations` wrapped each result under `citingPaper`, and `get_paper` returned one paper object with a nested `references` list. Because of that, the script normalizes records before formatting them.

The tool response payload also required two parsing steps. First, the code unwraps the MCP transport response, including SSE-style `data:` lines when necessary. Then it prefers structured JSON when available and falls back to parsing the `content[0]["text"]` strings into Python objects.

## Exercise C: Asta-Powered Research Chatbot

Exercise C builds an interactive research chatbot that fetches tool schemas dynamically with `tools/list`, converts them into OpenAI tool definitions, and lets the model decide when to call Asta tools. The same chatbot loop works for multiple tools because the schemas come from the MCP server rather than being handwritten in the client.

### Launch

Run the chatbot from the repository root:

```bash
python Topic7MCP/mcp/exercise_c/exercise_c.py
```

To save the full chat history as JSON when exiting:

```bash
python Topic7MCP/mcp/exercise_c/exercise_c.py --save-history Topic7MCP/mcp/exercise_c/output/history.json
```

### Key Files

- [exercise_c/exercise_c.py](exercise_c/exercise_c.py)
- [exercise_c/render_history_md.py](exercise_c/render_history_md.py)
- [exercise_c/output_examples.md](exercise_c/output_examples.md)

### Outputs

- [exercise_c/output_examples.md](exercise_c/output_examples.md)
- [exercise_c/output/exercise_c_q1.txt](exercise_c/output/exercise_c_q1.txt)
- [exercise_c/output/exercise_c_q2.txt](exercise_c/output/exercise_c_q2.txt)
- [exercise_c/output/exercise_c_q3.txt](exercise_c/output/exercise_c_q3.txt)
- [exercise_c/output/exercise_c_q4.json](exercise_c/output/exercise_c_q4.json)
- [exercise_c/output/exercise_c_q4.md](exercise_c/output/exercise_c_q4.md)

### Known Issues

The tool infrastructure worked, but ambiguous title resolution still caused mistakes. In the "Summarize the references used in the ReAct paper" example, the agent picked the video-generation paper titled "ReAct: An Action-Conditioned Transformer for Interactive Video Generation" instead of "ReAct: Synergizing Reasoning and Acting in Language Models."

### Question Answers

Compared to Exercise B, Exercise C moved schema ownership from the client into the MCP server. The chatbot no longer needs tool-specific schema definitions for each Asta capability, so the same loop can work with newly added tools without rewriting the client.

The tradeoff is that the main risks shift from hand-written plumbing to model behavior. The agent can still choose the wrong tool, resolve an ambiguous paper incorrectly, or use a valid tool sequence that answers the wrong interpretation of the user's question.

## Exercise D: Citation Network Explorer Agent

Exercise D is not a chatbot. It is a small autonomous pipeline that retrieves a citation neighborhood for a seed paper, builds a structured payload from MCP calls, and then uses the LLM only for final markdown generation. The retrieval order is hardcoded in the program rather than chosen by the model.

### Launch

Run from the repository root with the helper script and save the generated report:

```bash
.venv/bin/python scripts/run_with_env.py Topic7MCP/mcp/exercise_d/exercise_d.py ARXIV:2210.03629 --all-authors --output Topic7MCP/mcp/exercise_d/out/ARXIV_2210_03629.md
```

You can also inspect the prompt payload used for generation:

```bash
.venv/bin/python scripts/run_with_env.py Topic7MCP/mcp/exercise_d/exercise_d_prompt_debug.py ARXIV:2210.03629 --all-authors --output Topic7MCP/mcp/exercise_d/out/prompt_debug_all_authors.md
```

### Key Files

- [exercise_d/exercise_d.py](exercise_d/exercise_d.py)
- [exercise_d/exercise_d_prompt_debug.py](exercise_d/exercise_d_prompt_debug.py)
- [exercise_d/prompt_exercise_d.md](exercise_d/prompt_exercise_d.md)

### Outputs

- [exercise_d/out/ARXIV_2210_03629.md](exercise_d/out/ARXIV_2210_03629.md)
- [exercise_d/out/prompt_debug_all_authors.md](exercise_d/out/prompt_debug_all_authors.md)

### Known Issues

The generation prompt is large, with the all-authors debug capture showing a combined prompt size of about 7,024 tokens. Some recent citing papers also arrive without abstracts, so the report quality depends on how well the retrieval payload is curated and clipped before generation.

### Question Answers

This agent keeps the tool-calling order deterministic: fetch the seed paper, expand references, fetch recent citations, gather author profiles, then ask the LLM to write the report. That makes failures easier to reason about, but it also means the system is only as good as the handcrafted retrieval plan.

Because the LLM is generation-only here, most of the engineering work is in deciding what evidence to include. The script trims long text, normalizes missing fields, and backfills metadata where possible so the final markdown prompt stays informative without becoming unmanageably large.

## Closing Discussion

MCP automation removes the need to hand-author and maintain tool schemas in every client. That makes agents more adaptable to server-side changes and lets one generic tool loop support many capabilities. The cost is extra protocol complexity and new failure modes around schema discovery, transport parsing, and model tool selection.

When the Asta tools returned rich JSON, I kept the fields that were useful for answering the task and dropped or clipped the rest. Passing everything through would waste tokens, bury the important facts, and make the model more likely to latch onto irrelevant details. The Exercise D payload shows the practical middle ground: normalize records, keep the fields that support the report, and truncate long text aggressively.

If the LLM were allowed to decide tool order in Exercise D, it would need access to the tool schemas, a multi-step execution loop, and guardrails for partial failures and stopping conditions. The main risks would be redundant calls, missing a required dependency between steps, selecting the wrong paper or author, or spending context budget on unnecessary data collection.

What I would want from a more mature MCP ecosystem is better discovery and observability: reliable registries, stronger debugging tools, clearer transport/error diagnostics, better support for auth and capability negotiation, and standardized tracing so it is easier to inspect what the client asked for, what the server returned, and why the model made a given tool decision.
