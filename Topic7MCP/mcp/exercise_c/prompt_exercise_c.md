# Prompt: Exercise C

Review [Topic7MCP/instructions/mcp_asta_lesson_plan.md](/Topic7MCP/instructions/mcp_asta_lesson_plan.md). Complete Exercise C. The instructions are copied bellow.

## instructions
### Exercise C: Asta-Powered Research Chatbot (20 min)

Build a chatbot that fetches tool schemas dynamically from the MCP server at startup and uses GPT-4o mini to decide which tools to call.

**Architecture:**

1. At startup, call `tools/list` and retrieve all Asta tool schemas.
2. Convert MCP schemas to OpenAI function-calling format (the mapping is direct — MCP input schemas are already valid JSON Schema).
3. When GPT-4o mini emits a `tool_calls` response, extract the name and arguments.
4. POST a `tools/call` request to Asta, extract the result, and append it as a `tool` message.
5. Continue the loop until the model produces a final text answer.

**Code structure:**
```python
def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI format."""
    # Call tools/list, then transform each tool:
    # MCP: { name, description, inputSchema }
    # OpenAI: { type: "function", function: { name, description, parameters } }

def call_asta_tool(name, arguments):
    """Execute a tools/call and return the text content."""

def chat(user_message, messages, tools):
    """One turn of the chatbot loop, handling tool calls."""
```

**Test queries — these should each trigger a different tool or chain:**

- *"Find recent papers about large language model agents"* → `search_papers`
- *"Who wrote Attention is All You Need and what else have they published?"* → `search_papers` → `get_paper` → `get_author_papers`
- *"What papers cite the original BERT paper?"* → `get_citations`
- *"Summarize the references used in the ReAct paper"* → `get_references`

**Required features:**
- A system prompt identifying the assistant as a research tool with Semantic Scholar access
- Print which tool is being called and with what arguments (so you can observe the model's decisions)
- Graceful error handling: if an MCP call fails, return the error as the tool result so the model can acknowledge it

> **Discussion:** What changed compared to calling tools manually in Exercise B? You wrote almost no tool-specific code — the schema came from the server. The chatbot would work identically if Asta added new tools tomorrow. That is the core value of MCP.

---

### Notes

1) The instructions do not use the correct tool names. See the exercises a and b for the correct names. [asta_tools](../exercise_a/asta_tools.md)

2) You should model the agent after the following example:
    - [Tool use example open ai agent](../../../demos/tool_use/cli/open_ai_assistant.py)

### Testing

You can run tests using the venv. 

If you want to test with api keys do this:
- Environment variables are stored in `.env` at the repo root.
- Run Python scripts with:
  `.venv/bin/python scripts/run_with_env.py <script> [args...]`
- Do not rely on shell activation persisting across commands.

### extended tasks and limitations

- All your edits should be in the `Topic7MCP/mcp/exercise_c` directory.

- Add documentation. If you are able to run the file the tests then answer the disscussion questions.