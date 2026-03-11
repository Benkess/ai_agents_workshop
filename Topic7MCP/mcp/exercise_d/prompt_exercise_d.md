# Prompt: Exercise D

Review [Topic7MCP/instructions/mcp_asta_lesson_plan.md](/Topic7MCP/instructions/mcp_asta_lesson_plan.md). Complete Exercise D. The instructions are copied bellow.

## instructions
### Exercise D: Citation Network Explorer Agent (15 min design + implementation)

This is not a chatbot. Build a small autonomous agent that performs a multi-step research task with no human in the loop.

**The task:** Given a seed paper by ArXiv ID, autonomously build a "citation neighborhood" and produce a structured markdown report.

**Steps the agent must perform:**

1. Retrieve full metadata for the seed paper (title, abstract, year, authors, fields of study).
2. Fetch the paper's references and retrieve abstracts for the 5 most-cited ones.
3. Fetch recent citing papers (last 3 years).
4. For each author of the seed paper, retrieve their most-cited other work.
5. Generate a structured markdown report containing:
   - A one-paragraph summary of the seed paper
   - A "Foundational Works" section with the 5 key references
   - A "Recent Developments" section with 5 citing papers
   - An "Author Profiles" section with each author's most notable other work

**Implementation notes:**

- The agent makes all MCP calls directly — no LLM deciding which to call. The LLM's role is generation only: receive all retrieved data and write the final report.
- Takes a paper ID as a command-line argument.
- Think carefully about the order of operations: which calls depend on earlier results?

**Suggested seed paper for testing:** `ARXIV:2210.03629` (ReAct: Synergizing Reasoning and Acting in Language Models)

**Deliverable:** A Python script that prints a well-formatted markdown report to stdout.

**Bonus extensions:**
- Detect if any citing paper's authors also appear in the reference list (recurring collaboration in the field)
- Accept a topic keyword instead of a paper ID: search for the most-cited paper on that topic, then run the full pipeline
- Add a "Research Gaps" section where the LLM, having seen both old references and new citations, identifies open problems

---

### Notes

1) The instructions do not use the correct tool names. See the exercises a and b for the correct names. [asta_tools](../exercise_a/asta_tools.md)

2) Your agent should use langgraph but can be minimal. No tools are needed since the MCP calls are meant to be done without the LLM. Use langgraph messages. Include a system message explaining what the agent should do (generate the markdown, no chat). Then format the MCP responses from the hardcoded MCP calls into a User message. Send that history to the model using openai api. Expect the md as a response.

3) Add an arg to save the response as an md file.

4) Do not do the Bonus extensions.

### Testing

You can run tests using the venv. 

If you want to test with api keys do this:
- Environment variables are stored in `.env` at the repo root.
- Run Python scripts with:
  `.venv/bin/python scripts/run_with_env.py <script> [args...]`
- Do not rely on shell activation persisting across commands.

or it should just work if you use `load_dotenv()`.

### extended tasks and limitations

- All your edits should be in the `Topic7MCP/mcp/exercise_d` directory.

- When you are done try running the script with "ARXIV:2210.03629" and save the output to `Topic7MCP/mcp/exercise_d/out/ARXIV_2210_03629.md`.