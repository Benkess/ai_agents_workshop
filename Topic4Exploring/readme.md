# Topic 4: Exploring Tools
This repo is part of my portfolio for the UVA Agentic AI course. This is my submission for [Topic 4: Exploring Tools](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic4Exploring/exploring.html).

## Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/Topic4Exploring
```
Use the venv module to create a new virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
# On macOS/Linux.
source .venv/bin/activate
```
```bash
# On Windows.
.venv\Scripts\activate
```

Upgrade pip:
```bash
python -m pip install --upgrade pip
```
```bash
# Also upgrade these for TensorFlow or PyTorch
python -m pip install --upgrade pip wheel setuptools
```

Use pip to install all the packages listed in your requirements.txt:
```bash
pip install -r requirements.txt
```

Install the correct PyTorch version:
```bash
pip3 install torch torchvision
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Install PyTorch first (CUDA 13.0)
# python -m pip install torch torchvision `
#   --index-url https://download.pytorch.org/whl/cu130 `
#   --extra-index-url https://pypi.org/simple `
#   --timeout 120
```
> **Note:** 
> For specific PyTorch installs see [get-started](https://pytorch.org/get-started/locally/).
> See your CUDA version with `nvidia-smi`.

When you are done working in the virtual environment, you can deactivate it.
```bash
deactivate
```

## Resorces:
- [Lang Graph](https://docs.langchain.com/oss/python/langgraph/graph-api)

---

## Tree

```text
Topic4Exploring
│   readme.md
│
├───output
│       langchain_conversation_graph.png
│       langchain_manual_tool_graph.png
│       langchain_react_agent.png
│
├───part3
│       react_agent_example.py
│       toolnode_example.py
│
└───part5
        doc.md
        readme.md
        topic4_youtube_video_analyzer.py
        youtube_tool_test.py
```

---

## Table of contents
### Part 3: Tool Node vs React Agent
**Files:**
- [Topic4Exploring\part3\react_agent_example.py](): LangGraph ReAct Agent with Persistent Multi-Turn Conversation
- [Topic4Exploring\part3\toolnode_example.py](): Manual ToolNode Implementation
- [Topic4Exploring\output\langchain_conversation_graph.png](): Mermaid from react agent showing Conversation loop wrapper.
- [Topic4Exploring\output\langchain_manual_tool_graph.png](): Mermaid from Manual tool calling agent
- [Topic4Exploring\output\langchain_react_agent.png](): Mermaid from react agent showing ReAct agent (thought/action/observation)

**Questions:**
- What features of Python does ToolNode use to dispatch tools in parallel?  What kinds of tools would most benefit from parallel dispatch?
  - Answer: It uses `asyncio` to dispatch tools in parallel. This might be usefull for search tools where multiple searches could run at the same time.
- How do the two programs handle special inputs such as "verbose" and "exit"?
  - Answer: Both the `react_agent_example.py` and `toolnode_example.py` uses an input node that parses the user input and routes to the special location if there is a special input.
- Compare the graph diagrams of the two programs.  How do they differ if at all?
  - Answer: The `react_agent_example.py` has a sub graph that contains the agent and tool nodes and a wrapper graph for everything else. The `toolnode_example.py` has all the nodes on the same graph.
- What is an example of a case where the structure imposed by the LangChain react agent is too restrictive and you'd want to pursue the toolnode approach?  
  - Answer: The react approach does not allow for branching based on tool results or running multiple actions in parallel. If we wanted to have the agent run multiple tools then depending on the tool branch to an aggrigate results node then return them to the model, react would not work.

### Part 5: 2-Hour Agent Projects
**Files:**
- [Topic4Exploring\part5\topic4_youtube_video_analyzer.py](/Topic4Exploring/part5/topic4_youtube_video_analyzer.py): The code for the youtube transcript agent.
- [readme](/Topic4Exploring/part5/readme.md): Description and Usage Instructions.


**Description:**
I did "Option 3: YouTube Transcript" and made an agent that gives summaries of educational videos.
---
