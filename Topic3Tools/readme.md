# Topic 3: Agent Orchestration Frameworks
This repo is part of my portfolio for the UVA Agentic AI course. This is my submission for [Topic 3: Agent Tool Use](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic3Tools/tools.html).

## Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/Topic3Tools
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
Topic3Tools
│   OpenAI_GPT4o_Mini_Test.py
│   readme.md
│   requirements.txt
│   requirements_explicit.txt
│
├───langgraph_tools
│   │   4_calc.py
│   │   4_langgraph_tools.py
│   │   4_letter_count.py
│   │   5_langgraph_framework.py
│   │   langgraph-tool-handling.py
│   │
│   └───__pycache__
│           5_langgraph_framework.cpython-312.pyc
│
├───manual-tool-handling
│       3_calculator_tool.py
│       manual-tool-handling.py
│
├───ollama
│       llama_mmlu_eval.py
│       llama_mmlu_eval_ollama.py
│       ollama_test.py
│
└───output
        3_calc.txt
        3_startercode.txt
        4_letter_count.txt
        4_startercode.txt
        4_tools.txt
        4_use_all_tools.txt
        5.txt
        5_lg_graph.png
```

---

## Table of contents

### 1: Timing Ollama
- [Topic3Tools\ollama\llama_mmlu_eval_ollama.py](): Code for part 1.

Discussion: 
I observed that it was incredably slow. I had to cut it off because it was taking to long. I am not sure where i messed up. I was running locally and it would get to like 20 minutes.

### 2: API setup 
```bash
# set up a client to communicate using oOpenAI's API
client = OpenAI()
# saves the response from the request sent using the open ai client
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: Working!"}],
 max_tokens=5
```
### 3: Manual tool handling
- [/Topic3Tools/manual-tool-handling/3_calculator_tool.py](/Topic3Tools/manual-tool-handling/3_calculator_tool.py): code for question 3.
- [/Topic3Tools/output/3_calc.txt](/Topic3Tools/output/3_calc.txt): Output from calculator tool.
- [/Topic3Tools/output/3_startercode.txt](/Topic3Tools/output/3_startercode.txt): Starter Code Output.

### 4: LangGraph tool handling
- [Topic3Tools\langgraph_tools\4_langgraph_tools.py](): Full code for using all tools with langgraph.
- [Topic3Tools\output\4_tools.txt](): Out puts from each tool.
- [Topic3Tools\output\4_use_all_tools.txt](): Uses all 3 tools in one prompt.

Discussion:
I was able to make a querry that used all 3 tools. I also was able to hit the turn limit by just asking it too use the tools repeatedly. 

### 5: LangGraph Conversation Framework
- [Topic3Tools\langgraph_tools\5_langgraph_framework.py](): Code for part 5.
- [Topic3Tools\output\5.txt](): A conversation that demonstrates tool use, the conversation context, and recovery.
- [Topic3Tools\output\5_lg_graph.png](): Mermaid diagram of the system.

### 6: Question
Question: where is there an opportunity for parallelization in your agent that is not yet being taken advantage of?

Answer: I feel like the speaking tool could run parallel to the agent since the response is not needed to continue. This could allow for speaking to happen while the user is typing their next prompt. Maybe submitting the next prompt will interupt the tool. 

---
