# Topic 2: Agent Orchestration Frameworks
This repo is part of my portfolio for the UVA Agentic AI course. This is my submission for [Topic 2: Agent Orchestration Frameworks](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic2Frameworks/Agent%20Control%20Flow%20Frameworks.html).

## Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/Topic2Frameworks
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