# Agentic AI Course Toppic 1
This repo is part of my portfolio for the UVA Agentic AI course. This is my submission for Topic 1: Running an LLM. 

## Enviroment Setup
> **Note:** if `python` does not default to `python3` on your system, then substitute `python3` for `python` in the following commands.

Navigate to the root directory:
```bash
cd path/to/topic1
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
```
> **Note:** 
> For specific PyTorch installs see [get-started](https://pytorch.org/get-started/locally/).

When you are done working in the virtual environment, you can deactivate it.
```bash
deactivate
```

## Comparing LLMs on the MMLU benchmark
```bash
python llama_mmlu_eval.py -h
```
```
======================================================================
Llama 3.2-1B MMLU Evaluation (Quantized)
======================================================================

usage: llama_mmlu_eval.py [-h] [--model-name MODEL_NAME] [--use-gpu | --no-gpu] [--max-new-tokens MAX_NEW_TOKENS]
                          [--quantization-bits {4,8,none}] [--print-answers]

Llama MMLU evaluation (launch arguments)

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Hugging Face model name (default: meta-llama/Llama-3.2-1B-Instruct)
  --use-gpu             Enable GPU if available (default)
  --no-gpu              Force CPU execution
  --max-new-tokens MAX_NEW_TOKENS
                        Max new tokens for generation (default: 1)
  --quantization-bits {4,8,none}
                        Quantization bits: 4, 8, or none (default: None)
  --print-answers       Print each question, the model's answer, and whether it was correct
```
## Simple Chat Agent
```bash
python simple_chat_agent.py -h
```
```
usage: simple_chat_agent.py [-h] [--max-context MAX_CONTEXT] [--no-history]

Simple chat agent with summarization/dump helpers

options:
  -h, --help            show this help message and exit
  --max-context MAX_CONTEXT
                        Override model max context
  --no-history          Disable storing conversation history (use only system prompt and current user message)
```
## Results
A summary of the results is provided in [results_summary](/Running%20An%20LLM.md). For pdf format see [Running An LLM.pdf](/Running%20An%20LLM.pdf)