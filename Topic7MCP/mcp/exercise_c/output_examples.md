# Exercise C Output Examples

Captured from live runs on March 10, 2026 using [exercise_c.py](/home/user/projects/school/ai_agents_workshop/Topic7MCP/mcp/exercise_c/exercise_c.py).

## System Prompt

```text
You are a research assistant with access to Semantic Scholar tools. Use tools when appropriate to answer questions about research papers, authors, citations, and references.
```

## Query 1

**User**

```text
Find recent papers about large language model agents
```

**Tool trace**

```text
Calling tool: search_papers_by_relevance
Arguments: {
  "keyword": "large language model agents",
  "limit": 5,
  "fields": "title,url,year,authors"
}
```

**Assistant**

```text
Here are some recent papers about large language model agents:

1. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents (2024)
2. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning (2025)
3. A Survey of Large Language Model Agents for Question Answering (2025)
4. Emergence of human-like polarization among large language model agents (2025)
5. PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time (2025)
```

## Query 2

**User**

```text
Who wrote Attention is All You Need and what else have they published?
```

**Tool trace**

```text
Calling tool: search_paper_by_title
Arguments: {
  "title": "Attention is All You Need",
  "fields": "url,year,authors"
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "40348417",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "1846258",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "3877127",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "39328010",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "145024664",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "19177000",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "40527594",
  "paper_fields": "url,year,authors",
  "limit": 5
}

Calling tool: get_author_papers
Arguments: {
  "author_id": "3443442",
  "paper_fields": "url,year,authors",
  "limit": 5
}
```

**Assistant**

```text
The paper "Attention is All You Need" was co-authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and I. Polosukhin.

The chatbot then listed example publications for each author. Representative examples from the run:

- Ashish Vaswani: "Adversarial Training on Tactics Data" (2024), "Quantum-enhanced samples of random quantum states at two temperatures" (2023)
- Noam Shazeer: "Fast training of Transformers" (2025), "Advances in Language Modelling" (2021)
- Niki Parmar: "A Systematic Survey of Entity Extraction from Text" (2021), "The importance of new data in improving clinical text classification" (2024)
```

## Query 3

**User**

```text
What papers cite the original BERT paper?
```

**Tool trace**

```text
Calling tool: get_citations
Arguments: {
  "paper_id": "DOI:10.18653/v1/N19-1423",
  "fields": "title,authors,url",
  "limit": 10
}
```

**Assistant**

```text
Here are some papers that cite the original BERT paper:

1. Enhancing large language models for knowledge graph question answering via multi-granularity knowledge injection and structured reasoning path-augmented prompting
2. Multi-view dynamic perception framework for Chinese harmful meme detection
3. SEGA: Selective cross-lingual representation via sparse guided attention for low-resource multilingual named entity recognition
4. Validating generative agent-Based modeling in social media simulations through the lens of the friendship paradox
5. SRSPSQL: A dual-stage Text-to-SQL framework with semantic rewriting and schema pruning
6. A model for early propagation of derivative adversarial topics based on emotional transfer and evolutionary game theory
7. Risk-informed design of Software as a Medical Device through Natural Language Processing techniques
8. Vuln2Action: An LLM-based framework for generating vulnerability reproduction steps and mapping exploits
9. From patents to predictive analytics: Leveraging R-GCNs for technological opportunity discovery in converging industries
10. Privacy-preserving transfer learning framework for building energy forecasting with fully anonymized data
```

## Query 4

**User**

```text
Summarize the references used in the ReAct paper
```

**Tool trace**

```text
Calling tool: search_paper_by_title
Arguments: {
  "title": "ReAct: An Action-Conditioned Transformer for Interactive Video Generation",
  "fields": "references"
}

Calling tool: search_papers_by_relevance
Arguments: {
  "keyword": "ReAct: An Action-Conditioned Transformer for Interactive Video Generation",
  "fields": "url"
}

Calling tool: get_paper
Arguments: {
  "paper_id": "283f9baf525e4973fed3a23136f68216b6f8d790",
  "fields": "references"
}
```

**Assistant**

```text
The run resolved "ReAct paper" to "ReAct: An Action-Conditioned Transformer for Interactive Video Generation" rather than the language-model paper "ReAct: Synergizing Reasoning and Acting in Language Models."

It then summarized references focused on video generation, diffusion models, and interactive simulation, including:
- Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion
- GameGen-X: Interactive Open-world Game Video Generation
- Diffusion Models Are Real-Time Game Engines
- CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
- Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
```

**Observation**

```text
This query shows a real limitation of the current agent: with an ambiguous title, the model may select the wrong paper even though the tool infrastructure works correctly.
```

## Discussion

Compared to Exercise B, Exercise C moved the tool knowledge out of the client and into the MCP server. In Exercise B, the code knew specific tool names, specific argument shapes, and how each drill should be routed. In Exercise C, the client starts by calling `tools/list`, converts the returned schemas into OpenAI function definitions, and then lets the model decide which tool to call and with what arguments.

That changes the implementation in two important ways. First, there is almost no tool-specific code in the chatbot itself. The same request loop works for `search_papers_by_relevance`, `get_citations`, `get_author_papers`, or any new Asta tool that appears tomorrow, because the interface is discovered dynamically from MCP. Second, the new failure mode is no longer just "did my hardcoded tool function work"; it also includes schema discovery, tool selection quality, and ambiguity in model reasoning. The ReAct example shows that clearly: the MCP plumbing worked, but the model chose the wrong paper because the prompt was ambiguous.

The core value of MCP is that tool definitions are standardized and server-owned. The client does not need to manually author or maintain per-tool schemas, so adding or changing tools on the server side does not require rewriting the chatbot loop.
