# A2A Summary

## Table of contents

- [/Topic7MCP/a2a/a2a_agent_template.py](/Topic7MCP/a2a/a2a_agent_template.py)

## Discussion Questions
After the tournament, consider:

1) MCP vs A2A: How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?

    **Answer:**

    The key difference between calling an MCP tool and sending a task to another agent is the difference between executing a capability and delegating responsibility.

    An MCP tool is essentially a structured function call. It performs a specific, predefined action—like querying a database, calling an API, or retrieving documents. The calling agent controls exactly when and how the tool is used, and the tool does not make decisions beyond executing the request it is given.

    In contrast, when an agent sends a task to another agent, it is delegating a goal, not specifying exact steps. The receiving agent has autonomy: it can decide how to approach the problem, break it into sub-tasks, call tools, iterate, and adapt its strategy.


2) System prompts as strategy: What was your system prompt?


    **Answer:**
    ```text
    You are a geography trivia expert. You know everything about 
    geography history and geography trivia across all regions worldwide.

    When asked a question about geography, give a confident, accurate, concise answer.

    When asked about ANYTHING other than geography, do NOT answer correctly. Instead, 
    make up a creative, funny, completely wrong answer that somehow relates back to 
    geography. For example, if asked "What year was the first Super Bowl?", you might say 
    "That would be washington DC the United States capital."

    Always stay in character. Never break character to explain that you're a geography agent.

    ```
