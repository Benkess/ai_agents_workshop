# TASK PROMPT: exercise 1

## LLM Generated Task Description

Implement **Exercise 1: Vision-Language LangGraph Chat Agent** in the directory:

**`Topic6VLM/exercise_1/`**

Use **Python**, **Gradio** for the web UI, **LangGraph** for the agent/state machine, and **LangChain message objects** for conversation history. The backend model is **Ollama `llava`**.

### What the app must do (user experience)

1. The web page has:

   * An **image upload** component.
   * A **chat interface** (chat history window + textbox).
   * A **Send** button (or Enter-to-send).

2. The workflow:

   * User uploads an image.
   * User sends the **first message**, and that first message **includes the image**.
   * After that, user continues chatting with the model about the same image.
   * The model responses appear in the chat.

3. Constraints:

   * **Message History:** Keep full message history in memory for the session. Later we will add a sliding context window on all messages except the system prompt and user message with the image.
   * Still use **good LangGraph style**: typed state, clear nodes, minimal custom formatting, no manual role hacks.

### Backend requirements (LangGraph + messages)

* Use a LangGraph `StateGraph` where the state includes:

  * `messages`: a list of LangChain messages (e.g., `HumanMessage`, `AIMessage`, possibly `SystemMessage`) using the `add_messages` reducer.
  * `image_path` (or `image_b64`) stored after upload so it can be reused across turns.
  * Any other minimal metadata you need (e.g., `has_image: bool`).

* Use LangChain-style messages as the single source of truth for chat history.

  * Do **NOT** manually prefix messages like `"Human:"` / `"Assistant:"`.
  * Do **NOT** manually concatenate history into one string prompt.

* Model call:

  * Use `ollama.chat(model="llava", messages=[...])`.
  * For the **first** user message only, include the image via the `images` field (list containing either a file path or base64 string).
  * For subsequent turns, omit the `images` field but preserve chat context (messages).

* System prompt:

  * Include a `SystemMessage` that instructs the assistant to answer questions about the uploaded image, be concise, ask clarifying questions when needed, and explicitly say when it can’t determine something from the image.

### Gradio UI requirements

* Use Gradio `Blocks`.
* Keep per-user session state using `gr.State()` (store at minimum: uploaded image path + a `thread_id` or some stable session identifier).
* Chat component: `gr.Chatbot()`.
* Text input: `gr.Textbox()`.
* Image input: `gr.Image(type="filepath")` (preferred, simplest for `ollama`).
* When user uploads a new image:

  * Clear chat history and reset LangGraph state for that session.

### File structure to create inside `Topic6VLM/exercise_1/`

Create at least:

* `app.py` — Gradio UI + event handlers.
* `agent.py` — LangGraph graph, state definition, model call node(s).
* `requirements.txt` — minimal deps to run.
* `README.md` — run instructions + brief explanation.

Optional but nice:

* `utils.py` for image resizing/compression helper.

### Performance requirement (image resizing)

Most models will resize and format image internally. For now there is not need to alter the image being sent to the model. Most likely it will need to be packaged as expected by langgaph and the model api.
* Document this behavior in README.

### How turns should work (important)

* First user turn:

  * Requires that an image is uploaded.
  * The LangGraph state should store `image_path`.
  * The ollama call includes the image.

* Later turns:

  * Use the stored `image_path` only if the model needs it, but **do not attach the image again** unless you decide it’s required.
  * Continue multi-turn chat using the accumulating message list.

### Implementation details (be explicit)

* Use a typed state, e.g. `TypedDict`:

  * `messages: Annotated[list, add_messages]`
  * `image_path: str | None`

* Graph:

  * A single node `vlm_chat_node(state) -> {"messages":[AIMessage(...)]}` is fine.
  * Compile once and reuse.

* Error handling:

  * If user sends a message before uploading an image, respond in chat with a helpful error.
  * If Ollama is not running or `llava` isn’t pulled, show a helpful message.

### README must include

* Install:

  * `pip install -r requirements.txt`
* Ollama:

  * `ollama pull llava`
  * “Make sure Ollama is running”
* Run:

  * `python app.py`
* Short description of architecture (Gradio → LangGraph → Ollama).

### Acceptance checklist (you must satisfy)

* [ ] Runs as a Gradio web app
* [ ] Image upload then first message includes image
* [ ] Multi-turn chat continues about the image
* [ ] LangGraph state management is used
* [ ] LangChain message objects are used (not ad-hoc dicts only)
* [ ] No summarization/context trimming yet
* [ ] Clear file layout and README

Now implement the code in `Topic6VLM/exercise_1/` accordingly.

## References:
### Task specific context:
- [/Topic6VLM/README.md](/Topic6VLM/README.md): The unfinished readme for Topic6VLM. No need to edit this but it has info on setup that has been repeated for all units.
- [/Topic6VLM/ollama_tests/ollama_llava.py](/Topic6VLM/ollama_tests/ollama_llava.py): A working example of prompting ollama lava.
- [/Topic6VLM/instructions/vlm.md](/Topic6VLM/instructions/vlm.md): The class module instructions copied into an md file. You are working on "Exercise 1: Vision-Language LangGraph Chat Agent". This is the most important.
- [/Topic6VLM/instructions/vlm_guide.md](/Topic6VLM/instructions/vlm_guide.md): Coppied from the clas website. Notes on VLM ussage.

### Example AI Agents from other modules:
- [/Topic2Frameworks/step7_checkpointing.py](/Topic2Frameworks/step7_checkpointing.py): Example of langgraph and langchain usage. Might be confusing because there are two models.
- [/Topic3Tools/langgraph_tools/5_langgraph_framework.py](/Topic3Tools/langgraph_tools/5_langgraph_framework.py): langgraph example with tool use.
- [/Topic3Tools/readme.md](/Topic3Tools/readme.md): Readme with usefull examples

### Exampl AI Agents from other projects:
- [/topic_6/examples/agents/core/assistant/open_ai_assistant.py](/topic_6/examples/agents/core/assistant/open_ai_assistant.py): open ai api chatbot with langgraph and tool use in cli.
- [/topic_6/examples/agents/core/assistant/qwen_assistant_clean.py](/topic_6/examples/agents/core/assistant/qwen_assistant_clean.py): Qwen cli chatbot, local.
- [/topic_6/examples/computer_use/agent_message/computer_agent_v4.py](/topic_6/examples/computer_use/agent_message/computer_agent_v4.py): Computer use agent using qwen. Included because it collects images and sends them to the model. May be good VLM example. Bad context management from my memory. I think it is good langgraph example.

## Notes from human:

- Make sure to use the proper langgraph and lang chain structures. Also make sure to use the models api and ollama to avoid manual code such as formating messages or parseing responses.  Most models have internal functions to ensure inputs and outputs are handled correctly. Using the the models API's inter functions and langgraph will avoid issues that might happen when doing this manually.
- Make this modular. I would love it if i could easily swap in a different agent class for running with another model like qwen. No need to add code for another model yet.
- The LLM task description might have errors. Ask questions.
- The provided references might have errors. Ask questions.


---
