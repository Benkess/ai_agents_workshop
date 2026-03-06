# Task: Complete exercise 1

You are provided with demo code for gradio wrapper around a minimal langgraph agent. You must edit it to meet the requirments of Topic6VLM exercise 1. Your should only make changes within the `Topic6VLM` directory.

## References:
### Task specific context:
- [/Topic6VLM/README.md](/Topic6VLM/README.md): The unfinished readme for Topic6VLM. No need to edit this but it has info on setup that has been repeated for all units.
- [/Topic6VLM/ollama_tests/ollama_llava.py](/Topic6VLM/ollama_tests/ollama_llava.py): A working example of prompting ollama lava.
- [/Topic6VLM/instructions/vlm.md](/Topic6VLM/instructions/vlm.md): The class module instructions copied into an md file. You are working on "Exercise 1: Vision-Language LangGraph Chat Agent". This is the most important.
- [/Topic6VLM/instructions/vlm_guide.md](/Topic6VLM/instructions/vlm_guide.md): Coppied from the class website. Notes on VLM ussage.
- [/topic_6/Topic6VLM/instructions/gradio_quickstart.py](/topic_6/Topic6VLM/instructions/gradio_quickstart.py): Coppied from the class website. Example gradio usage.

### Example AI Agents from other modules:
- [/Topic2Frameworks/step7_checkpointing.py](/Topic2Frameworks/step7_checkpointing.py): Example of langgraph and langchain usage. Might be confusing because there are two models.
- [/Topic3Tools/langgraph_tools/5_langgraph_framework.py](/Topic3Tools/langgraph_tools/5_langgraph_framework.py): langgraph example with tool use.
- [/Topic3Tools/readme.md](/Topic3Tools/readme.md): Readme with usefull examples

### Exampl AI Agents from other projects:
- [/demos/gradio/chatbot/README.md](/demos/gradio/chatbot/README.md): The demo I coppied into `Topic6VLM\exercise_1` as starter code.
- [/demos/image_use/agent_state_obs_api/agent_state_obs_api_agent/agents/open_ai_agent.py](/demos/image_use/agent_state_obs_api/agent_state_obs_api_agent/agents/open_ai_agent.py): An example of how the openai api can be used to send images to ollama hosted models.
- [/demos/image_use/agent_state_obs_api/README.md](/demos/image_use/agent_state_obs_api/README.md): A demo program that can already send images to ollama hosted models using the openai api.
- [/demos/image_use/agent_state_obs_api/agent_state_obs_api_server/config/qwen_server.json](/demos/image_use/agent_state_obs_api/agent_state_obs_api_server/config/qwen_server.json): The config used to setup the demo to interface with qwen hosted locally on ollama. 

## Description

I have included the assignment instruction in my references. We are going to make a gradio app that allows the user to chat about an image they uploaded. I have already coppied example code into `Topic6VLM\exercise_1`. You are to copy the pattern in [/demos/image_use/agent_state_obs_api/agent_state_obs_api_agent/agents/open_ai_agent.py](/demos/image_use/agent_state_obs_api/agent_state_obs_api_agent/agents/open_ai_agent.py) so that [/exercise_1/Topic6VLM/exercise_1/agent.py](/exercise_1/Topic6VLM/exercise_1/agent.py) supports sending images. 

### Agent changes

The edited agent will now save two messages objects. In the state there will be `base_messages` and `chat_messages` objects storing the langgraph message types. 

The `base_messages` will have the system prompt and a multimodal user message. The default system prompt is "You are an ai assistant. You will be give an image to discuss with the user". The human message is auto generated. It will contain context blocks with the user uploaded image and text that by default says "This message contains the image to discuss with the user.". 

The `chat_messages` object is basically the current `messages: Annotated[list[AnyMessage], add_messages]` field. It will store the chat history and have a sliding window context management. This is already done in the starter code. 

The two messages objects should be combined (with `chat_messages` coming after `base_messages`) before sending to the model. The `Exact Model Context` pannel in gradio will display the full context passed to the model so I can confirm it is structured correctly.

As mentioned I have provided an example of how the openai api can be used to send multimodal messages and interface with local ollama. Edit the configs to contain the following: 
- An Ollama llava agent as requested by the exercise instructions, a openai gpt-5.2 agent, an ollama qwen3-vl agent.

### Gradio changes

After the start screen, the user is brought to an image upload screen. Here they must upload the image that will be used for the chat. This is the image to be included in `base_messages`. The user will press a next button to get to the chat window. The chat window still has all the same features. 

---
